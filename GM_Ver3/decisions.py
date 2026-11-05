"""OOF ensemble fitting, group allocation, and settlement calibration."""

import lightgbm as lgb
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SEED = 20261106
ENSEMBLES = ("mean", "weighted", "ridge", "nonlinear", "mlp")


def group_sum(values, first):
    return np.stack([values[:, first].sum(axis=1, dtype=np.float64),
                     values[:, ~first].sum(axis=1, dtype=np.float64)], axis=-1)


def penalty(residuals, q, first):
    return float(0.1 * q @ np.abs(group_sum(residuals, first)).sum(axis=1))


def choose_experts(history, y, q, first):
    losses = {name: penalty(pred - y, q, first) for name, pred in history.items()}
    chosen = []
    vectors = []
    for name in sorted(losses, key=losses.get):
        residuals = group_sum(history[name] - y, first).ravel()
        centered = residuals - residuals.mean()
        vector = centered / max(1e-12, np.linalg.norm(centered))
        if any(vector @ other > .998 for other in vectors):
            continue
        chosen.append(name)
        vectors.append(vector)
        if len(chosen) == 4:
            break
    return chosen


def blend_l1(history, y, q, first):
    signed = np.column_stack([group_sum(history[..., i] - y, first).ravel() for i in range(history.shape[-1])])
    n, models = signed.shape
    errors = csr_matrix(signed)
    slack = -eye(n, format="csr")
    # Group signed errors share the same simplex weights across both VPPs.
    inequalities = vstack([hstack([errors, slack]), hstack([-errors, slack])], format="csr")
    equality = csr_matrix((np.ones(models), (np.zeros(models, dtype=int), np.arange(models))), shape=(1, models + n))
    result = linprog(np.r_[np.zeros(models), np.repeat(q, 2)], A_ub=inequalities, b_ub=np.zeros(2 * n),
                     A_eq=equality, b_eq=[1.], bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Ensemble fit failed: {result.message}")
    return result.x[:models]


def ensemble_fit(history, y, q, future, context, future_context, first, capacity, threads):
    chosen = choose_experts(history, y, q, first)
    h = np.stack([history[name] for name in chosen], axis=-1)
    f = np.stack([future[name] for name in chosen], axis=-1)
    weights = blend_l1(h, y, q, first)
    predictions = {"mean": f.mean(axis=-1), "weighted": (f @ weights).astype(np.float32)}
    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(y.size, min(120_000, y.size), replace=False))
    h = (h / capacity[None, :, None]).reshape(-1, len(chosen))
    f = (f / capacity[None, :, None]).reshape(-1, len(chosen))
    labels = (y / capacity).ravel()[indices]
    ridge = Ridge(alpha=3.).fit(h[indices], labels)
    predictions["ridge"] = ridge.predict(f).reshape(-1, len(capacity)).clip(0) * capacity
    meta_h = np.column_stack([h[indices], context.reshape(y.size, -1)[indices]])
    meta_f = np.column_stack([f, future_context.reshape(len(f), -1)])
    offset = h[indices].mean(axis=-1)
    tree = lgb.LGBMRegressor(objective="regression_l1", n_estimators=110, num_leaves=15, min_child_samples=750,
                           learning_rate=.055, verbosity=-1, random_state=SEED, n_jobs=threads)
    tree.fit(meta_h, labels - offset)
    delta = tree.booster_.predict(meta_f, num_threads=threads)
    predictions["nonlinear"] = (f.mean(axis=-1) + delta).reshape(-1, len(capacity)).clip(0) * capacity
    # Regularized residual MLP, with imputation and normalization fitted on prior OOF rows only.
    small = indices.size // 2
    impute = np.nan_to_num(np.nanmedian(meta_h[:small], axis=0))
    train_mlp = np.where(np.isfinite(meta_h[:small]), meta_h[:small], impute)
    transform = StandardScaler().fit(train_mlp)
    net = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=12, batch_size=1024, alpha=.005,
                       random_state=SEED, shuffle=True)
    net.fit(transform.transform(train_mlp), labels[:small] - offset[:small])
    delta = net.predict(transform.transform(np.where(np.isfinite(meta_f), meta_f, impute)))
    predictions["mlp"] = (f.mean(axis=-1) + delta).reshape(-1, len(capacity)).clip(0) * capacity
    models = {"experts": chosen, "weights": weights, "ridge": ridge, "nonlinear": tree,
              "mlp": net, "mlp_impute": impute, "mlp_scale": transform}
    return predictions, models


def prefix_partition(order, capacity, reference):
    running = capacity[order].cumsum()
    feasible = np.flatnonzero((running >= 20_000) & (capacity.sum() - running >= 20_000))
    if not len(feasible):
        return reference.copy()
    stop = feasible[np.argmin(np.abs(running[feasible] - capacity.sum() / 2))] + 1
    selected = np.zeros(len(order), dtype=bool)
    selected[order[:stop]] = True
    return selected


def balanced_partition(capacity, reference):
    first = np.zeros(len(capacity), dtype=bool)
    totals = np.zeros(2)
    for site in np.argsort(-capacity, kind="stable"):
        group = int(totals.argmin())
        totals[group] += capacity[site]
        first[site] = group == 0
    return first if totals.min() >= 20_000 else reference.copy()


def allocation_candidates(residuals, q, capacity, longitude, reference, periods=None):
    """Use only prior OOF residuals. All reported alternatives preserve 20 MW per group."""
    residuals = residuals.astype(np.float64)
    total_error = residuals.sum(axis=1)
    lower, upper = 20_000., capacity.sum() - 20_000.
    balanced = balanced_partition(capacity, reference)
    geographic = prefix_partition(longitude.argsort(kind="stable"), capacity, reference)
    centered = residuals - residuals.mean(axis=0)
    standardized = centered / np.maximum(1., centered.std(axis=0))
    _, vectors = np.linalg.eigh(standardized.T @ standardized)
    correlation = prefix_partition(vectors[:, -1].argsort(kind="stable"), capacity, reference)
    seeds = np.array([reference, balanced, geographic, correlation])
    month_masks = [] if periods is None else [periods == month for month in np.unique(periods)]

    def objective(first_error):
        absolute_error = np.abs(first_error) + np.abs(total_error[:, None] - first_error)
        if not month_masks:
            return .1 * q @ absolute_error
        hourly = .1 * q[:, None] * absolute_error
        average = hourly.mean(axis=0)
        worst = np.maximum.reduce([hourly[mask].mean(axis=0) for mask in month_masks])
        return average + .30 * worst

    def feasible(bits):
        caps = bits @ capacity
        return (caps >= lower) & (caps <= upper)

    rng = np.random.default_rng(SEED)
    best = None
    best_loss = np.inf
    for _ in range(3):
        population = seeds[rng.integers(len(seeds), size=40)].copy()
        mutated = population ^ (rng.random(population.shape) < .08)
        population = np.where(feasible(mutated)[:, None], mutated, population)
        population[:len(seeds)] = seeds
        for _ in range(30):
            losses = objective(residuals @ population.T)
            indices = np.argsort(losses, kind="stable")[:8]
            if losses[indices[0]] < best_loss:
                best = population[indices[0]].copy()
                best_loss = float(losses[indices[0]])
            elites = population[indices]
            left, right = elites[rng.integers(len(elites), size=(2, 32))]
            children = np.where(rng.random(left.shape) < .5, left, right)
            children ^= rng.random(children.shape) < (2 / len(capacity))
            population = np.vstack([elites, np.where(feasible(children)[:, None], children, left)])

    refined = best.copy()
    sentinel = len(capacity)
    extended = np.column_stack([residuals, np.zeros(len(q))])
    extended_capacity = np.r_[capacity, 0.]
    for _ in range(5):
        moves_out, moves_in = np.meshgrid(np.r_[np.flatnonzero(refined), sentinel],
                                         np.r_[np.flatnonzero(~refined), sentinel])
        moves_out, moves_in = moves_out.ravel(), moves_in.ravel()
        caps = capacity @ refined - extended_capacity[moves_out] + extended_capacity[moves_in]
        feasible_moves = (caps >= lower) & (caps <= upper)
        moves_out, moves_in = moves_out[feasible_moves], moves_in[feasible_moves]
        before = (residuals @ refined)[:, None]
        best_move = None
        for start in range(0, len(moves_out), 256):
            a, b = moves_out[start:start + 256], moves_in[start:start + 256]
            losses = objective(before + extended[:, b] - extended[:, a])
            idx = losses.argmin()
            if losses[idx] < best_loss - 1e-7:
                best_loss = float(losses[idx])
                best_move = a[idx], b[idx]
        if best_move is None:
            break
        a, b = best_move
        if a != sentinel:
            refined[a] = False
        if b != sentinel:
            refined[b] = True
    return {"balanced": balanced, "geographic": geographic, "correlation": correlation, "ga": best, "ga_local": refined}


def hourly_calibration(actual, prediction, q, hours, target_hours):
    residual = actual - prediction
    offsets = np.zeros((24, 2))
    for hour in range(24):
        use = (hours == hour) & (q > 0)
        if use.sum() < 25:
            continue
        for group in range(2):
            value = residual[use, group]
            order = value.argsort()
            weights = q[use][order].cumsum()
            median = value[order[np.searchsorted(weights, .5 * weights[-1])]]
            offsets[hour, group] = .6 * median
    return offsets[target_hours], offsets
