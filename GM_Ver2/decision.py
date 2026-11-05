"""Settlement-aligned blending and capacity-constrained VPP allocation."""

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def penalty(errors, q, first):
    e1 = errors @ first.astype(float)
    return .1*q*(np.abs(e1) + np.abs(errors.sum(axis=1)-e1))


def aggregate(prediction, first):
    return np.stack([prediction[..., first].sum(axis=-1), prediction[..., ~first].sum(axis=-1)],
                    axis=-1, dtype=np.float64)


def blend_weights(prediction, truth, q, regime, first, conditional, shrink=.01):
    # The design matrix holds signed group errors for each expert. Nonnegative
    # weights summing to one make their linear combination a coherent forecast.
    n_models = prediction.shape[-1]
    n_regimes = 2 if conditional else 1
    residual = np.stack([aggregate(prediction[..., m]-truth, first)
                         for m in range(n_models)], axis=-1).reshape(-1, n_models)
    blocks = np.repeat(regime if conditional else np.zeros(len(regime), int), 2)
    design = np.zeros((len(residual), n_models*n_regimes))
    for block in range(n_regimes):
        cols = slice(block*n_models, (block+1)*n_models)
        design[blocks == block, cols] = residual[blocks == block]
    n_weights = design.shape[1]
    n = len(residual)
    ident = sparse.eye(n, format="csr")
    left = sparse.csr_matrix(design)
    # Shrink towards uniform; penalty units are comparable to the mean weighted group error.
    w_eye = sparse.eye(n_weights, format="csr")
    ref = np.full(n_weights, 1/n_models)
    inequalities = sparse.bmat([[left, -ident, None], [-left, -ident, None],
                                [w_eye, None, -w_eye], [-w_eye, None, -w_eye]], format="csr")
    eq = sparse.hstack([np.kron(np.eye(n_regimes), np.ones(n_models)),
                        sparse.csr_matrix((n_regimes, n+n_weights))])
    weights = np.repeat(q, 2)
    typical = max(1., np.average(np.abs(residual).mean(axis=1), weights=weights))
    cost = np.r_[np.zeros(n_weights), weights/weights.sum(),
                 np.full(n_weights, shrink*typical/n_regimes)]
    result = linprog(cost, A_ub=inequalities, b_ub=np.r_[np.zeros(2*n), ref, -ref],
                     A_eq=eq, b_eq=np.ones(n_regimes), method="highs")
    if not result.success:
        raise RuntimeError(f"Blend LP failed: {result.message}")
    w = np.maximum(result.x[:n_weights].reshape(n_regimes, n_models), 0)
    return w/w.sum(axis=1, keepdims=True)


def apply_blend(prediction, regime, weights):
    return np.einsum("hsm,hm->hs", prediction, weights[regime if len(weights)>1 else np.zeros_like(regime)])


def ga_baseline(errors, q, capacity, reference, minimum=20000.):
    """Capacity-constrained XGBoost+GA comparison under the same fold cutoff."""
    rng = np.random.default_rng(20261105)
    n = len(capacity)
    upper = capacity.sum() - minimum
    total = errors.sum(axis=1)[:, None]
    pop = np.repeat(reference[None, :], 64, axis=0)

    def repair(rows):
        for row in rows:
            mass = row @ capacity
            for _ in range(n):
                if minimum <= mass <= upper:
                    break
                if mass < minimum:
                    options = np.flatnonzero(~row & (capacity <= upper-mass))
                else:
                    options = np.flatnonzero(row & (capacity <= mass-minimum))
                if not len(options):
                    row[:] = reference
                    break
                j = rng.choice(options)
                row[j] = not row[j]
                mass += capacity[j] if row[j] else -capacity[j]
            if not minimum <= row @ capacity <= upper:
                row[:] = reference

    pop[1:] ^= rng.random((63, n)) < .12
    repair(pop)
    for generation in range(61):
        first = errors @ pop.T.astype(float)
        values = .1*(q[:, None]*(np.abs(first) + np.abs(total-first))).sum(axis=0)
        if generation == 60:
            return pop[np.argmin(values)].copy()
        elite = pop[np.argsort(values)[:8]]
        parents = rng.integers(len(elite), size=(2, 56))
        children = np.where(rng.random((56, n)) < .5, elite[parents[0]], elite[parents[1]])
        children ^= rng.random(children.shape) < 2/n
        repair(children)
        pop = np.vstack([elite, children])


def vpp(errors, q, capacity, reference, month, seconds=12., minimum=20000., risk=.2):
    # |s|+|T-s| equals |T| plus twice the distance from s to [min(0,T),max(0,T)].
    # |T| is independent of membership and can be removed from the mean objective.
    n, m = errors.shape
    total = errors.sum(axis=1)
    low, high = np.minimum(total, 0), np.maximum(total, 0)
    upper = capacity.sum() - minimum
    month_ind = np.unique(month, return_inverse=True)[1]
    counts = np.bincount(month_ind)
    weights = .2*q
    monthly = sparse.csr_matrix((weights/counts[month_ind], (month_ind, np.arange(n))),
                                shape=(len(counts), n))
    fleet_month = np.bincount(month_ind, weights=.1*q*np.abs(total))/counts

    def cost(bits):
        scores = penalty(errors, q, bits)
        return scores.mean() + risk*(np.bincount(month_ind, weights=scores)/counts).max()

    def feasible(bits):
        return minimum-1e-8 <= capacity @ bits <= upper+1e-8

    if not feasible(reference):
        raise ValueError("Invalid reference VPP")
    # A deterministic pair-swap incumbent is retained when branch-and-bound times out.
    chosen = reference.copy()
    rng = np.random.default_rng(20261105)
    best = cost(chosen)
    for _ in range(5):
        left = rng.choice(np.flatnonzero(chosen), 36)
        right = rng.choice(np.flatnonzero(~chosen), 36)
        a, b = np.meshgrid(left, right)
        a, b = a.ravel(), b.ravel()
        cap = capacity @ chosen - capacity[a] + capacity[b]
        a, b = a[(cap >= minimum) & (cap <= upper)], b[(cap >= minimum) & (cap <= upper)]
        if not len(a):
            break
        e = (errors @ chosen)[:, None] + errors[:, b] - errors[:, a]
        p = .1*q[:, None]*(np.abs(e) + np.abs(total[:, None]-e))
        by_month = np.stack([p[month_ind == k].mean(axis=0) for k in range(len(counts))])
        objective = p.mean(axis=0) + risk*by_month.max(axis=0)
        j = objective.argmin()
        if objective[j] < best:
            chosen[a[j]], chosen[b[j]] = False, True
            best = float(objective[j])

    E = sparse.csr_matrix(errors)
    ident = sparse.eye(n, format="csr")
    A = sparse.bmat([[E, -ident, sparse.csr_matrix((n, 1))],
                     [-E, -ident, sparse.csr_matrix((n, 1))],
                     [sparse.csr_matrix(capacity.reshape(1, -1)), sparse.csr_matrix((1, n)),
                      sparse.csr_matrix((1, 1))],
                     [sparse.csr_matrix((len(counts), m)), monthly, -np.ones((len(counts), 1))]],
                    format="csc")
    lower = np.r_[np.full(2*n, -np.inf), minimum, np.full(len(counts), -np.inf)]
    right = np.r_[high, -low, upper, -fleet_month]
    # Break label symmetry with the first site. Bound the solve by wall time.
    ub = np.r_[np.ones(m), np.full(n+1, np.inf)]
    lb = np.r_[1., np.zeros(m+n)]
    result = milp(np.r_[np.zeros(m), weights/n, risk], integrality=np.r_[np.ones(m), np.zeros(n+1)],
                  bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lower, right),
                  options={"time_limit": seconds, "mip_rel_gap": .001})
    source = "swap_or_reference"
    if result.x is not None:
        integer = result.x[:m]
        bits = integer > .5
        if np.allclose(integer, bits, atol=1e-5, rtol=0) and feasible(bits) and cost(bits) <= best:
            chosen, source = bits, "milp"
    dual = getattr(result, "mip_dual_bound", None)
    lower_bound = None if dual is None or not np.isfinite(dual) else float(dual + (.1*q*np.abs(total)).mean())
    return chosen, dict(source=source, solver_status=int(result.status),
                        solver_message=result.message, objective=float(cost(chosen)),
                        reference_objective=float(cost(reference)), objective_lower_bound=lower_bound)
