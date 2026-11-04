"""Local-only OIBC submission validator. Python standard library only."""
import argparse
import csv
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
START = datetime(2025, 6, 1, 1, tzinfo=KST)
EXPECTED = {START + timedelta(hours=i) for i in range(8760)}


def read_csv(path, columns):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate headers')
        if not set(columns).issubset(reader.fieldnames):
            raise ValueError('Required columns: ' + ', '.join(columns))
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('Malformed CSV row')
        return rows


def validate(composition, forecast, sites):
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    try:
        meta = read_csv(sites, ['site_id', 'capacity_kw'])
        comp = read_csv(composition, ['site_id', 'group'])
        pred = read_csv(forecast, ['group', 'ts', 'forecast_kwh'])
        ids = [r['site_id'] for r in meta]
        check(len(ids) == 253 and len(set(ids)) == 253, 'sites: expected 253 unique sites')
        capacities = {r['site_id']: float(r['capacity_kw']) for r in meta}
        check(all(math.isfinite(v) and v > 0 for v in capacities.values()), 'sites: invalid capacity')
        counts = Counter(r['site_id'] for r in comp)
        check(len(comp) == 253 and set(counts) == set(ids) and all(v == 1 for v in counts.values()),
              'composition: all 253 sites must occur exactly once')
        check(all(r['group'] in ('G1', 'G2') for r in comp), 'composition: group must be G1 or G2')
        for group in ('G1', 'G2'):
            total = math.fsum(capacities.get(r['site_id'], 0) for r in comp if r['group'] == group)
            check(total >= 20000, 'composition: ' + group + ' capacity below 20000 kW')
        check(len(pred) == 17520, 'forecast: expected 17520 data rows (excluding header)')
        keys = Counter()
        bad_time = bad_value = bad_group = 0
        for row in pred:
            if row['group'] not in ('G1', 'G2'):
                bad_group += 1
            try:
                ts = datetime.fromisoformat(row['ts'].replace('Z', '+00:00'))
                if ts.utcoffset() != timedelta(hours=9):
                    raise ValueError('KST offset required')
                keys[(row['group'], ts)] += 1
            except (ValueError, TypeError):
                bad_time += 1
            try:
                value = float(row['forecast_kwh'])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid prediction')
            except (ValueError, TypeError):
                bad_value += 1
        expected_keys = {(g, ts) for g in ('G1', 'G2') for ts in EXPECTED}
        check(not bad_group, 'forecast: invalid group')
        check(not bad_time, 'forecast: invalid timestamp or missing +09:00 offset')
        check(not bad_value, 'forecast: negative, missing, nonnumeric or nonfinite values')
        check(set(keys) == expected_keys and all(v == 1 for v in keys.values()),
              'forecast: missing, duplicate or out-of-range (group, ts) pairs')
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        errors.append('Cannot validate input: ' + type(exc).__name__)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition', required=True, type=Path)
    parser.add_argument('--forecast', required=True, type=Path)
    parser.add_argument('--sites', required=True, type=Path)
    args = parser.parse_args()
    errors = validate(args.composition, args.forecast, args.sites)
    for error in errors:
        print('FAIL: ' + error)
    print('FAIL: submission rejected locally' if errors else 'PASS: all local format checks passed')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
