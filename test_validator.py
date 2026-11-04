"""Synthetic data only; no competition data is loaded."""
import csv
import tempfile
import unittest
from pathlib import Path
from validate_submission import EXPECTED, validate


class ValidatorTests(unittest.TestCase):
    def test_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, header, rows):
                with (root / name).open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    w.writerows(rows)
            sites = [('PV%03d' % i, 170) for i in range(1, 254)]
            comp = [(s, 'G1' if i < 126 else 'G2') for i, (s, _) in enumerate(sites)]
            pred = [(g, ts.isoformat(), 0) for g in ('G1', 'G2') for ts in sorted(EXPECTED)]
            write('sites.csv', ['site_id', 'capacity_kw'], sites)
            write('composition.csv', ['site_id', 'group'], comp)
            def run(rows=pred, composition=comp):
                write('forecast.csv', ['group', 'ts', 'forecast_kwh'], rows)
                write('composition.csv', ['site_id', 'group'], composition)
                return validate(root / 'composition.csv', root / 'forecast.csv', root / 'sites.csv')
            self.assertEqual(run(), [])
            mutations = [pred[:-1], [pred[1]] + pred[1:]]
            for value in [-1, 'NaN', 'inf', '']:
                mutations.append([(pred[0][0], pred[0][1], value)] + pred[1:])
            mutations.append([('G1', '2025-06-01T01:00:00', 0)] + pred[1:])
            mutations.append([('G1', '2025-06-01T01:00:00+00:00', 0)] + pred[1:])
            mutations.append([('G3', pred[0][1], 0)] + pred[1:])
            for rows in mutations:
                self.assertTrue(run(rows))
            self.assertTrue(run(composition=comp[:-1]))
            self.assertTrue(run(composition=[(s, 'G1') for s, _ in comp]))
            self.assertTrue(run(composition=[comp[1]] + comp[1:]))


if __name__ == '__main__':
    unittest.main()
