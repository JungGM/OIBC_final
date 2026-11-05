import pandas as pd
import numpy as np
from pathlib import Path

def build_features(data, data_root, split):
    d = data.copy()
    r = Path(data_root)
    base = []
    rev = []
    for p in ['wa', 'wb']:
        f = pd.read_parquet(r / split / f'forecast_{p}.parquet').sort_values([p + '_station', 'round', 'ts'])
        cs = [c for c in f if c.startswith(p + '_') and c != p + '_station' and (c not in ['wa_temp_min', 'wa_temp_max'])]
        f['day'] = (f.ts - pd.Timedelta(hours=1)).dt.floor('D')
        for c in [p + '_cloud', p + '_temp']:
            f[c + '_change'] = f.groupby([p + '_station', 'round', 'day'])[c].diff()
            for stat in ['mean', 'std']:
                f[c + '_daily_' + stat] = f.groupby([p + '_station', 'round', 'day'])[c].transform(stat)
        f = f.drop(columns='day')
        for rd in [17, 10]:
            sub = f[f['round'].eq(rd)].drop(columns='round')
            d = d.merge(sub.rename(columns={c: c + f'_r{rd}' for c in sub if c not in ['ts', p + '_station']}), on=['ts', p + '_station'], how='left', validate='many_to_one')
        base += [c + '_r17' for c in cs]
        for c in cs:
            d[c + '_revision'] = d[c + '_r17'] - d[c + '_r10']
            rev.append(c + '_revision')
    del f, sub
    d['site_code'] = d.site_id.str[2:].astype(int)
    d['hour'] = d.ts.dt.hour
    d['dow'] = d.ts.dt.dayofweek
    d['weekend'] = d.dow.ge(5)
    base += ['site_code', 'capacity_kw', 'lat', 'lon', 'wa_dist', 'wb_dist', 'hour']
    d['solar_sine'], d['solar_cos_hour'] = geo(d.ts - pd.Timedelta(minutes=30), d.lat, d.lon)
    d['solar_proxy'] = np.maximum(d.solar_sine, 0) * (1 - 0.75 * (d.wa_cloud_r17 / 100) ** 3)
    diff = []
    for aa, bb, scale in [('cloud', 'cloud', 1), ('temp', 'temp', 1), ('humidity', 'rel_hum', 1), ('wind_speed', 'wind_speed', 1 / 3.6), ('rain', 'rain', 1), ('snow', 'snow', 10)]:
        c = 'diff_' + aa
        d[c] = d['wa_' + aa + '_r17'] - d['wb_' + bb + '_r17'] * scale
        diff.append(c)
    d['freeze'] = d.wb_temp_r17.lt(0).astype(float) * d.wb_total_liq_r17
    d['thaw'] = d.wb_temp_r17.clip(lower=0) * d.wb_snow_r17
    d['missing_count'] = d[base[:28]].isna().sum(axis=1)
    cols = base + ['solar_sine', 'solar_proxy', 'solar_cos_hour', 'dow', 'weekend'] + [c for c in d if c.endswith('_change_r17')] + [c for c in d if c.endswith(('_daily_mean_r17', '_daily_std_r17'))] + rev + diff + ['freeze', 'thaw', 'missing_count']
    assert len(cols) == 89
    d = d[cols + ['ts', 'energy', 'site_id', 'current_group', 'smp']]
    return (d, cols)

def geo(ts, lat, lon):
    b = 2 * np.pi * (ts.dt.dayofyear - 81) / 364
    eq = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    sh = ts.dt.hour + ts.dt.minute / 60 + (4 * (lon - 135) + eq) / 60
    ha = np.deg2rad(15 * (sh - 12))
    dec = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + ts.dt.dayofyear) / 365))
    lat = np.deg2rad(lat)
    return (np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha), np.cos(ha))

def night_mask(data):
    ts = data.ts
    highest = np.maximum.reduce([geo(ts - pd.Timedelta(hours=1), data.lat, data.lon)[0].to_numpy(), geo(ts - pd.Timedelta(minutes=30), data.lat, data.lon)[0].to_numpy(), geo(ts, data.lat, data.lon)[0].to_numpy()])
    return highest <= np.sin(np.deg2rad(-1))
