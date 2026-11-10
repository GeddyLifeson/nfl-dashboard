"""nfelo (github.com/greerreNFL/nfelo, output_data/nfelo_games.csv) on SAM's frozen 2025-26 test games.
Model-only number: nfelo_dif_base (Elo + home field + QB, before market regression). Market-blended: nfelo_dif_close.
Elo points -> spread points at 25 per point (538/nfelo convention)."""
import json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault('TRAIN_FROM', '2014')
import mf_backtest as m
NF = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/ext/nfelo/output_data/nfelo_games.csv'
nf = pd.read_csv(NF)
nf['game_id'] = nf.game_id.str.replace('_LAR', '_LA', regex=False).str.replace('_OAK', '_LV', regex=False)
nf = nf.set_index('game_id')
rs = [r for r in json.load(open('cache/mf_rows.json')) if r['season'] >= 2013]
lam, f, sig, _, _ = m.run(rs, m.SETS['E + coaching changes'])
te = [r for r in rs if r['season'] in m.TEST and r['spread'] is not None and r['id'] in nf.index]
X, y = m.mat(te, m.SETS['E + coaching changes']); sam = f(X)
line = np.array([r['spread'] for r in te]); base = np.array([r['f']['base'] for r in te])
nb = nf.loc[[r['id'] for r in te]]
def ev(pred, prob=None, label=''):
    nt = y != 0; w = (y > 0).astype(float)
    p = prob if prob is not None else 0.5 * (1 + np.vectorize(math.erf)(pred / sig / math.sqrt(2)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = dict(games=len(y), winners=round(float(np.mean((pred[nt] > 0) == (y[nt] > 0))), 4), mse=round(float(np.mean((pred - y) ** 2)), 1),
               mae=round(float(np.mean(np.abs(pred - y))), 2), log_loss=round(float(-np.mean((w * np.log(p) + (1 - w) * np.log(1 - p))[nt])), 4))
    print(f'{label:34s} {out}'); return out
res = {}
res['sam'] = ev(sam, label='SAM all-factor')
res['scores_only'] = ev(base, label='old scores-only')
nfb = nb.nfelo_dif_base.values / 25; res['nfelo_model'] = ev(nfb, 1 / (1 + 10 ** (-nb.nfelo_dif_base.values / 401.62)), 'nfelo model only (pre-market)')
nfc = nb.nfelo_dif_close.values / 25; res['nfelo_blend'] = ev(nfc, nb.nfelo_home_probability_close.values, 'nfelo market-blended (close)')
mlp = np.array([np.nan] * len(te))
res['close'] = ev(line, None, 'closing line (normal curve)')
for s in m.TEST:
    k = np.array([r['season'] == s for r in te])
    print(s, 'games', int(k.sum()), 'winners: SAM %.3f nfelo-model %.3f nfelo-blend %.3f line %.3f' % tuple(
        np.mean((v[k][y[k] != 0] > 0) == (y[k][y[k] != 0] > 0)) for v in (sam, nfb, nfc, line)))
json.dump(res, open('research/nfelo_bench.json', 'w'), indent=1)
