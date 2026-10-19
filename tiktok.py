"""1080x1920 rankings image. Run: python tiktok.py   (needs: pip install playwright)
Optional: logos.json ({abbr: image URL or data URI}, else ESPN's logo images) and fonts/ (Barlow Condensed woff2 files)."""
import base64, json, os
from playwright.sync_api import sync_playwright

D = json.load(open('league_data.json'))
L = json.load(open('logos.json')) if os.path.exists('logos.json') else {}
FONT = 'fonts/package/files/barlow-condensed-latin-%s-%s.woff2'
VARIANTS = [(w, s) for w in (700, 800) for s in ('normal', 'italic')]
if all(os.path.exists(FONT % v) for v in VARIANTS):
    fonts = ''.join(f"@font-face{{font-family:BC;font-weight:{w};font-style:{s};src:url(data:font/woff2;base64,"
                    f"{base64.b64encode(open(FONT % (w, s), 'rb').read()).decode()})}}" for w, s in VARIANTS)
else:
    fonts = "@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,700;0,800;1,800&display=swap');"


def col(teams):
    h = ''
    for e in teams:
        c = '#d4121f' if e['pos'] % 2 else '#1457d9'
        espn = {'WAS': 'wsh'}.get(e['abbr'], e['abbr'].lower())
        src = L.get(e['abbr'], f'https://a.espncdn.com/i/teamlogos/nfl/500-dark/{espn}.png')
        logo = f'<img src="{src}" alt="{e["abbr"]}">'
        mv = (f'<div class="mv up">▲{e["move"]}</div>' if e['move'] > 0 else
              f'<div class="mv dn">▼{-e["move"]}</div>' if e['move'] < 0 else '')
        h += (f'<div class=row style="border-left-color:{c}"><div class="rk{" t" if e["pos"] <= 3 else ""}">{e["pos"]}</div>'
              f'<div class=lg style="border-color:{c}">{logo}</div><div class=n>{e["team"]}</div>{mv}</div>')
    return h


html = f"""<html><head><style>{fonts}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:1080px;height:1920px;background:#000;color:#fff;font-family:BC,'Barlow Condensed','Arial Narrow',sans-serif;position:relative;overflow:hidden}}
.bg{{position:absolute;inset:0;background:linear-gradient(160deg,#0a1a44 0%,#000 40%,#000 62%,#3a0509 100%)}}
.bar{{position:absolute;left:0;right:0;height:14px;background:linear-gradient(90deg,#d4121f 50%,#1457d9 50%)}}
.hd{{position:absolute;top:170px;left:40px;right:80px;text-align:center}}
.k{{font-weight:700;font-size:30px;letter-spacing:6px;color:#9aa0ab;text-transform:uppercase}}
.t1{{font-weight:800;font-style:italic;font-size:58px;letter-spacing:3px;color:#2f6bff;text-transform:uppercase;line-height:1;margin-top:6px}}
.t2{{font-weight:800;font-style:italic;font-size:118px;text-transform:uppercase;line-height:.95;white-space:nowrap}}
.cols{{position:absolute;top:420px;left:31px;display:flex;gap:18px}}
.c{{width:500px}}
.row{{height:68px;margin-bottom:6px;display:flex;align-items:center;gap:12px;background:linear-gradient(90deg,#14171f,#0b0c10);border-left:7px solid #1457d9;padding-right:10px}}
.rk{{width:56px;text-align:center;font-weight:800;font-style:italic;font-size:44px}}
.rk.t{{color:#ff3b47}}
.lg{{width:84px;height:58px;border:3px solid;overflow:hidden;flex-shrink:0;background:#000;display:flex;align-items:center;justify-content:center}}
.lg img{{width:100%;height:100%;object-fit:contain;display:block}}
.n{{font-weight:700;font-size:28px;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}}
.mv{{font-weight:700;font-size:24px}}
.up{{color:#3ddc6a}}
.dn{{color:#ff4d57}}
</style></head><body><div class=bg></div><div class=bar style="top:0"></div>
<div class=hd><div class=k>{D['season']} season · thru {D['through']}</div><div class=t1>GeddyLifeson's</div><div class=t2>Power Rankings</div></div>
<div class=cols><div class=c>{col(D['teams'][:16])}</div><div class=c>{col(D['teams'][16:])}</div></div>
<div class=bar style="bottom:0"></div></body></html>"""

os.makedirs('tiktok', exist_ok=True)

def launch(p):
    """Playwright's own Chromium if installed, otherwise the system Chrome or Edge."""
    if os.path.exists('/opt/pw-browsers/chromium'):
        return p.chromium.launch(executable_path='/opt/pw-browsers/chromium')
    for channel in (None, 'chrome', 'msedge'):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:
            continue
    raise SystemExit('no browser: run "playwright install chromium" or install Chrome')


with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page(viewport={'width': 1080, 'height': 1920})
    pg.set_content(html)
    pg.wait_for_load_state('networkidle')
    pg.screenshot(path='tiktok/geddylifeson_power_rankings_tiktok.png')
    b.close()
