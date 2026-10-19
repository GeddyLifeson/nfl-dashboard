import json,base64,os
from playwright.sync_api import sync_playwright
D=json.load(open('league_data.json')); L=json.load(open('logos.json'))
F='fonts/package/files/barlow-condensed-latin-%s.woff2'
def ff(w,st):
    b=base64.b64encode(open(F%f'{w}-{st}','rb').read()).decode()
    return f"@font-face{{font-family:BC;font-weight:{w};font-style:{st};src:url(data:font/woff2;base64,{b})}}"
fonts="".join(ff(w,s) for w in(700,800) for s in('normal','italic'))
def col(teams):
    h=""
    for e in teams:
        c="#d4121f" if e['pos']%2 else "#1457d9"
        h+=f"""<div class=row style="border-left-color:{c}"><div class="rk{' t' if e['pos']<=3 else ''}">{e['pos']}</div><div class=lg style="border-color:{c}"><img src="{L[e['abbr']]}"></div><div class=n>{e['team']}</div></div>"""
    return h
html=f"""<html><head><style>{fonts}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:1080px;height:1920px;background:#000;color:#fff;font-family:BC,sans-serif;position:relative;overflow:hidden}}
.bg{{position:absolute;inset:0;background:linear-gradient(160deg,#0a1a44 0%,#000 40%,#000 62%,#3a0509 100%)}}
.bar{{position:absolute;left:0;right:0;height:14px;background:linear-gradient(90deg,#d4121f 50%,#1457d9 50%)}}
.hd{{position:absolute;top:170px;left:40px;right:80px;text-align:center}}
.k{{font-weight:700;font-size:30px;letter-spacing:6px;color:#9aa0ab;text-transform:uppercase}}
.t1{{font-weight:800;font-style:italic;font-size:58px;letter-spacing:3px;color:#2f6bff;text-transform:uppercase;line-height:1;margin-top:6px}}
.t2{{font-weight:800;font-style:italic;font-size:118px;text-transform:uppercase;line-height:.95;white-space:nowrap}}
.cols{{position:absolute;top:420px;left:36px;display:flex;gap:18px}}
.c{{width:476px}}
.row{{height:68px;margin-bottom:6px;display:flex;align-items:center;gap:12px;background:linear-gradient(90deg,#14171f,#0b0c10);border-left:7px solid #1457d9}}
.rk{{width:56px;text-align:center;font-weight:800;font-style:italic;font-size:44px}}
.rk.t{{color:#ff3b47}}
.lg{{width:84px;height:58px;border:3px solid;overflow:hidden;flex-shrink:0;background:#000}}
.lg img{{width:100%;height:100%;object-fit:cover;display:block}}
.n{{font-weight:700;font-size:30px;text-transform:uppercase;white-space:nowrap}}
</style></head><body><div class=bg></div><div class=bar style="top:0"></div>
<div class=hd><div class=k>2026 season · thru Oct 18</div><div class=t1>GeddyLifeson's</div><div class=t2>Power Rankings</div></div>
<div class=cols><div class=c>{col(D[:16])}</div><div class=c>{col(D[16:])}</div></div>
<div class=bar style="bottom:0"></div></body></html>"""
os.makedirs('tiktok',exist_ok=True)
for f in os.listdir('tiktok'): os.remove('tiktok/'+f)
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/opt/pw-browsers/chromium') if os.path.exists('/opt/pw-browsers/chromium') else p.chromium.launch()
    pg=b.new_page(viewport={'width':1080,'height':1920}); pg.set_content(html); pg.wait_for_timeout(300)
    pg.screenshot(path='tiktok/geddylifeson_power_rankings_tiktok.png'); b.close()
