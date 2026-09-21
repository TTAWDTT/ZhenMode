import requests, bs4, pathlib
base='https://sites.nemo-ocean.io/user-guide/'
wanted=['install.html','cfgs.html','tests.html','tracers.html','setup.html','diags.html','mixed.html','psyclone.html','changes.html','cite.html','acro.html']
out=pathlib.Path('research/literature')
for p in wanted:
    try:
        r=requests.get(base+p,timeout=30); r.raise_for_status()
        text=bs4.BeautifulSoup(r.text,'html.parser').get_text('\n',strip=True)
        (out/('nemo_'+p)).write_text(text,encoding='utf-8'); print(p,'OK',len(text))
    except Exception as e: print(p,'ERR',type(e).__name__,str(e)[:100])
