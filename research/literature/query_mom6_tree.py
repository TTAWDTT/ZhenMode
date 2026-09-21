import requests, json, re
url='https://api.github.com/repos/NOAA-GFDL/MOM6/git/trees/main?recursive=1'
r=requests.get(url,timeout=30)
print(r.status_code, len(r.text))
if r.ok:
    data=r.json()
    paths=[x['path'] for x in data.get('tree',[]) if 'docs' in x['path'] and (x['path'].endswith('.rst') or x['path'].endswith('.md'))]
    for p in paths[:200]: print(p)
    print('count',len(paths))
else: print(r.text[:200])
