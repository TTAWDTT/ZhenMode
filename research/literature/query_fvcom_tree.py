import requests, json, pathlib
r=requests.get('https://api.github.com/repos/FVCOM-GitHub/FVCOM/git/trees/main?recursive=1',timeout=30)
print('status',r.status_code,'count',len(r.json().get('tree',[])) if r.ok else r.text[:100])
if r.ok:
    data=r.json(); pathlib.Path('research/literature/FVCOM_tree.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    paths=[x['path'] for x in data.get('tree',[])]
    for p in paths:
        if any(k in p.lower() for k in ['readme','doc','manual','src/mix','src/mod_main','src/momentum','src/advection','src/tracer','src/pressure','src/vertical','src/bc','src/parallel']) and p.endswith(('.md','.txt','.F90','.f90','.h','.c')):
            print(p)
