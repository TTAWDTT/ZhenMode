import requests, json, pathlib, re
repo='HYCOM/HYCOM-src'; branch='master'
r=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1',timeout=30)
print('status',r.status_code,'count',len(r.json().get('tree',[])) if r.ok else r.text[:100])
if r.ok:
    data=r.json(); pathlib.Path('research/literature/HYCOM_HYCOM-src_tree.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    tree=data.get('tree',[])
    interesting=[x['path'] for x in tree if re.search(r'(readme|doc|manual|src|mix|adv|prs|step|therm|layer|kpp|mixing|advection|momentum|topog)',x['path'],re.I) and x['path'].endswith(('.F','.f','.F90','.h','.txt','.md'))]
    print('interesting',len(interesting))
    for p in interesting[:200]: print(p)
