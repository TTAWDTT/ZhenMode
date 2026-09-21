import requests, json, pathlib, re
api='https://forge.nemo-ocean.eu/api/v4/projects/24'
r=requests.get(api,timeout=30); print(r.status_code); d=r.json(); print(d.get('default_branch'),d.get('http_url_to_repo'),d.get('web_url')); print({k:d.get(k) for k in ['name','default_branch','path_with_namespace']})
r=requests.get(api+'/repository/tree?recursive=true&per_page=100',timeout=30)
print('tree status',r.status_code,'count',len(r.json()) if r.ok else r.text[:100])
items=r.json()
out=pathlib.Path('research/literature/NEMO_gitlab_tree_page1.json'); out.write_text(json.dumps(items,indent=2),encoding='utf-8')
for x in items[:100]:
    print(x['type'],x['path'])
