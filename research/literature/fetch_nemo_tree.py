import requests, json, pathlib, re
api='https://forge.nemo-ocean.eu/api/v4/projects/24/repository/tree'
all_items=[]; page=1
while True:
    r=requests.get(api,params={'recursive':'true','per_page':100,'page':page},timeout=30)
    r.raise_for_status(); items=r.json(); all_items+=items
    print('page',page,'got',len(items))
    if len(items)<100: break
    page+=1
out=pathlib.Path('research/literature/NEMO_gitlab_tree.json'); out.write_text(json.dumps(all_items,indent=2),encoding='utf-8')
print('total',len(all_items))
interesting=[x['path'] for x in all_items if any(k in x['path'].lower() for k in ['doc/','src/','readme','documentation','manual','algorithm','numer','physics','parameter','advection','vertical','horizontal','parallel'])]
print('interesting',len(interesting))
for p in interesting[:300]: print(p)
