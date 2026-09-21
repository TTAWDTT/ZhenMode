import requests, pathlib, re, json
repos={
 'myroms/roms':'develop',
 'FESOM/fesom2':'main',
 'MPAS-Dev/MPAS-Model':'master',
 'ESCOMP/POP2-CESM':'master',
 'MITgcm/MITgcm':'master',
 'NOAA-GFDL/MOM6':'dev/gfdl',
}
for repo,branch in repos.items():
    url=f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1'
    r=requests.get(url,timeout=30)
    print('REPO',repo,r.status_code,'count',len(r.json().get('tree',[])) if r.ok else '')
    if r.ok:
        tree=r.json().get('tree',[])
        out=pathlib.Path('research/literature')/f'{repo.replace("/", "_")}_tree.json'
        out.write_text(json.dumps(tree,indent=2),encoding='utf-8')
        paths=[x['path'] for x in tree]
        docs=[p for p in paths if re.search(r'(readme|documentation|manual|doc|docs|algorithm|numer|physics|parameter|advection|vertical|horizontal|parallel)',p,re.I) and (p.endswith(('.rst','.md','.tex','.txt','.pdf','.f90','.F90','.h','.py','.txt')))]
        print(' docs-ish',len(docs))
        for p in docs[:80]: print('  ',p)
        if len(docs)>80: print('  ...',len(docs)-80,'more')
