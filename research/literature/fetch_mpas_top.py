import requests, pathlib
repo='MPAS-Dev/MPAS-Model'; branch='master'
wanted=['README.md','docs/ocean/index.rst','docs/ocean/design_docs/index.rst']
out=pathlib.Path('research/literature')
for p in wanted:
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); name=repo.replace('/','_')+'_'+p.replace('/','_'); (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
