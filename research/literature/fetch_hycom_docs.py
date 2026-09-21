import requests, pathlib
repo='HYCOM/HYCOM-src'; branch='master'
wanted=['README.md','inikpp.F90','mod_tsadvc.F90','thermf.F90']
out=pathlib.Path('research/literature')
for p in wanted:
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); name=repo.replace('/','_')+'_'+p.replace('/','_')
    (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
