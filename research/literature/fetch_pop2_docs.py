import requests, pathlib
repo='ESCOMP/POP2-CESM'; branch='master'
wanted=[
'README.md',
'source/step_mod.F90',
'source/baroclinic.F90',
'source/barotropic.F90',
'source/advection.F90',
'source/horizontal_mix.F90',
'source/hmix_del2.F90',
'source/hmix_del4.F90',
'source/hmix_gm.F90',
'source/vertical_mix.F90',
'source/vmix_kpp.F90',
'source/operators.F90',
'source/pressure_grad.F90',
'source/diagnostics.F90',
'source/grid.F90',
]
out=pathlib.Path('research/literature')
for p in wanted:
    name=repo.replace('/','_')+'_'+p.replace('/','_')
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
