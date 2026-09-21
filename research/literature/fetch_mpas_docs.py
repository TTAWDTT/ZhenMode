import requests, pathlib
repo='MPAS-Dev/MPAS-Model'; branch='master'
wanted=[
'src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F',
'src/core_ocean/shared/mpas_ocn_thick_ale.F',
'src/core_ocean/shared/mpas_ocn_thick_hadv.F',
'src/core_ocean/shared/mpas_ocn_tracer_advection.F',
'src/core_ocean/shared/mpas_ocn_tracer_advection_mono.F',
'src/core_ocean/shared/mpas_ocn_tracer_hmix.F',
'src/core_ocean/shared/mpas_ocn_vel_hmix.F',
'src/core_ocean/shared/mpas_ocn_vel_hmix_del2.F',
'src/core_ocean/shared/mpas_ocn_vel_hmix_del4.F',
'src/core_ocean/shared/mpas_ocn_vel_hmix_leith.F',
'src/core_ocean/shared/mpas_ocn_vmix_cvmix.F',
'src/core_ocean/shared/mpas_ocn_gm.F',
'src/core_ocean/shared/mpas_ocn_vel_pressure_grad.F',
]
out=pathlib.Path('research/literature')
for p in wanted:
    name=repo.replace('/','_')+'_'+p.replace('/','_')
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
