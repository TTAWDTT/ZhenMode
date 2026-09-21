import requests, pathlib, json
repo='FESOM/fesom2'; branch='main'
wanted=[
'docs/geometry.rst',
'docs/main_equations.rst',
'docs/spatial_discretization.rst',
'docs/vertical_discretization.rst',
'docs/temporal_discretization.rst',
'docs/time_stepping_transport.rst',
'docs/subcycling_instead_solver.rst',
'docs/isoneutral_diffusion_triangular_prisms.rst',
'docs/ocean_configuration/ocean_configuration.rst',
'docs/developer_documentation/gpu_and_dwarfs.rst',
]
out=pathlib.Path('research/literature')
for p in wanted:
    name=repo.replace('/','_')+'_'+p.replace('/','_')
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
