import requests, pathlib
base='https://raw.githubusercontent.com/NOAA-GFDL/MOM6/main/docs/'
files=['grids.rst','discrete_space.rst','discrete_time.rst','parameterizations_lateral.rst','parameterizations_vertical.rst','tracers.rst','parallel.rst','other_physics.rst','testing.rst','code_organization.rst']
out=pathlib.Path('research/literature')
for f in files:
    r=requests.get(base+f,timeout=30); r.raise_for_status(); (out/('mom6_'+f)).write_text(r.text,encoding='utf-8'); print(f, len(r.text))
