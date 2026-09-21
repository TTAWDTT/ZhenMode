import requests, bs4, pathlib, re, time, json
urls = {
 # NEMO
 'nemo_user_guide_home':'https://sites.nemo-ocean.io/user-guide/',
 'nemo_preparing':'https://sites.nemo-ocean.io/user-guide/basics/preparing.html',
 'nemo_running':'https://sites.nemo-ocean.io/user-guide/basics/running.html',
 'nemo_reference_configs':'https://sites.nemo-ocean.io/user-guide/basics/reference.html',
 'nemo_inputs':'https://sites.nemo-ocean.io/user-guide/basics/inputs.html',
 'nemo_outputs':'https://sites.nemo-ocean.io/user-guide/basics/outputs.html',
 # ROMS
 'roms_home':'https://www.myroms.org/',
 'roms_wiki_home':'https://www.myroms.org/wiki/Documentation',
 'roms_numerical_solution':'https://www.myroms.org/wiki/Numerical_Solution',
 'roms_vertical_coordinate':'https://www.myroms.org/wiki/Vertical_Coordinate',
 'roms_horizontal_coordinate':'https://www.myroms.org/wiki/Horizontal_Coordinate',
 'roms_advection':'https://www.myroms.org/wiki/Advection',
 'roms_boundary_conditions':'https://www.myroms.org/wiki/Boundary_Conditions',
 'roms_parallelization':'https://www.myroms.org/wiki/Parallelization',
 # MPAS
 'mpas_ocean_home':'https://mpas-dev.github.io/ocean/default/',
 'mpas_model_home':'https://mpas-dev.github.io/',
 'mpas_docs_main':'https://mpas-dev.github.io/Mesh/default/',
 'mpas_docs_ocean':'https://mpas-dev.github.io/Ocean/default/',
 # FESOM
 'fesom_home':'https://fesom.de/',
 'fesom_documentation':'https://fesom.de/documentation/',
 'fesom_models':'https://fesom.de/models/',
 # ICON
 'icon_home':'https://icon-model.org/',
 'icon_ocean':'https://icon-model.org/ocean-model',
 'icon_docs':'https://icon-model.org/documentation/',
 # HYCOM
 'hycom_home':'https://www.hycom.org/',
 'hycom_model_description':'https://www.hycom.org/hycom/model-description',
 # FVCOM
 'fvcom_home':'https://fvcom.smast.umassd.edu/',
 'fvcom_documentation':'https://fvcom.smast.umassd.edu/documentation/',
 # POP
 'pop_home':'https://www.cesm.ucar.edu/models/cesm2/ocean/',
 'pop_model_page':'https://www2.cesm.ucar.edu/models/cesm2/ocean/',
 # MOM6
 'mom6_docs_home':'https://mom6.readthedocs.io/en/latest/',
 'mom6_readme':'https://raw.githubusercontent.com/NOAA-GFDL/MOM6/main/README.md',
}
out=pathlib.Path('research/literature')
def slug(url):
    # derive stable name from url
    s=url.rstrip('/').split('//',1)[-1]
    s=re.sub(r'https?/','',s)
    s=s.replace('/','_').replace(':','_').replace('?','_').replace('=','_')
    return s
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        ctype=r.headers.get('content-type','')
        if 'html' in ctype:
            soup=bs4.BeautifulSoup(r.text,'html.parser')
            text=soup.get_text('\n',strip=True)
            (out/f'{name}.txt').write_text(text,encoding='utf-8')
            print(name,'OK',len(text))
        else:
            (out/f'{name}.bin').write_bytes(r.content)
            print(name,'OK',ctype,len(r.content))
    except Exception as e:
        print(name,'ERR',type(e).__name__,str(e)[:120])
    time.sleep(0.2)
