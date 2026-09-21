import requests, bs4, pathlib
urls={
'hycom_overview':'https://www.hycom.org/hycom/overview',
'hycom_documentation':'https://www.hycom.org/hycom/documentation',
'hycom_source_code':'https://www.hycom.org/hycom/source-code',
'hycom_readme':'https://www.hycom.org/readme',
'fvcom_source_code':'https://github.com/FVCOM-GitHub/FVCOM',
'fvcom_model_validation':'https://fvcom.smast.umassd.edu/?p=168',
'fvcom_about':'https://fvcom.smast.umassd.edu/?p=81',
'fvcom_help':'https://fvcom.smast.umassd.edu/?p=15',
'pop2_model_page':'https://www.cesm.ucar.edu/models/pop2',
'cesm_ocean_component':'https://www.cesm.ucar.edu/models/cesm2/ocean',
'cesm_pop':'https://www.cesm.ucar.edu/models/pop',
'cesm_mom_interface':'https://github.com/ESCOMP/MOM_interface/wiki/Detailed-Instructions',
}
out=pathlib.Path('research/literature')
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30); r.raise_for_status()
        text=bs4.BeautifulSoup(r.text,'html.parser').get_text('\n',strip=True)
        (out/(name+'.txt')).write_text(text,encoding='utf-8'); print(name,'OK',len(text))
    except Exception as e:
        print(name,'ERR',type(e).__name__,str(e)[:100])
