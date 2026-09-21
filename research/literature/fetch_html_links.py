import requests, pathlib
urls={
'hycom_home.html':'https://www.hycom.org/',
'fvcom_home.html':'https://fvcom.smast.umassd.edu/',
'pop_home.html':'https://www.cesm.ucar.edu/models/cesm2/ocean/',
'fesom_home.html':'https://fesom.de/',
'icon_home.html':'https://icon-model.org/'
}
out=pathlib.Path('research/literature')
for name,url in urls.items():
    r=requests.get(url,timeout=30); r.raise_for_status(); (out/name).write_bytes(r.content); print(name,len(r.content))
