import requests
for u in ['https://mom6.readthedocs.io/en/latest/api/generated/pages/Discrete_Grids.html','https://mom6.readthedocs.io/en/latest/api/generated/pages/PPM.html','https://mom6.readthedocs.io/en/latest/api/generated/pages/Discrete_Coriolis.html']:
    r=requests.get(u,timeout=30)
    print(u,r.status_code,len(r.text))
    if r.ok:
        print(r.text[:300].replace('\n',' '))
