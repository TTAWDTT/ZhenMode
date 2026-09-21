import requests
urls=['https://forge.nemo-ocean.eu/nemo/nemo/-/raw/trunk/README.md','https://forge.nemo-ocean.eu/nemo/nemo/-/raw/trunk/README','https://forge.nemo-ocean.eu/nemo/nemo/-/raw/main/README.md']
for u in urls:
    try:
        r=requests.get(u,timeout=20); print(u,r.status_code,len(r.content)); print(r.text[:120].replace('\n',' '))
    except Exception as e: print(u,type(e).__name__,e)
