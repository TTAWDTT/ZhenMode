import requests
for p in ['https://forge.nemo-ocean.eu/api/v4/projects/nemo%2Fnemo','https://forge.nemo-ocean.eu/api/v4/projects/103','https://forge.nemo-ocean.eu/api/v4/projects?search=NEMO&per_page=10']:
    r=requests.get(p,timeout=20); print(r.status_code, len(r.text)); print(r.text[:500].replace('\n',' '))
