import requests
url='https://api.github.com/repos/NOAA-GFDL/MOM6/git/trees/main?recursive=1'
paths=[x['path'] for x in requests.get(url,timeout=30).json().get('tree',[]) if x['path'].startswith('docs/api')]
print('\n'.join(paths))
