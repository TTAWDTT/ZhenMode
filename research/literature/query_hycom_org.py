import requests
r=requests.get('https://api.github.com/orgs/HYCOM/repos',params={'per_page':100},timeout=30)
print(r.status_code, len(r.json()))
for it in r.json():
    print(it['full_name'],it['stargazers_count'],it.get('description'))
