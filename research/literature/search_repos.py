import requests, json
queries = ['ROMS ocean model','FESOM2','MPAS-Model','ICON model ocean','HYCOM model','FVCOM ocean model','POP2-CESM ocean','NEMO ocean model']
for q in queries:
    r=requests.get('https://api.github.com/search/repositories', params={'q':q,'per_page':5}, timeout=30)
    print('QUERY',q,'STATUS',r.status_code)
    if r.ok:
        for it in r.json().get('items',[]):
            print(' ',it['full_name'],it['stargazers_count'],it.get('description'))
    else: print(r.text[:200])
