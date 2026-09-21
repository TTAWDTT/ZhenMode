import requests, bs4, pathlib
out=pathlib.Path('research/literature')
for name,url in {'nemo_user_guide_home.html':'https://sites.nemo-ocean.io/user-guide/'}.items():
    r=requests.get(url,timeout=30); r.raise_for_status(); (out/name).write_bytes(r.content); print(name,len(r.content))
    soup=bs4.BeautifulSoup(r.text,'html.parser')
    seen=set()
    for a in soup.find_all('a',href=True):
        href=a['href'].strip(); label=' '.join(a.get_text(' ',strip=True).split())
        if href and href not in seen and label and len(label)<120:
            seen.add(href); print(label,'->',href)
