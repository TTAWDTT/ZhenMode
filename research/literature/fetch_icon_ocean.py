import requests, bs4, pathlib
base='https://docs.icon-model.org/documentation/ocean/'
wanted=['ocean.html']
out=pathlib.Path('research/literature')
for p in wanted:
    r=requests.get(base+p,timeout=30); r.raise_for_status()
    text=bs4.BeautifulSoup(r.text,'html.parser').get_text('\n',strip=True)
    open(out/('icon_ocean_'+p),'w',encoding='utf-8').write(text); print(p,len(text))
    soup=bs4.BeautifulSoup(r.text,'html.parser')
    seen=set()
    print('links:')
    for a in soup.find_all('a',href=True):
        href=a['href'].strip(); label=' '.join(a.get_text(' ',strip=True).split())
        if href and href not in seen and label and len(label)<140:
            seen.add(href); print('  ',label,'->',href)
