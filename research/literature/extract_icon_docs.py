import requests, bs4, pathlib, re
url='https://docs.icon-model.org/'
r=requests.get(url,timeout=30); r.raise_for_status(); text=r.text
open('research/literature/icon_docs_root.html','w',encoding='utf-8').write(text)
soup=bs4.BeautifulSoup(text,'html.parser')
seen=set()
for a in soup.find_all('a',href=True):
    href=a['href'].strip(); label=' '.join(a.get_text(' ',strip=True).split())
    if href and href not in seen:
        seen.add(href)
        if label and len(label)<120: print(label,'->',href)
