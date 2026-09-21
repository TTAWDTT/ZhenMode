import requests, bs4, re, pathlib
for f in ['hycom_home.txt','fvcom_home.txt','pop_home.txt','fesom_home.txt','icon_home.txt']:
    text=open('research/literature/'+f,encoding='utf-8',errors='ignore').read()
    soup=bs4.BeautifulSoup(text,'html.parser')
    print('\n###',f)
    seen=set()
    for a in soup.find_all('a',href=True):
        href=a['href'].strip()
        if href and href not in seen:
            seen.add(href)
            label=' '.join(a.get_text(' ',strip=True).split())
            if label and len(label)<90:
                print(label,'->',href)
