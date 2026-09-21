import requests, bs4, pathlib
for f in ['hycom_home.html','fvcom_home.html','pop_home.html','fesom_home.html','icon_home.html']:
    try:
        text=open('research/literature/'+f,encoding='utf-8',errors='ignore').read()
        soup=bs4.BeautifulSoup(text,'html.parser')
        print('\n###',f)
        seen=set()
        for a in soup.find_all('a',href=True):
            href=a['href'].strip()
            if href and href not in seen:
                seen.add(href)
                label=' '.join(a.get_text(' ',strip=True).split())
                if label and len(label)<100:
                    print(label,'->',href)
    except FileNotFoundError:
        print('missing',f)
