import requests, bs4, pathlib
out=pathlib.Path('research/literature')
for name,url in {'icon_home.html':'https://www.icon-model.org/','icon_docs.html':'https://www.icon-model.org/documentation','icon_download.html':'https://www.icon-model.org/release-download','icon_getting_started.html':'https://www.icon-model.org/getting-started-with-icon'}.items():
    try:
        r=requests.get(url,timeout=30); r.raise_for_status(); (out/name).write_bytes(r.content); print(name,'OK',len(r.content))
        soup=bs4.BeautifulSoup(r.text,'html.parser')
        seen=set()
        print(' links:')
        for a in soup.find_all('a',href=True):
            href=a['href'].strip(); label=' '.join(a.get_text(' ',strip=True).split())
            if href and href not in seen and label:
                seen.add(href); print('  ',label,'->',href)
    except Exception as e: print(name,'ERR',type(e).__name__,str(e)[:120])
