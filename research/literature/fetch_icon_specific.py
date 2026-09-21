import requests, bs4, pathlib
urls={
'icon_model':'https://www.icon-model.org/icon_model',
'icon_documentation':'https://www.icon-model.org/icon_model/documentation',
'icon_release_download':'https://www.icon-model.org/icon_model/release-download',
'icon_getting_started':'https://www.icon-model.org/icon_model/getting_started',
'icon_community_interface':'https://www.icon-model.org/icon_model/community-interface',
'icon_supported_configurations':'https://www.icon-model.org/icon_model/supported-configurations',
'icon_reference_publications':'https://www.icon-model.org/publications/reference-publications',
'icon_docs':'https://docs.icon-model.org/',
}
out=pathlib.Path('research/literature')
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30); r.raise_for_status()
        text=bs4.BeautifulSoup(r.text,'html.parser').get_text('\n',strip=True)
        (out/(name+'.txt')).write_text(text,encoding='utf-8'); print(name,'OK',len(text))
    except Exception as e: print(name,'ERR',type(e).__name__,str(e)[:120])
