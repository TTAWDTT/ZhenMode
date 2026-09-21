import requests, bs4, pathlib, re
urls = [
    ('nemo_install', 'https://sites.nemo-ocean.io/user-guide/install.html'),
    ('nemo_experiment', 'https://sites.nemo-ocean.io/user-guide/basics/preparing.html'),
    ('nemo_run', 'https://sites.nemo-ocean.io/user-guide/basics/running.html'),
    ('nemo_reference_configs', 'https://sites.nemo-ocean.io/user-guide/basics/reference.html'),
]
out = pathlib.Path('research/literature')
for name, url in urls:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text('\n', strip=True)
        (out / f'{name}.txt').write_text(text, encoding='utf-8')
        print(f'{name}: {len(text)} chars')
    except Exception as e:
        print(f'{name}: ERROR {e}')
