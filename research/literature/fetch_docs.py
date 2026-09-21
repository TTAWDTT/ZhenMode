import requests, bs4, pathlib, re
urls = [
    ('nemo_user_guide_home', 'https://sites.nemo-ocean.io/user-guide/'),
    ('mitgcm_overview', 'https://mitgcm.readthedocs.io/en/latest/overview/overview.html'),
    ('mitgcm_algorithm', 'https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html'),
    ('mitgcm_software_arch', 'https://mitgcm.readthedocs.io/en/latest/software_arch/software_arch.html'),
    ('roms_home', 'https://www.myroms.org/'),
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
