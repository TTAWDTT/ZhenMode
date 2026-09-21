import requests
repos=['DWD-ICON/icon-model','icon-model/icon-model','myroms/roms','FESOM/fesom2','MPAS-Dev/MPAS-Model','CHLNDDEV/fvcom','ESCOMP/POP2-CESM','NEMO-ocean/NEMO','NEMO-ocean/NEMO','MITgcm/MITgcm','NOAA-GFDL/MOM6']
for repo in repos:
    r=requests.get('https://api.github.com/repos/'+repo,timeout=20)
    print(repo, r.status_code)
    if r.ok:
        d=r.json(); print(' ',d.get('full_name'),d.get('default_branch'),d.get('stargazers_count'),d.get('description'))
    else: print(r.text[:120])
