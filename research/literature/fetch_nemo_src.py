import requests, pathlib
api='https://forge.nemo-ocean.eu/api/v4/projects/24/repository/files'
branch='main'
wanted=[
'README.rst',
'src/OCE/DOM/domhgr.F90',
'src/OCE/DOM/domzgr.F90',
'src/OCE/DOM/dommsk.F90',
'src/OCE/DYN/divhor.F90',
'src/OCE/DYN/dynadv.F90',
'src/OCE/DYN/dynadv_cen2.F90',
'src/OCE/DYN/dynadv_up3.F90',
'src/OCE/DYN/dynspg.F90',
'src/OCE/DYN/dynspg_exp.F90',
'src/OCE/DYN/dynspg_ts.F90',
'src/OCE/DYN/dynvor.F90',
'src/OCE/DYN/dynzad.F90',
'src/OCE/DYN/dynzdf.F90',
'src/OCE/DYN/sshwzv.F90',
'src/OCE/DYN/wet_dry.F90',
'src/OCE/LDF/ldfdyn.F90',
'src/OCE/LDF/ldftra.F90',
'src/OCE/LDF/ldfslp.F90',
'src/OCE/LDF/ldfeke.F90',
'src/OCE/ZDF/zdfphy.F90',
'src/OCE/ZDF/zdfosm.F90',
'src/OCE/ZDF/zdfkpp.F90',
'src/OCE/TRA/trabbc.F90',
'src/OCE/TRA/trabbl.F90',
'src/OCE/TRA/tradmp.F90',
'src/OCE/TRA/traadv.F90',
'src/OCE/TRA/traadv_cen.F90',
'src/OCE/TRA/traadv_fct.F90',
'src/OCE/TRA/traadv_mus.F90',
'src/OCE/TRA/traadv_qck.F90',
'src/OCE/TRA/traadv_ubs.F90',
'src/OCE/TRA/traadv_mus.F90',
'src/OCE/TRA/trazdf.F90',
'src/OCE/TRA/traldf.F90',
'src/OCE/TRA/traldf_iso.F90',
'src/OCE/TRA/traldf_triad.F90',
]
out=pathlib.Path('research/literature')
for p in wanted:
    name='NEMO_'+p.replace('/','_')
    import urllib.parse
    url=api+'/'+urllib.parse.quote(p,safe='')+'/raw?ref='+branch
    r=requests.get(url,timeout=30)
    if r.ok:
        (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
    else: print('ERR',name,r.status_code)
