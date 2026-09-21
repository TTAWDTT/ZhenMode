import requests, pathlib
repo='myroms/roms'; branch='develop'
wanted=[
'ROMS/Nonlinear/main3d.F',
'ROMS/Nonlinear/rhs3d.F',
'ROMS/Nonlinear/step2d_LF_AM3.h',
'ROMS/Nonlinear/step2d_FB_LF_AM3.h',
'ROMS/Nonlinear/step3d_t.F',
'ROMS/Nonlinear/step3d_uv.F',
'ROMS/Nonlinear/prsgrd.F',
'ROMS/Nonlinear/prsgrd40.h',
'ROMS/Nonlinear/t3dmix.F',
'ROMS/Nonlinear/t3dmix2_iso.h',
'ROMS/Nonlinear/t3dmix4_iso.h',
'ROMS/Nonlinear/uv3dmix.F',
'ROMS/Nonlinear/uv3dmix2_s.h',
'ROMS/Nonlinear/uv3dmix4_s.h',
'ROMS/Nonlinear/lmd_vmix.F',
'ROMS/Nonlinear/lmd_skpp.F',
'ROMS/Nonlinear/lmd_bkpp.F',
'ROMS/Nonlinear/mpdata_adiff.F',
'ROMS/Nonlinear/set_depth.F',
'ROMS/Nonlinear/rho_eos.F',
'ROMS/Nonlinear/omega.F',
'ROMS/Nonlinear/wvelocity.F',
'ROMS/Nonlinear/hmixing.F',
'ROMS/Nonlinear/bc_3d.F',
]
out=pathlib.Path('research/literature')
for p in wanted:
    name=repo.replace('/','_')+'_'+p.replace('/','_')
    r=requests.get(f'https://raw.githubusercontent.com/{repo}/{branch}/{p}',timeout=30)
    r.raise_for_status(); (out/name).write_text(r.text,encoding='utf-8'); print(name,len(r.text))
