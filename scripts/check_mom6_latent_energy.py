"""Compile actual pinned SIS/MOM/coupler energy excerpts against SI controls.

This checks source expressions, units, sign and two bottom exchange ledgers.
It is NOT a coupled time step, atmosphere ledger or full ice budget proof.
Run inside the one-CPU/180s/4GiB supervisor; full interfaces also need a build.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from zhenmode.baselines.mom6.coupled_sources import PINS, corrected_latent_energy
from zhenmode.provenance.sources import sha256_file


def between(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)]


def check(examples, fms_build, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {path: corrected_latent_energy(path, (Path(examples)/path).read_bytes()).decode()
               for path in PINS['latent_energy_sources']}
    sis = next(text for path,text in sources.items() if path.endswith('SIS_fast_thermo.F90'))
    mom = next(text for path,text in sources.items() if path.endswith('MOM_surface_forcing_gfdl.F90'))
    coupler = next(text for path,text in sources.items() if path.endswith('ice_ocean_flux_exchange.F90'))
    producer = between(sis,'    flux_lh(i,j,0) = (2.5008e6','  enddo ; enddo')
    consumer = between(mom,'    fluxes%latent(i,j) = 0.0','    if (associated(IOB%sw_flux_vis_dir))')
    ice_stock = between(coupler,'    from_dq = Dt_cpl * SUM( Ice%area * ( &','    Ice_stock(ISTOCK_HEAT)')
    ocean_stock = between(coupler,'    from_dq = SUM( ocean_cell_area * wet *( Ice_Ocean_Boundary%sw_flux_vis_dir','    Ocn_stock(ISTOCK_HEAT)')
    # Stub declarations only supply the expression's input/output storage. The
    # expected SI values below do not call any model/adapter flux implementation.
    driver = '''module storage
implicit none
integer,parameter::FATAL=1
type units
real::C_to_degC=2.,J_kg_to_Q=.01,W_m2_to_QRZ_T=.02
end type
type sea
real::SST_C(1,1)
end type
type geometry
real::mask2dT(1,1)=1
end type
type boundary
real,pointer::fprec(:,:)=>null(),calving(:,:)=>null(),latent_flux(:,:)=>null(),q_flux(:,:)=>null()
real::sw_flux_vis_dir(1,1)=0,sw_flux_vis_dif(1,1)=0,sw_flux_nir_dir(1,1)=0,sw_flux_nir_dif(1,1)=0
real::lw_flux(1,1)=0,t_flux(1,1)=0
end type
type ice_fields
real::area(1,1)=3,fprec(1,1)=0,calving(1,1)=0,flux_t(1,1)=0,flux_q(1,1)=0,flux_lh(1,1)=0
real::flux_sw_vis_dir(1,1)=0,flux_sw_vis_dif(1,1)=0,flux_sw_nir_dir(1,1)=0,flux_sw_nir_dif(1,1)=0,flux_lw(1,1)=0
end type
type flux_fields
real::latent(1,1),latent_fprec_diag(1,1),latent_frunoff_diag(1,1),latent_evap_diag(1,1)
end type
type controls
real::latent_heat_fusion
logical::check_no_land_fluxes=.false.
end type
contains
subroutine MOM_error(level,message)
integer::level
character(*)::message
write(*,*)message
error stop 9
end subroutine
subroutine check_mask_val_consistency(value,mask,i,j,label,G)
real::value,mask
integer::i,j
character(*)::label
type(geometry)::G
end subroutine
end module
program check_energy
use storage
use constants_mod,only:HLF
implicit none
type(units)::US
type(sea)::sOSS
type(geometry)::G
type(boundary)::IOB,Ice_Ocean_Boundary
type(ice_fields)::Ice
type(flux_fields)::fluxes
type(controls)::CS
real::flux_lh(1,1,0:0),evap(1,1,0:0),from_dq,Dt_cpl,kg_m2_s_conversion
real::ocean_cell_area(1,1),wet(1,1),temperature(4),mass(4)
integer::i,j,i0,j0,k
i=1;j=1;i0=0;j0=0;Dt_cpl=1800;kg_m2_s_conversion=2
CS%latent_heat_fusion=HLF*US%J_kg_to_Q
allocate(IOB%latent_flux(1,1),IOB%fprec(1,1),IOB%calving(1,1),IOB%q_flux(1,1))
IOB%fprec=0;IOB%calving=0;IOB%q_flux=999
temperature=[-1.,0.,10.,30.];mass=[1e-5,0.,-1e-5,2e-5]
do k=1,4
sOSS%SST_C=temperature(k)/US%C_to_degC;evap(1,1,0)=mass(k)
''' + producer + '''
write(*,'(a,es25.17)') 'WATER ',flux_lh(1,1,0)/US%J_kg_to_Q
IOB%latent_flux=flux_lh(1,1,0)/US%J_kg_to_Q
''' + consumer + '''
write(*,'(a,es25.17)') 'MOM ',fluxes%latent(1,1)/US%W_m2_to_QRZ_T
Ice%flux_lh=IOB%latent_flux
Ice_Ocean_Boundary%latent_flux=>IOB%latent_flux
Ice_Ocean_Boundary%fprec=>IOB%fprec
Ice_Ocean_Boundary%calving=>IOB%calving
Ice_Ocean_Boundary%q_flux=>IOB%q_flux
ocean_cell_area=Ice%area;wet=1
''' + ice_stock + '''
write(*,'(a,es25.17)') 'ICE_STOCK ',from_dq
''' + ocean_stock + '''
write(*,'(a,es25.17)') 'OCEAN_STOCK ',from_dq*Dt_cpl
enddo
! Condensation/precipitation, a deliberately unrelated mass flux, and dry mask.
IOB%latent_flux=-10;IOB%fprec=1e-5;IOB%calving=2e-5
''' + consumer + '''
write(*,'(a,2(es25.17,1x))') 'FROZEN ',fluxes%latent(1,1)/US%W_m2_to_QRZ_T,HLF
G%mask2dT=0
''' + consumer + '''
write(*,'(a,es25.17)') 'DRY ',fluxes%latent(1,1)
end program
'''
    (output/'driver.f90').write_text(driver)
    command=['gfortran','-O0','-ffree-line-length-none','-fdefault-real-8','-fdefault-double-8',
             '-I'+str(Path(fms_build).resolve()),str(output/'driver.f90'),'-o',str(output/'check_energy')]
    compilation=subprocess.run(command,cwd=output,text=True,capture_output=True,timeout=60)
    (output/'compile.txt').write_text(compilation.stdout+compilation.stderr)
    compilation.check_returncode()
    execution=subprocess.run([str(output/'check_energy')],cwd=output,text=True,capture_output=True,timeout=20)
    (output/'stdout.txt').write_text(execution.stdout)
    (output/'stderr.txt').write_text(execution.stderr)
    execution.check_returncode()
    values={key:[] for key in ('WATER','MOM','ICE_STOCK','OCEAN_STOCK','FROZEN','DRY')}
    for line in execution.stdout.splitlines():
        parts=line.split()
        if parts and parts[0] in values:
            values[parts[0]].append([float(v) for v in parts[1:]])
    expected=np.array([25.031,0,-24.778,48.636])
    np.testing.assert_allclose(np.array(values['WATER']).ravel(),expected,rtol=1e-14,atol=1e-12)
    np.testing.assert_allclose(np.array(values['MOM']).ravel(),-expected,rtol=1e-14,atol=1e-12)
    for key in ('ICE_STOCK','OCEAN_STOCK'):
        np.testing.assert_allclose(np.array(values[key]).ravel(),-expected*3*1800,rtol=1e-14,atol=1e-9)
    frozen,latent_fusion=values['FROZEN'][0]
    np.testing.assert_allclose(frozen,10-3e-5*latent_fusion,rtol=1e-14,atol=1e-12)
    assert values['DRY']==[[0.]]
    report={'status':'PASS','scope':'compiled_actual_energy_excerpts_not_coupled_step_or_atmosphere_ice_budget',
            'source_identity':{path:{'upstream':PINS['latent_energy_sources'][path],
                                   'adapted_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}
                               for path,text in sources.items()},
            'harness_sha256':sha256_file(__file__),'driver_sha256':sha256_file(output/'driver.f90'),
            'fms_constants_sha256':sha256_file(Path(fms_build)/'constants_mod.mod'),
            'binary_sha256':sha256_file(output/'check_energy'),'command':command,'actual':values,
            'execution_ready':False,'climate_qualification':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples',required=True)
    parser.add_argument('--fms-build',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(check(args.examples,args.fms_build,args.output),indent=2))
    except Exception as error:
        directory=Path(args.output)
        if directory.is_dir() and not (directory/'report.json').exists():
            with (directory/'failure.json').open('x') as stream:
                json.dump({'status':'FAIL','reason':str(error),'scope':'energy_excerpts_only'},stream,indent=2)
        raise
