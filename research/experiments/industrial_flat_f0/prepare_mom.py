"""Build only the frozen coarse MOM files; no model launch or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

from mom_diagnostics import write_native_diag_table
from prepare_inputs import native_arrays, write_mom_initial
from resolved_config import BASE_MOM_CONTRACT


def settings(preflight):
    values = dict(BASE_MOM_CONTRACT)
    values.update(SOUTHLAT=0., WESTLON=0., ISOTROPIC=False, NIHALO=4, NJHALO=4,
                  BOUSSINESQ=True, SPLIT=True, SPLIT_RK2B=False,
                  DT=100., DT_THERM=100., DT_FORCING=100., DTBT=100.,
                  USE_REGRIDDING=False, BULKMIXEDLAYER=False, ADIABATIC=True,
                  COORD_CONFIG='none', LIGHTEST_DENSITY=1025., GFS=9.81,
                  P_REF=0., P_REF_LINEAR_EOS=0.,
                  THICKNESS_CONFIG='file', THICKNESS_FILE='initial.nc',
                  INTERFACE_IC_VAR='eta', ADJUST_THICKNESS=False,
                  TS_CONFIG='file', TS_FILE='initial.nc',
                  TEMP_IC_VAR='PTEMP', SALT_IC_VAR='SALT', INPUTDIR='INPUT',
                  THICKNESSDIFFUSE=False, MIXEDLAYER_RESTRAT=False,
                  USE_NEUTRAL_DIFFUSION=False, KH=0., AH=0.,
                  KH_VEL_SCALE=0., AH_VEL_SCALE=0., SMAGORINSKY_KH=False,
                  SMAGORINSKY_AH=False, SMAG_BI_CONST=0.,
                  KHTH=0., KHTR=0., KV=0., KD=0.,
                  HMIX_FIXED=0., KV_ML_INVZ2=0., DIRECT_STRESS=False,
                  DYNAMIC_VISCOUS_ML=False, FIXED_DEPTH_LOTW_ML=False,
                  LOTW_VISCOUS_ML_FLOOR=False,
                  BOTTOMDRAGLAW=True, CDRAG=0., DRAG_BG_VEL=0., HBBL=10.,
                  FRAZIL=False, DO_GEOTHERMAL=False,
                  USE_IDEAL_AGE_TRACER=False, SAVE_INITIAL_CONDS=True,
                  IC_OUTPUT_FILE='native_initial', RESTART_CONTROL=0,
                  MAXTRUNC=0, MAXCPU=600., DAYMAX=(100. if preflight else 32000.)/86400.,
                  ENERGYSAVEDAYS=1000./86400., DEBUG=True)
    return values


def prepare(directory, preflight):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'INPUT').mkdir()
    (directory / 'RESTART').mkdir()
    write_mom_initial(directory / 'INPUT/initial.nc', native_arrays(0.))
    values = settings(preflight)
    lines = []
    for name, value in values.items():
        formatted = ('"' + value + '"') if isinstance(value, str) else str(value)
        lines.append(name + ' = ' + formatted)
    (directory / 'MOM_input').write_text('\n'.join(lines) + '\n')
    (directory / 'MOM_override').write_text('')
    (directory / 'input.nml').write_text(
        "&mom_input_nml\n output_directory='./'\n input_filename='n'\n"
        " restart_input_dir='INPUT/'\n restart_output_dir='RESTART/'\n"
        " parameter_filename='MOM_input','MOM_override'\n/\n"
        "&diag_manager_nml\n/\n&fms_nml\n clock_grain='ROUTINE'\n"
        " clock_flags='SYNC'\n domains_stack_size=955296\n stack_size=0\n/\n")
    write_native_diag_table(directory / 'diag_table', 100 if preflight else 1000)
    identity = {str(path.relative_to(directory)): dict(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path in directory.rglob('*') if path.is_file()}
    (directory / 'requested_config.json').write_text(json.dumps(dict(
        preflight=preflight, simulated_seconds=100 if preflight else 32000, settings=values,
        input_identities=identity, qualification_passed=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true', help='independent one-step qualification run')
    args = parser.parse_args()
    prepare(args.output_directory, args.preflight)
