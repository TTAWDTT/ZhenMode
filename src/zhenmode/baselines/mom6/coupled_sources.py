"""Explicit latent-energy adaptation and clean pinned coupled build sources.

No upstream cache is changed. This bottom-interface patch does not qualify
the atmosphere/ice stock ledger, or the full ice thermodynamic experiment.
"""
from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

from zhenmode.provenance.sources import load_json, sha256_file

PINS = load_json(Path(__file__).with_name('omip2-pins.json'))
BUILD_CONTROLS = ('Makefile', 'shared', 'ice_ocean_SIS2/Makefile', 'ice_ocean_SIS2/configure.ice_ocean.ac')


def corrected_latent_energy(path, source):
    """Carry actual energy separately from water mass and sensible heat."""
    if path not in PINS['latent_energy_sources'] or sha256_bytes(source) != PINS['latent_energy_sources'][path]:
        raise ValueError('latent-energy source does not match the audited pin')
    text = source.decode('utf-8')

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('ambiguous pinned latent-energy patch: ' + old)
        text = text.replace(old, new)

    if path.endswith('SIS_fast_thermo.F90'):
        replace('flux_lh(i,j,0) = LatHtVap * evap(i,j,0)',
                '! OMIP Gill open water; native ice/snow latent physics is unchanged.\n'
                '    flux_lh(i,j,0) = (2.5008e6 - 2300.0*US%C_to_degC*sOSS%SST_C(i,j)) &\n'
                '                       * US%J_kg_to_Q * evap(i,j,0)')
    elif path.endswith('MOM_surface_forcing_gfdl.F90'):
        replace('type, public :: ice_ocean_boundary_type',
                'type, public :: ice_ocean_boundary_type\n'
                '  real, pointer, dimension(:,:) :: latent_flux =>NULL() !< upward actual latent energy [W m-2]')
        old = ('    if (associated(IOB%q_flux)) then\n'
               '      fluxes%latent(i,j) = fluxes%latent(i,j) - IOB%q_flux(i-i0,j-j0)*kg_m2_s_conversion * CS%latent_heat_vapor\n'
               '      fluxes%latent_evap_diag(i,j) = -G%mask2dT(i,j) * IOB%q_flux(i-i0,j-j0)*kg_m2_s_conversion * CS%latent_heat_vapor\n'
               '    endif')
        replace(old, '    if (.not.associated(IOB%latent_flux)) &\n'
                '      call MOM_error(FATAL,"OMIP requires explicit ice-to-ocean latent energy")\n'
                '    fluxes%latent(i,j) = fluxes%latent(i,j) - US%W_m2_to_QRZ_T*IOB%latent_flux(i-i0,j-j0)\n'
                '    fluxes%latent_evap_diag(i,j) = -G%mask2dT(i,j)*US%W_m2_to_QRZ_T*IOB%latent_flux(i-i0,j-j0)\n'
                '    if (CS%check_no_land_fluxes) &\n'
                "      call check_mask_val_consistency(IOB%latent_flux(i-i0,j-j0), G%mask2dT(i,j), i, j, 'latent_flux', G)")
    else:
        # The allocation belongs after ocean domain bounds are obtained.
        replace('    !ML ocean only requires t, q, lw, sw, fprec, calving',
                '    allocate(ice_ocean_boundary%latent_flux(is:ie,js:je)); ice_ocean_boundary%latent_flux = 0.0\n'
                '    !ML ocean only requires t, q, lw, sw, fprec, calving')
        # The pinned SIS ice_model.F90 exports IOF internal energy using
        # US%QRZ_T_to_W_m2; the public Ice%flux_lh is already physical W/m2.
        replace('    call mpp_clock_end(fluxIceOceanClock)',
                '    call flux_ice_to_ocean_redistribute(Ice, Ocean, Ice%flux_lh, &\n'
                '         Ice_Ocean_Boundary%latent_flux, Ice_Ocean_Boundary%xtype, do_area_weighted_flux)\n\n'
                '    call mpp_clock_end(fluxIceOceanClock)')
        replace('    call data_override(\'OCN\', \'q_flux\',    Ice_Ocean_Boundary%q_flux   , Time )',
                '    ! Offline overrides of mass or energy must not silently separate this pair.\n'
                '    call data_override(\'OCN\', \'q_flux\',    Ice_Ocean_Boundary%q_flux   , Time )\n'
                '    call data_override(\'OCN\', \'latent_flux\', Ice_Ocean_Boundary%latent_flux, Time )')
        replace('Ice%flux_t - Ice%flux_q*HLV', 'Ice%flux_t - Ice%flux_lh')
        replace('Ice_Ocean_Boundary%t_flux - Ice_Ocean_Boundary%q_flux*HLV',
                'Ice_Ocean_Boundary%t_flux - Ice_Ocean_Boundary%latent_flux')
    return text.encode('utf-8')


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def stage_sources(examples, destination, patches):
    """Extract only committed build inputs; ignored config.mk cannot enter make."""
    destination.mkdir(parents=True, exist_ok=False)
    entries = [('', PINS['examples']['commit'], BUILD_CONTROLS)] + [
        (path, commit, ()) for path, commit in (PINS['components'] | PINS['nested']).items()]
    for index, (relative, commit, paths) in enumerate(entries):
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        archive = destination.parent / f'source-{index}.tar'
        subprocess.run(['git', '-C', str(examples / relative), 'archive', '--format=tar',
                        f'--output={archive}', commit, *paths], check=True, timeout=60)
        with tarfile.open(archive) as stream:
            stream.extractall(target, filter='data')
        archive.unlink()  # owned temporary, pinned bytes remain in source checkpoint
    for relative, content in patches.items():
        path = destination / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError('staged patch target is not a committed regular file: ' + relative)
        path.write_bytes(content)
    identity = source_tree(destination)
    (destination.parent / 'executed-sources.json').write_text(json.dumps(identity, indent=2) + '\n')
    return identity


def expected_sources(examples, identities, patches):
    """Derive staged byte/mode expectations from checked pins, not mutable receipts."""
    identity = {}
    for component, source in identities.items():
        prefix = '' if component == 'examples' else component + '/'
        for name, checksum in source['tracked_files'].items():
            if not prefix and not any(name == path or name.startswith(path + '/') for path in BUILD_CONTROLS):
                continue
            relative = prefix + name
            original = examples / relative
            identity[relative] = {'link': original.readlink().as_posix()} if original.is_symlink() else {
                'sha256': sha256_bytes(patches[relative]) if relative in patches else checksum,
                'executable': bool(original.stat().st_mode & 0o111)}
        # A committed directory symlink is not is_file(), so the older
        # tracked-byte inventory excludes it. Bind its target to the Git blob,
        # rather than accepting an extra entry from the staged inventory.
        directory = examples if component == 'examples' else examples / component
        tree = subprocess.check_output(['git', '-C', str(directory), 'ls-tree', '-r', '-z',
                                        source['commit']]).decode('utf-8')
        for record in tree.split('\0'):
            if not record:
                continue
            attributes, name = record.split('\t', 1)
            mode, kind, blob = attributes.split()
            if mode != '120000' or (not prefix and not any(
                    name == path or name.startswith(path + '/') for path in BUILD_CONTROLS)):
                continue
            target = subprocess.check_output(['git', '-C', str(directory), 'cat-file', kind, blob]).decode('utf-8')
            if (directory / name).readlink().as_posix() != target:
                raise ValueError('original pinned symlink target changed: ' + prefix + name)
            identity[prefix + name] = {'link': target}
    return identity


def source_tree(directory):
    """Hash actual staged sources, executable modes and links, not a facade."""
    identity = {}
    for path in sorted(directory.rglob('*')):
        name = path.relative_to(directory).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('staged source link escapes the source tree: ' + name)
            identity[name] = {'link': path.readlink().as_posix()}
        elif path.is_file():
            identity[name] = {'sha256': sha256_file(path), 'executable': bool(path.stat().st_mode & 0o111)}
    return identity
