"""Candidate pins/configuration do not confer run or climate qualification."""
import pytest

from zhenmode.baselines.mom6.omip2 import PINS, corrected_configuration, preparation_plan


def test_pins_cover_coupled_driver_and_nested_dependencies():
    assert {'src/MOM6', 'src/SIS2', 'src/FMS2', 'src/coupler', 'src/ice_param'} <= PINS['components'].keys()
    assert len(PINS['nested']) == 2
    assert all(len(commit) == 40 for commit in (PINS['components'] | PINS['nested']).values())
    assert len(PINS['configuration_files']) == 5
    assert PINS['coupler_audit']['drag'] == 'LY2004'


def test_candidate_plan_remains_blocked():
    short = preparation_plan('integration-6h')
    long = preparation_plan('climate-6cycle')
    assert short['physics_sha256'] == long['physics_sha256']
    assert not short['execution_ready'] and not long['climate_qualification']
    assert short['build_plan']['jobs'] == 1
    assert short['mechanism_gaps'] and 'compiled reference' in short['mechanism_gaps'][0]


def test_corrections_are_concrete_without_hiding_remaining_inputs():
    source = {'input.nml': "calendar = 'julian',\ndays = 1,\nhours = 0,\n&surface_flux_nml\nncar_ocean_flux=.true.\n/\n",
              'MOM_saltrestore': 'FLUXCONST = .1667 ! m/day\nADJUST_NET_FRESH_WATER_TO_ZERO = True\nMAX_DELTA_SRESTORE = 5\nSALT_RESTORE_FILE = "old.nc"\n'}
    prepared = corrected_configuration(source, 'integration-6h')
    assert "calendar = 'gregorian'," in prepared['input.nml']
    assert 'hours = 6,' in prepared['input.nml']
    assert 'ADJUST_NET_FRESH_WATER_TO_ZERO = False' in prepared['MOM_saltrestore']
    assert 'old.nc' in prepared['MOM_saltrestore']
    assert 'gust_const=0.5' in prepared['input.nml']
    assert '.1667' in source['MOM_saltrestore']  # upstream evidence is not rewritten
    with pytest.raises(ValueError, match='unambiguous'):
        corrected_configuration(source | {'input.nml': source['input.nml'] + 'days = 2,\n'}, 'integration-6h')
    with pytest.raises(ValueError, match='first-cycle'):
        corrected_configuration(source, 'climate-6cycle')


def test_source_patch_refuses_unpinned_or_already_changed_implementation():
    from zhenmode.baselines.mom6.omip2 import corrected_coefficients

    with pytest.raises(ValueError, match='implementation changed'):
        corrected_coefficients(b'subroutine ncar_ocean_fluxes\nend subroutine')


def test_latent_patch_refuses_other_source_bytes():
    from zhenmode.baselines.mom6.coupled_sources import corrected_latent_energy

    for path in PINS['latent_energy_sources']:
        with pytest.raises(ValueError, match='audited pin'):
            corrected_latent_energy(path, b'uncertified source')
    with pytest.raises(ValueError, match='audited pin'):
        corrected_latent_energy('unregistered.F90', b'uncertified source')


@pytest.mark.skipif(__import__('os').name != 'posix', reason='native Git modes and archive semantics')
def test_staging_excludes_ignored_build_controls_and_binds_actual_source(tmp_path, monkeypatch):
    import subprocess

    from zhenmode.baselines.mom6 import coupled_sources

    cache = tmp_path / 'cache'
    cache.mkdir()
    for name in ('Makefile', 'shared/config/Libs.mk', 'ice_ocean_SIS2/Makefile',
                 'ice_ocean_SIS2/configure.ice_ocean.ac'):
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('committed build control\n')
    (cache / 'driver').write_text('#!/bin/sh\nexit 0\n')
    (cache / 'driver').chmod(0o755)
    # shared executable is part of the selected build-control archive.
    (cache / 'driver').rename(cache / 'shared/driver')
    (cache / 'shared/directory-link').symlink_to('config', target_is_directory=True)
    subprocess.run(['git', 'init', str(cache)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(cache), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(cache), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    commit = subprocess.check_output(['git', '-C', str(cache), 'rev-parse', 'HEAD'], text=True).strip()
    (cache / 'config.mk').write_text('FCFLAGS = injected\n')
    (cache / 'shared/config/ignored.mk').write_text('unregistered control\n')
    monkeypatch.setattr(coupled_sources, 'PINS', {'examples': {'commit': commit}, 'components': {}, 'nested': {}})
    from zhenmode.provenance.sources import sha256_file
    tracked = {name: sha256_file(cache / name) for name in subprocess.check_output(
        ['git', '-C', str(cache), 'ls-files'], text=True).splitlines() if (cache / name).is_file()}
    expected = coupled_sources.expected_sources(cache, {'examples': {'tracked_files': tracked, 'commit': commit}}, {})
    destination = tmp_path / 'work/sources'
    identity = coupled_sources.stage_sources(cache, destination, {})
    assert not (destination / 'config.mk').exists()
    assert not (destination / 'shared/config/ignored.mk').exists()
    assert identity['shared/driver']['executable']
    assert coupled_sources.source_tree(destination) == identity
    assert identity == expected
    assert expected['shared/directory-link'] == {'link': 'config'}
    (destination / 'shared/config/Libs.mk').write_text('changed actual build input\n')
    assert coupled_sources.source_tree(destination) != identity
    # Replacing a mutable inventory together with the input cannot change the
    # expectation independently derived from the checked original pin.
    forged = coupled_sources.source_tree(destination)
    (destination.parent / 'executed-sources.json').write_text(__import__('json').dumps(forged))
    assert forged != expected
    (destination / 'escaped.F90').symlink_to(cache / 'Makefile')
    with pytest.raises(ValueError, match='escapes'):
        coupled_sources.source_tree(destination)


@pytest.mark.parametrize('wall_seconds', [0, 181, 180.5])
def test_coupled_build_rejects_out_of_budget_before_io(wall_seconds):
    from zhenmode.baselines.mom6.omip2 import build_segment

    with pytest.raises(ValueError, match='1-180s'):
        build_segment('missing', 'unused', wall_seconds=wall_seconds)


def test_truncated_dependency_recovery_preserves_evidence_and_other_objects(tmp_path):
    from zhenmode.baselines.mom6.omip2 import _recover_dependencies

    build = tmp_path / 'build/ice_ocean_SIS2'
    build.mkdir(parents=True)
    dependency = build / 'Makefile.dep'
    original = '# Makefile.dep created by makedep\nall: coupler_main\npartial.o: source.F90\n'
    dependency.write_text(original)
    (build / 'finished.o').write_bytes(b'complete object')
    report = _recover_dependencies(tmp_path)
    assert (tmp_path / report['retained']).read_text() == original
    assert not dependency.exists() and (build / 'finished.o').read_bytes() == b'complete object'
    dependency.write_text(original + 'coupler_main: partial.o\n\t$(FC) -o $@ $<\n')
    assert _recover_dependencies(tmp_path) is None and dependency.exists()
    dependency.write_text('unexpected manual recipe')
    with pytest.raises(ValueError, match='unrecognized'):
        _recover_dependencies(tmp_path)
