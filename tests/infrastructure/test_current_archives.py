"""Current snapshots contain real implementations and moved independent helpers."""
import hashlib
import shutil
from pathlib import Path

import pytest

from ocean_solver.provenance.archives import current_source_files
from ocean_solver.provenance.sources import production_source_modules, source_paths
from tests.support.paths import REPOSITORY_ROOT


def test_current_archive_resolves_legacy_keys_and_contains_complete_source_closure():
    files = current_source_files(REPOSITORY_ROOT, [
        'src/config.py', 'tests/test_rstar_weak_time.py', 'tests/_helpers.py',
    ])
    assert files['src/config.py'] == REPOSITORY_ROOT / 'src/compat/config.py'
    assert files['tests/test_rstar_weak_time.py'] == REPOSITORY_ROOT / 'tests/research/rstar/test_rstar_weak_time.py'
    assert files['tests/_helpers.py'] == REPOSITORY_ROOT / 'tests/support/grid.py'
    assert files['tests/support/rstar/weak_time.py'].is_file()
    required = source_paths(REPOSITORY_ROOT / 'src', production_source_modules())
    assert set(required.values()) <= set(files.values())
    assert len({key for key in files if key.startswith('src/ocean_solver/')}) == 96
    # Every key hashes actual current bytes; aliases and implementations differ.
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    assert hashes['src/config.py'] == hashes['src/compat/config.py']
    assert hashes['src/config.py'] != hashes['src/ocean_solver/configuration.py']


def test_current_archive_rejects_missing_implementation_instead_of_hash_fallback(tmp_path):
    shutil.copytree(REPOSITORY_ROOT / 'src', tmp_path / 'src')
    (tmp_path / 'docs').mkdir()
    shutil.copyfile(REPOSITORY_ROOT / 'docs/source_test_layout.json', tmp_path / 'docs/source_test_layout.json')
    (tmp_path / 'src/ocean_solver/data/quality.py').unlink()
    with pytest.raises(ValueError, match='missing required source: ocean_solver/data/quality'):
        current_source_files(tmp_path, ['src/config.py'])


def test_current_archive_rejects_missing_declared_test():
    with pytest.raises(ValueError, match='missing declared current source'):
        current_source_files(REPOSITORY_ROOT, ['tests/missing_test.py'])


@pytest.mark.parametrize("producer", ["verify_production_restart_cpu", "verify_debug_integration"])
def test_current_producer_hashes_real_complete_source_closure_without_model_run(producer):
    import importlib

    owner = importlib.import_module("scripts." + producer)
    hashes = owner.source_hashes()
    files = current_source_files(REPOSITORY_ROOT, [Path(owner.__file__).resolve()])
    assert hashes == {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    assert len([name for name in hashes if name.startswith("src/ocean_solver/")]) == 96
    assert len([name for name in hashes if name.startswith("src/compat/")]) == 40
    assert "tests/support/driver.py" in hashes
    assert "tests/support/grid.py" in hashes


@pytest.mark.parametrize("logical,actual", [
    ("src/config.py", "src/compat/config.py"),
    ("tests/_driver_helpers.py", "tests/support/driver.py"),
    ("tests/_helpers.py", "tests/support/grid.py"),
    ("tests/test_cgrid_pressure_work.py", "tests/candidates/fv/test_cgrid_pressure_work.py"),
    ("src/ocean_solver/configuration.py", "src/ocean_solver/configuration.py"),
])
def test_current_source_resolver_maps_logical_labels_to_actual_files(logical, actual):
    from ocean_solver.provenance import archives

    assert archives.current_source_path(REPOSITORY_ROOT, logical) == (REPOSITORY_ROOT / actual).resolve()


@pytest.mark.parametrize("invalid", ["../outside.py", "src/missing_module.py"])
def test_current_source_resolver_rejects_traversal_and_missing_files(invalid):
    from ocean_solver.provenance import archives

    with pytest.raises(ValueError):
        archives.current_source_path(REPOSITORY_ROOT, invalid)


def test_current_source_resolver_rejects_absolute_and_mapped_outside_paths(tmp_path, monkeypatch):
    from ocean_solver.provenance import archives

    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        archives.current_source_path(root, str(outside))
    monkeypatch.setattr(archives, "current_source_files", lambda *args: {"src/config.py": outside})
    with pytest.raises(ValueError):
        archives.current_source_path(root, "src/config.py")


@pytest.mark.parametrize("corruption", [None, "different_bytes", "missing_canonical", "missing_marker"])
def test_current_manifest_requires_actual_hashes_and_complete_current_closure(corruption):
    from ocean_solver.provenance import archives

    files = current_source_files(REPOSITORY_ROOT, [])
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    if corruption == "different_bytes":
        hashes["src/config.py"] = "0" * 64
    elif corruption == "missing_canonical":
        hashes.pop("src/ocean_solver/configuration.py")
    elif corruption == "missing_marker":
        hashes = {name: value for name, value in hashes.items()
                  if name != "docs/source_test_layout.json" and not name.startswith("src/ocean_solver/")}
    if corruption:
        with pytest.raises(ValueError, match="source mismatch|incomplete current source manifest"):
            archives.verify_current_source_hashes(REPOSITORY_ROOT, hashes)
    else:
        archives.verify_current_source_hashes(REPOSITORY_ROOT, hashes)


def test_rotation_probe_keeps_current_ledger_and_executed_operator_separate():
    from research.experiments.cgrid_hydrostatic_momentum import probe_partial_rotation as probe

    operator_bytes = Path(probe.cgrid_momentum.__file__).read_bytes()
    ledger, executed = probe.source_provenance(operator_bytes)
    assert ledger["src/cgrid_momentum.py"] == hashlib.sha256(
        (REPOSITORY_ROOT / "src/compat/cgrid_momentum.py").read_bytes()).hexdigest()
    assert executed == {"kind": "current_file", "path": "src/ocean_solver/candidates/fv/momentum.py",
                        "sha256": hashlib.sha256(operator_bytes).hexdigest()}
    retained_blob = b"independent frozen-operator bytes"
    frozen_ledger, frozen = probe.source_provenance(retained_blob, "1" * 40)
    assert frozen_ledger == ledger
    assert frozen == {"kind": "git_blob", "revision": "1" * 40, "path": "src/cgrid_momentum.py",
                      "sha256": hashlib.sha256(retained_blob).hexdigest()}


@pytest.mark.parametrize("tampered", [False, True])
def test_current_attribution_cli_verifies_new_receipt_sources_without_model(tmp_path, monkeypatch, tampered):
    import json
    import sys

    from research.experiments.nonlinear_process_budgets import analyze_attribution
    from scripts.verify_debug_integration import source_hashes

    hashes = source_hashes()
    if tampered:
        hashes["src/config.py"] = "0" * 64
    report = {"status": "complete", "arguments": {"audit_budget": True},
              "provenance": {"source_sha256": hashes}, "cases": []}
    input_path = tmp_path / "metadata_only.json"
    output_path = tmp_path / "attribution.json"
    input_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["analyze_attribution", "--input", str(input_path),
                                      "--out", str(output_path)])
    if tampered:
        with pytest.raises(ValueError, match="runtime source mismatch"):
            analyze_attribution.main()
        assert not output_path.exists()
    else:
        analyze_attribution.main()
        result = json.loads(output_path.read_text(encoding="utf-8"))
        assert result["runtime_source_hashes_match"] is True
        assert result["cases"] == []
