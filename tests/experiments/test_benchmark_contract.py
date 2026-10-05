"""Frozen scope, calendar and negative controls, without model execution."""

from copy import deepcopy

import pytest

from zhenmode.evaluation.protocols import digest, load_json
from zhenmode.execution.benchmark import contract, main, plan, validate_frozen


def test_freeze_installed_contract_refuses_overwrite_and_modified_mechanisms(tmp_path):
    filename = tmp_path / "physical.json"
    assert main(["freeze", "--output", str(filename)]) == 0
    assert validate_frozen(load_json(filename))
    with pytest.raises(SystemExit):
        main(["freeze", "--output", str(filename)])
    for field in ("version", "temperature_humidity_height_m"):
        changed = deepcopy(contract())
        changed["forcing"][field] = "wrong"
        # Even a recomputed hash must not make different physics pass.
        with pytest.raises(ValueError, match="differs"):
            validate_frozen({"contract": changed, "contract_sha256": digest(changed)})
    changed = deepcopy(contract())
    changed['schema_version'] = True
    with pytest.raises(ValueError, match='differs'):
        validate_frozen({'contract': changed, 'contract_sha256': digest(contract())})


def test_same_physics_short_profiles_never_qualify_climate():
    small = plan("integration-6h", "zhenmode", 600)
    full = plan("climate-6cycle", "mom6", 1800)
    assert small["steps"] == 36
    # 1958-2018 has 15 leap days, six identical forcing cycles.
    assert full["duration_seconds"] == 6 * (61 * 365 + 15) * 86400
    assert small["physics_sha256"] == full["physics_sha256"]
    assert not small["climate_qualification"] and not full["climate_qualification"]
    assert not full["execution_ready"]


@pytest.mark.parametrize("profile,method,dt", [
    ("unknown", "mom6", 600), ("integration-6h", "unknown", 600),
    ("integration-6h", "mom6", 601), ("integration-6h", "mom6", True),
])
def test_invalid_plan_rejected(profile, method, dt):
    with pytest.raises(ValueError):
        plan(profile, method, dt)
