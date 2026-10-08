"""Shared prepared fixed-partial initialization fixture."""
import json

from tests.support.data.native_initialization import native_inputs
from zhenmode.preparation.native_initial import prepare_native_initialization


def prepared_native_inputs(tmp_path, *, complete=True, deepest=7000.0):
    source, geometry, nodes, _, _ = native_inputs(tmp_path, unanchored=not complete)
    node_definition = json.loads(nodes.read_text())
    node_definition["z_nodes_m"][-1] = -deepest
    nodes.write_text(json.dumps(node_definition))
    native = tmp_path / "native"
    prepare_native_initialization(source, geometry, nodes, native)
    policy = tmp_path / "fd-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "horizontal_metric": "geographic_area_v1",
                "pressure_definition": "fixed_boussinesq_reference_v1",
                "inactive_ct_deg_c": 15.0,
                "inactive_sr_g_kg": 35.0,
            }
        )
    )
    return native, policy

