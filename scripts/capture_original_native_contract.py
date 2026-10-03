"""Capture source-bound original-interface evidence, never joint moving acceptance."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from ocean_solver.provenance.archives import current_source_files
from research.experiments.material_top_band.original_native import (
    OriginalNativeAudit,
    reference_pressure_kick,
)
from tests.support.original_native import (
    audit_native_receipt,
    make_original_native_case,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("Numerical evidence requires clean committed sources")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    state, params, grid = make_original_native_case()
    saved_fields = tuple(np.asarray(value).copy() for value in state)
    protocol = json.loads((root/"docs/original_small_grid_integration_protocol.json").read_text(encoding="utf-8"))
    frozen = protocol["case"]
    scalar_binding = {"dt":"macro_dt_s", "dt_bt":"fast_dt_s", "n_subcyc":"fast_substeps",
                      "nu_nsub":"nu_nsub", "T_ref":"T_ref_C", "S_ref":"S_ref_PSU",
                      "nu_h":"nu_h", "nu_v":"nu_v", "nu_bi":"nu_bi",
                      "kappa_h":"kappa_h", "kappa_v":"kappa_v", "kappa_bi":"kappa_bi",
                      "kappa_conv":"kappa_conv", "r_bot":"r_bot", "lambda_bulk":"lambda_bulk",
                      "tau_x_2d":"tau_x", "tau_y_2d":"tau_y", "Q_heat_2d":"Q_heat",
                      "T_atm_3d":"air_temperature_C", "column_geometry":"column_geometry",
                      "mode_split":"mode_split", "match_barotropic_transport":"match_barotropic_transport",
                      "process_time_scheme":"process_time_scheme", "conservative_kv":"conservative_kv",
                      "localize_conv":"localize_conv", "bottom_friction":"bottom_friction",
                      "use_scan":"use_scan", "fct_adv":"FCT", "projection_niter":"projection_niter",
                      "projection_rtol":"projection_rtol", "projection_preconditioner":"projection_preconditioner",
                      "projection_max_refinements":"projection_max_refinements"}
    for name, key in scalar_binding.items():
        if not np.all(np.asarray(getattr(params,name)) == frozen[key]):
            raise RuntimeError("Frozen case parameter mismatch: "+name)
    if (params.projection_niter_source != "explicit"
            or params.adv_nsub != frozen["factory_derived_counts"]["adv_nsub"]["expected"]
            or params.conv_nsub != frozen["factory_derived_counts"]["conv_nsub"]["expected"]):
        raise RuntimeError("Frozen original factory ownership/count mismatch")
    parameter_manifest = {}
    for name, value in params._asdict().items():
        if value is None:
            parameter_manifest[name] = None
        else:
            array = np.asarray(value)
            parameter_manifest[name] = {"dtype":array.dtype.str,"shape":list(array.shape),
                                       "sha256":hashlib.sha256(array.tobytes()).hexdigest()}
            if array.ndim == 0:
                parameter_manifest[name]["value"] = array.item()
    audit = OriginalNativeAudit(state, params, grid)
    receipt = audit.diagnose()
    kick = reference_pressure_kick(audit.view, params)
    gates = audit_native_receipt(audit.view,kick,receipt,grid)
    source_files = current_source_files(root, list(receipt.source_hashes))
    relative_paths = sorted({path.relative_to(root).as_posix() for path in source_files.values()})
    batch = subprocess.check_output(["git","cat-file","--batch"],cwd=root,
                                    input="".join("HEAD:"+name+"\n" for name in relative_paths).encode())
    blobs, offset = {}, 0
    for name in relative_paths:
        end = batch.index(b"\n",offset)
        object_id, kind, size = batch[offset:end].decode().split()
        if kind != "blob":
            raise RuntimeError("Source binding expected a committed blob")
        begin, count = end+1, int(size)
        content = batch[begin:begin+count]
        if batch[begin+count:begin+count+1] != b"\n":
            raise RuntimeError("Malformed git source batch")
        blobs[name] = (object_id,content)
        offset = begin+count+1
    source_bytes = {}
    for name, path in source_files.items():
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        object_id, blob = blobs[relative]
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != receipt.source_hashes[name]:
            raise RuntimeError("Executed source changed during witness")
        source_bytes[name] = {"checkout_path":relative,"actual_byte_sha256":actual_hash,
                              "byte_count":len(content),"CRLF_count":content.count(b"\r\n"),
                              "LF_count":content.count(b"\n"),"git_blob_sha1":object_id,
                              "git_blob_sha256":hashlib.sha256(blob).hexdigest(),
                              "equal_after_CRLF_to_LF":content.replace(b"\r\n",b"\n")==blob}
    result = {
        "contract":"original_nodal_reference_stock_and_fast_clock_handoff_gate",
        "frozen_reference_identity_gates":gates,
        "scientific_source_commit":head,
        "case_source":"tests/support/original_native.py::make_original_native_case",
        "backend":jax.default_backend(),
        "execution_versions":{"python":platform.python_version(),"numpy":np.__version__,"jax":jax.__version__,"jaxlib":version("jaxlib"),"pytest":version("pytest"),"ruff":version("ruff")},
        "grid_shape":[8,4,6],
        "native_state_authority":audit.view.authority,
        "parameter_digest":receipt.parameter_digest,
        "resolved_parameter_manifest":parameter_manifest,
        "frozen_case_parameter_binding_passed":True,
        "source_sha256":dict(receipt.source_hashes),
        "source_byte_metadata":source_bytes,
        "input_field_sha256":{name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
             for name,value in state._asdict().items()},
        "original_result_valid":receipt.original_valid,
        "original_stage_names":[row["name"] for row in receipt.stages],
        "original_generic_fast_calls":len(receipt.fast_calls),
        "per_fast_continuity_residual_max_m":gates["per_fast_continuity_residual_max_m"],
        "eta_change_abs_max_m":float(np.max(abs(receipt.original_state.eta-state.eta))),
        "u_change_abs_max_m_per_s":float(np.max(abs(receipt.original_state.u-state.u))),
        "T_change_abs_max_C":float(np.max(abs(receipt.original_state.T-state.T))),
        "deep_T_change_abs_max_C":float(np.max(abs(receipt.original_state.T[...,3:]-state.T[...,3:]))),
        "deep_IT_change_abs_max_C_m":float(np.max(abs(audit.view.h_ref[...,3:]*(receipt.original_state.T[...,3:]-state.T[...,3:])))),
        "deep_IS_change_abs_max_PSU_m":float(np.max(abs(audit.view.h_ref[...,3:]*(receipt.original_state.S[...,3:]-state.S[...,3:])))),
        "deep_reference_Mu_change_abs_max_kg_per_m_s":float(np.max(abs(1025.*audit.view.h_ref[...,3:]*(receipt.original_state.u[...,3:]-state.u[...,3:])))),
        "S_change_abs_max_PSU":float(np.max(abs(receipt.original_state.S-state.S))),
        "reference_pressure_impulse_abs_max_kg_m_per_s":float(np.max(abs(kick.after_momentum-kick.before_momentum))),
        "reference_pressure_transpose_roundoff_ratio_max":gates["identity_roundoff_ratios"]["pressure_transpose"],
        "reference_pressure_kick_delta_KE_J":kick.kinetic_change,
        "reference_pressure_kick_actual_impulse_work_J":kick.impulse_work,
        "reference_pressure_kick_dt_force_work_J":kick.pressure_work,
        "reference_pressure_kick_native_transport_work_J":kick.transport_work,
        "reference_pressure_kick_work_residual_J":kick.kinetic_change-kick.pressure_work,
        "reference_pressure_kick_work_roundoff_bound_J":gates["fixed_mass_work_roundoff_bound_J"],
        "convection_entry_RHS_preview_abs_max":receipt.convection_rhs_abs_max,
        "convection_actual_applied_ledger_qualified":False,
        "accepted_joint_moving_stock_steps":audit.accepted_joint_steps,
        "physical_moving_pressure_qualified":False,
        "order_qualified":False,
        "equal_error_speed_qualified":False,
        "original_state_six_fields_unchanged":all(np.asarray(a).tobytes()==np.asarray(b).tobytes()
                                                 for a,b in zip(audit.state,saved_fields,strict=True)),
        "scope":"Original native diagnostic step and reference fixed-mass pressure identity only."
    }
    if not (result["backend"]=="cpu" and receipt.original_valid
            and result["original_state_six_fields_unchanged"]):
        raise RuntimeError("Original interface witness did not pass its frozen checks")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({key:value for key,value in result.items()
                      if key not in {"source_sha256","input_field_sha256","resolved_parameter_manifest","source_byte_metadata"}},sort_keys=True))


if __name__ == "__main__":
    main()
