"""Frozen flat-channel wave definitions; v0 history and v1 three-model extension."""

import hashlib
import json

import numpy as np

SCHEMA_V0 = "standing-wave-v0"
SCHEMA = "standing-wave-v1"
MOM6 = "f49a00096df607b48354603e2398e14e189fd62e"
OCEANANIGANS = "1e8587b17171b0bba5bc6728118c3dbbd3c8acf6"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def legacy_contract(case="coarse", half=False):
    nx, dt, error = {
        "coarse": (64, 100.0, 0.10),
        "medium": (128, 50.0, 0.05),
        "fine": (256, 25.0, 0.02),
    }[case]
    if half and case != "medium":
        raise ValueError("half-amplitude control is frozen at medium only")
    return dict(
        schema=SCHEMA_V0,
        case=case,
        nx=nx,
        ny=8,
        nz=4,
        dt=dt,
        period_s=32000.0,
        output_s=1000.0,
        Lx_m=float(32000 * np.sqrt(981.0)),
        Ly_m=100000.0,
        H_m=100.0,
        gravity=9.81,
        rho0=1025.0,
        f=0.0,
        T_C=15.0,
        S_psu=35.0,
        amplitude_m=0.005 if half else 0.01,
        initial_h_m=[100 / 6, 100 / 3, 100 / 3, 100 / 6],
        eos=dict(name="linear", drho_dT=-0.205, drho_dS=0.779, drho_dp=0.0),
        boundary=dict(x="periodic", y="closed", bottom="flat"),
        native_sampling={
            "ocean-solver": {"eta": "node", "u": "node", "layout": "collocated"},
            "MOM6": {"eta": "cell_mean", "u": "node", "layout": "cgrid"},
        },
        ocean_options=dict(
            column_geometry="nodal_dual_v1",
            process_time_scheme="legacy",
            mode_split=False,
            conservative_kv=True,
            localize_conv=True,
            project_adv_vel=False,
            monotone_adv=False,
            fct_adv=False,
            polar_cap_rows=0,
            polar_cap_taper=0,
        ),
        mom_time_options=dict(
            SPLIT=True, SPLIT_RK2B=False, DT=dt, DT_THERM=dt, DT_FORCING=dt, DTBT=dt
        ),
        thresholds=dict(
            error=error,
            phase_rad=0.05,
            amplitude=0.02,
            energy=0.02,
            volume=1e-10,
            tracer=1e-10,
            v_over_U=1e-8,
            restart=1e-10,
        ),
        explicitly_zero=[
            "wind",
            "heat_salt_sources",
            "restoring",
            "bottom_drag",
            "explicit_viscosity",
            "explicit_diffusion",
            "convection_enhancement",
            "GM_Redi",
            "ice",
            "sponge",
            "polar_filter",
        ],
        qualification="engineering_screen_only",
        full_dynamics=True,
        budget=dict(cpu=1, ranks=1, aggregate_peak_bytes=4 * 1024**3, coarse_wall_seconds=600),
    )


def contract(case="coarse", half=False):
    """Extend the problem's model declarations, preserving all v0 metric thresholds."""
    value = legacy_contract(case, half)
    value["schema"] = SCHEMA
    value["native_sampling"]["Oceananigans"] = {"eta": "cell_mean", "u": "node", "layout": "cgrid"}
    value["oceananigans_options"] = {
        "version": "0.113.5",
        "source_commit": OCEANANIGANS,
        "vertical_coordinate": "ZStarCoordinate",
        "timestepper": "SplitRungeKutta3",
        "free_surface": "SplitExplicitFreeSurface",
        "requested_substeps": 8,
        "momentum_advection": "VectorInvariant",
        "tracer_advection": "Centered2",
        "closure": "nothing",
        "coriolis": "nothing",
        "thermodynamic_variables": "T/S anomalies about 15 C/35 psu; physical constant-scalar witnesses",
        "thermal_expansion": -value["eos"]["drho_dT"] / value["rho0"],
        "haline_contraction": value["eos"]["drho_dS"] / value["rho0"],
    }
    return value


def validate_contract(value):
    """Accept exact versioned definitions; v1 also rejects bool/numeric ambiguity."""
    if not isinstance(value, dict):
        raise ValueError("contract must be an object")
    builder = legacy_contract if value.get("schema") == SCHEMA_V0 else contract
    try:
        expected = builder(value.get("case"), value.get("amplitude_m") == 0.005)
    except (KeyError, TypeError) as error:
        raise ValueError("unknown frozen wave case") from error
    if digest(value) != digest(expected):
        raise ValueError("contract differs from frozen definition")
