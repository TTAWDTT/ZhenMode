"""Declared numerical settings on the existing three physical benchmark problems.

These are development comparisons. Physical inputs, references and screen
thresholds stay fixed; changing a setting creates a distinct contract identity.
"""

from itertools import product

from zhenmode.benchmarks.channel_dynamics import contract as channel_contract
from zhenmode.benchmarks.standing_wave import digest, method_contract

SCHEMA = "native-configuration-scan-v1"
COMBINATION_SCHEMA = "native-configuration-combinations-v1"
SCAN_SCHEMAS = (SCHEMA, COMBINATION_SCHEMA)
PROBLEMS = ("standing-wave", "geostrophic-adjustment", "thermal-wind")
COMMON = ("base", "dt-half", "dt-quarter", "grid-half")
MODEL_SETTINGS = {
    "zhenmode": ("zhen-second-order",),
    "mom6": ("mom-fast4", "mom-fast4-rk2b", "mom-less-damping"),
    "oceananigans": ("oa-sub16", "oa-sub32"),
}
THERMAL_SETTINGS = {
    "zhenmode": ("grid-double", "vertical-double", "zhen-centered-tracer"),
    "mom6": ("grid-double", "vertical-double", "mom-ppm-h3", "mom-ppm", "mom-grid-double-fast4"),
    "oceananigans": ("grid-double", "vertical-double", "oa-centered4", "oa-weno5"),
}


def settings(problem, model):
    if problem not in PROBLEMS or model not in MODEL_SETTINGS:
        raise ValueError("unknown scan problem/model")
    return COMMON + MODEL_SETTINGS[model] + (THERMAL_SETTINGS[model] if problem == "thermal-wind" else ())


def contract(problem, model, setting):
    if setting not in settings(problem, model):
        raise ValueError("setting not declared for this problem/model")
    method = "symmetric-closed-faces" if setting == "zhen-second-order" else "balanced-density-rotation"
    c = (method_contract("fine", method=method) if problem == "standing-wave"
         else channel_contract(problem, "fine" if problem == "geostrophic-adjustment" else "coarse", method))
    c.update(schema=SCHEMA, scan=dict(problem=problem, model=model, setting=setting))
    if setting in ("dt-half", "dt-quarter"):
        c["dt"] /= 2 if setting == "dt-half" else 4
    for key in ("DT", "DTBT", "DT_THERM", "DT_FORCING"):
        c["mom_time_options"][key] = c["dt"]
    if setting == "grid-half":
        if problem == "standing-wave":
            c["nx"] //= 2
        elif problem == "geostrophic-adjustment":
            c["ny"] //= 2
        else:
            c["nx"] //= 2
            c["ny"] //= 2
    if setting in ("grid-double", "mom-grid-double-fast4"):
        c["nx"] *= 2
        c["ny"] *= 2
    if setting == "vertical-double":
        c["nz"] *= 2
        h = [c["H_m"] / (c["nz"] - 1)] * c["nz"]
        h[0] *= .5
        h[-1] *= .5
        c["initial_h_m"] = h
    if setting.startswith("mom-"):
        o = c["mom_time_options"]
        o["DTBT"] = c["dt"] / 4
        o["SPLIT_RK2B"] = setting not in ("mom-fast4", "mom-grid-double-fast4")
        if setting == "mom-less-damping":
            o["BEBT"] = .06  # above the author's approximately 0.05 practical lower bound
        if setting in ("mom-ppm-h3", "mom-ppm"):
            o["TRACER_ADVECTION_SCHEME"] = "PPM:H3" if setting == "mom-ppm-h3" else "PPM"
    if setting.startswith("oa-"):
        o = c["oceananigans_options"]
        o["requested_substeps"] = 32 if setting == "oa-sub32" else 16
        if setting == "oa-centered4":
            o["tracer_advection"] = "Centered4"
        elif setting == "oa-weno5":
            o["tracer_advection"] = "WENO5"
    if setting == "zhen-centered-tracer":
        c["ocean_options"]["tracer_transport_scheme"] = "centered_second"
    return c


def validate_contract(c):
    if isinstance(c, dict) and c.get("schema") == COMBINATION_SCHEMA:
        return validate_combination(c)
    if not isinstance(c, dict) or c.get("schema") != SCHEMA or not isinstance(c.get("scan"), dict):
        raise ValueError("unknown numerical scan protocol")
    s = c["scan"]
    try:
        expected = contract(s["problem"], s["model"], s["setting"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete scan declaration") from error
    if digest(c) != digest(expected):
        raise ValueError("scan differs from its declared configuration")


def rows():
    """Finite declared matrix; no search depending on observed scores."""
    return [dict(problem=p, model=m, setting=s) for p in PROBLEMS
            for m in MODEL_SETTINGS for s in settings(p, m) if s != "mom-grid-double-fast4"]


def followup_rows():
    """Separate adaptive combination; never relabel an original matrix row."""
    return [dict(problem="thermal-wind", model="mom6", setting="mom-grid-double-fast4")]


def combination_rows():
    """Full products of declared axes on the original 32x32x16 thermal case.

    The bounds are a development search, not a claim to cover every native option.
    Existing compatible schemes only; pressure geometry and physics stay fixed.
    """
    result = []
    for dt, rotation, tracer in product(
        (50.0, 25.0, 12.5), ("joint_shear", "split_sources"),
        ("product_fourth", "centered_second"),
    ):
        result.append(dict(model="zhenmode", dt_s=dt, rotation=rotation, tracer=tracer))
    for dt, external_divisor, rk2b, bebt, tracer in product(
        (50.0, 25.0, 12.5), (1, 4), (False, True), (0.1, 0.06),
        ("PLM", "PPM:H3", "PPM"),
    ):
        result.append(dict(model="mom6", dt_s=dt, external_divisor=external_divisor,
                           rk2b=rk2b, bebt=bebt, tracer=tracer))
    for dt, substeps, tracer in product(
        (50.0, 25.0, 12.5), (8, 16, 32), ("Centered2", "Centered4", "WENO5"),
    ):
        result.append(dict(model="oceananigans", dt_s=dt, substeps=substeps, tracer=tracer))
    return [dict(index=i, **r) for i, r in enumerate(result)]


def combination_contract(index):
    declared = combination_rows()
    if type(index) is not int or not 0 <= index < len(declared):
        raise ValueError("unknown declared combination index")
    row = declared[index]
    c = channel_contract("thermal-wind", "coarse", "balanced-density-rotation")
    c.update(schema=COMBINATION_SCHEMA, dt=row["dt_s"],
             scan=dict(problem="thermal-wind", **row))
    for name in ("DT", "DTBT", "DT_THERM", "DT_FORCING"):
        c["mom_time_options"][name] = c["dt"]
    if row["model"] == "zhenmode":
        c["ocean_options"].update(density_rotation_scheme=row["rotation"],
                                   tracer_transport_scheme=row["tracer"])
    elif row["model"] == "mom6":
        c["mom_time_options"].update(
            DTBT=c["dt"] / row["external_divisor"], SPLIT_RK2B=row["rk2b"],
            BEBT=row["bebt"], TRACER_ADVECTION_SCHEME=row["tracer"],
        )
    else:
        c["oceananigans_options"].update(requested_substeps=row["substeps"],
                                        tracer_advection=row["tracer"])
    return c


def validate_combination(c):
    try:
        expected = combination_contract(c["scan"]["index"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete combination declaration") from error
    if digest(c) != digest(expected):
        raise ValueError("combination differs from declared configuration")
