# Independent source/interface review — 2026-10-13

Reviewer: Lady (`/root/scalar_parity_review`). Skills: acceptance-gate, independence-bookkeeping, planted-truth, timing-discipline, tool-stewardship. No production edits, solver rerun, native run, parameter tuning, scientific threshold change, or new qualification score.

## Verdict

Attempt01 is **REJECT for production-owner interpretation**. Attempt02 is **SCOPE-ACCEPT for a bounded DEVELOPMENT mechanism and an explicit opt-in source patch**, subject to preserving the original owner envelope and default limited gradient. Scientific acceptance remains **OPEN**. This is not a stability, full-time, native, rank, holdout, or independently derived physics certificate. The ece momentum option is not upgraded or expanded; the frozen best configuration remains 3c8.

## Owner/unit failure caught and retained

The helper `_horizontal_tracer_fluxes` returns concentration-times-volume-velocity fluxes, C·m/s. In the legacy branch supplied `face_transport` is divided by `p.dz_node` before multiplication by concentration; the channel product-fourth branch also divides supplied transport by `params.dz_node`. `_material_point_scalar_rate` expects capacity-integrated horizontal faces, C·m²/s. The ordinary `_advection_scalar` owner multiplies both helper outputs by nodal thickness before calling it. `integrated_reference_scalar_rate` explicitly documents capacity-integrated numerator inputs, then divides the total once by h.

Attempt01 passed the helper outputs directly. Its shifted horizontal divergence was C/s while the combined mass/vertical numerator terms were C·m/s, and the final division by h did not repair that inconsistency. Its horizontal moment correction also subtracted an unintegrated baseline from an integrated moment. Consequently its reported even errors and residual final odd errors cannot qualify the real owner. The saved attempt01 values remain untouched, along with the root failure receipt.

Attempt02 uses `fx,fy = helper(...)` and `faces=(fx*p.dz_node,fy*p.dz_node)`, exactly matching the production `_advection_scalar` bridge. The diagnostic calls the same `_layer_face_transports` and `_vertical_transport_iface` and constructs the same shifted reference q−q.flat[0]. Source review confirms this repaired units path; it is still a direct instantaneous owner test, not a full timestep test.

## Derivative, sign, and primitive audit

The array index increases downward. `dz_iface=abs(diff(z))>0`, so the physical tracer gradient is ∂q/∂d with d=−z, not ∂q/∂z. The prototype is the correct five-point derivative in d. At left node0 the weights divided by12h are (−25,48,−36,16,−3); at node1 (−3,−10,18,−6,1); interior (1,−8,0,8,−1); right weights are reflected and negated. Difference arithmetic preserves constants exactly and the derivative has units C/m. Exact Fraction moment checks passed degrees0..4 at all five stencil locations (25 checks), including the reflection signs; a wrong-sign derivative is rejected. The producer's polynomial controls report errors <=9.64e−16 C/m at16/32, below its declared1e−14 limit; wrong sign produces ~0.08 C/m above its1e−3 negative-control floor.

`Fz` and `point_w` are downward-positive. Hence `−point_w*gradient_d` equals `−w_up*gradient_z`: both signs reverse together. On the background q=15+gamma z, gradient_d=−gamma, yielding +gamma*point_w. With the prescribed upward w1=−A omega sin(mZ)cos(ky)sin(phase), the background RHS is +gamma A omega sin(mZ)cos(ky)sin(phase), matching the declaration.

For theta1=−gamma A sin(mZ)cos(ky)cos(phase), theta_y=gamma A k sin(mZ)sin(ky)cos(phase) and theta_z=−gamma A m cos(mZ)cos(ky)cos(phase). Substitution into −v1 theta_y−w1 theta_z cancels the sin²/cos² y dependence and gives −gamma A² omega m sin(2mZ)sin(2phase)/4. It is even in A. Therefore the odd wave response minus the odd background response is zero for these pinned continuum fields. No numerical fit or root error-evaluation function is needed for this algebra. Its lineage still shares the prescribed fields and parameters; it does not independently prove those fields describe the full native state. The prior TargetI workflow algebra grade and OPEN physical attribution are retained.

The prototype uses the first interface spacing h everywhere. This is valid only within the uniform-column envelope. It must not become a general nonuniform or partial-column derivative without new evidence.

## Baseline isolation

The diagnostic does not rebind `transport.physical_tracer_gradient` or the vertical module's gradient. It copies the transport module dictionary, replaces the gradient name only in that copied dictionary, and recompiles only the source of `_material_point_scalar_rate`. This is source-function recompilation, not a full AST transformation. The function has exactly two direct gradient call sites: horizontal column moment and point desired RHS. Its baseline `integrated_reference_scalar_rate` is the original function object, whose Python globals remain the original vertical module globals. The `muscl_band_v3` choice, conservative endpoint baseline, primitive/projection, and scalar blend functions remain the original objects and source.

Independent readback verifies all eight per-grid/amplitude/wave `low_rate` cases are bitwise identical between selectors, and all sixteen selector/grid/case alpha values are1 with feasibleTrue. Rates satisfy low_rate+delta numerically. This supports selector isolation only in the recorded cases; it is not a monotonicity or stability theorem for arbitrary states.

## Raw arrays and identities

The standalone attempt02 reviewer in `attempt02-owner-units/review/readback.py` imports NumPy/stdlib only, never the diagnostic or root science evaluator. It rebuilt parity from four raw cases per field, checked saved parity arrays bitwise using the recorded arithmetic association, and separately measured odd/even with independently grouped association. A positive parity plant O=2,E=3 and exact null passed; a sign-corrupted comparator was caught before opening the NPZ. Direct file hashing independently matched all133 installed `.py` files to wheel identity and result before/after maps. Diagnostic/declaration/source-function hashes match their recorded identities. This envelope is frozen installed source3c8e86b, not current worktree HEAD ece37c7. Actual package bytes were checked, not source presence alone.

| nodes | final odd excess: limited → fourth (C/s) | desired odd excess: limited → fourth (C/s) | final even error: limited → fourth (C/s) |
|---|---|---|---|
|16|1.43064530e−13 → 1.85569823e−23|1.28813454e−13 → 1.35733077e−23|4.37146327e−15 → 1.95790967e−15|
|32|6.91855737e−14 → 2.14232185e−23|6.25045339e−14 → 1.54020999e−23|1.62194289e−15 → 1.97979088e−15|

The independent grouping gives final fourth odd1.85409594e−23 and2.14174025e−23 C/s. Its maximum association drift8.27e−25 C/s lies within the declared raw-amplitude bound ~5.25e−23 C/s. Accordingly the fourth odd response is **at floating cancellation scale**, not a precisely certified1e−23 physical residual or a ten-digit physics claim. Limited odd responses around1e−13/1e−14 are unaffected at the reported scale. The symmetric gradient bias drops from4.63588e−7/2.24946e−7 to1.00154e−18/1.42777e−18 C/m. These are stored mechanism metrics, not heldout or precision-deepening certificates.

The first reviewer attempt on attempt01 used fixed parity atol1e−25 across independently associated subtraction and failed. Its original script, declared criterion, timing receipt, and `readback-failure-01.json` are preserved. The failure exposed cancellation sensitivity. Attempt02's new declaration separates bitwise stored-arithmetic reproduction from an epsilon-derived association bound; this revises only a post hoc readback check, never a scientific gate. No attempt01 diagnostic was rerun or relabeled PASS.

Review resource controls: one CPU via affinity,60s alarm,1GiB address-space cap, NumPy threads1. Metadata/fixture pilot0.503835s preceded NPZ; finite readback completed0.703627s, peak RSS41644KiB. No JAX compilation or PDE computation occurred in review.

## Permitted patch scope and remaining gate

Evidence supports adding an explicit optional fourth gradient only for the two material-point direct uses, while leaving the global physical tracer gradient and baseline MUSCL/endpoint/blend algorithms unchanged. Default must remain limited, and no default/profile/best-config switch is justified. Preserve current factory owner constraints: integrated_reference_v1, fully wet uniform physical-endpoint unsplit closed nodal channel, >=6 nodes, nodal_capacity, centered_second horizontal tracer, centered_fourth continuity, and no monotone/fct/project_adv_vel. The five-point primitive's theoretical>=5 support does not license relaxing the existing production>=6 envelope. Reject unknown selectors and unsupported owners rather than silently ignoring an option.

The eventual patch must demonstrate actual owner dispatch and default unchanged behavior, selector-specific low-rate isolation, constant/affine/stencil sign checks, and conservative-budget preservation within existing tolerances. This review does not itself audit code not yet written. These16/32 points and the oracle outputs now influenced the candidate choice and cannot be reused as independent qualification points. No amplitude scaling law A|A| is established by a single |A| pair. New independent/fulltime/stability/qualification work remains OPEN and needs its own predeclared evidence; do not promote this mechanism receipt to scientific closure or loosen existing gates.
