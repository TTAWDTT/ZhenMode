# Fresh hostile skeptic 1 — independent skeleton

Created UTC: 2026-10-12T21:38:20.3159163Z
TARGET_SPEC SHA256: 53B671E917D83BC0F26F1A260C47407AE1F99BADB3BFC6DF7D3C89613259E9B8
prove-protocol SKILL SHA256: 81A56B342EDB23D9AD10D0F2D4FBA1EF55CE97555DD77207E6E9472BF7F602EA
Read order so far: full prove-protocol SKILL, TARGET_SPEC only. No primary/root/variational/spectral proof read; no author code reused.

## Independent proof spine (written before supporting proof access)
- Tangential: represent each finite second difference as restriction of (shift minus identity)^2 on a zero extension. Shift is an isometry, so norm R <= 4 and 0 <= R^T R <= 16 I. F_t symmetric, spectrum in [0,1], hence contraction. R annihilates constants and arithmetic progressions; solving Rx=0 gives exactly that two-dimensional space. Symmetry then implies sum preservation. Compute equivalence is transpose scatter [d,-2d,d] at each triple, not a reflection claim.
- Normal: for symmetric L, direct form -x^T L x = sum adjacent squared differences + 2 x_0^2 + 2 x_(n-1)^2. Upper bound <=4||x||^2 follows from signed difference representation or from 4I+L having form sum adjacent (x_j+x_(j+1))^2. Thus eigenvalues in [-4,0], L^2 <=16I; F_n in [0,1]. Equality at upper -L edge is alternating vector. This route does not require sine basis as an import.
- Sine basis: recurrence with ghost values x_-1=-x_0 and x_n=-x_(n-1) holds for angles k*pi/n. Eigenvalue 2cos(theta)-2. All k=1..n distinct eigenvalues since angle/2 in (0,pi/2], giving n independent real eigenvectors. k=n is alternating, including exceptional sine normalization. B_n=(I+L/4)^2 multiplier cos^4(theta/2). No universal tangent boundary import.
- Symbolic small wave: 1-f=sin^4(theta/2)=theta^4/16+O(theta^6); 1-b=theta^2/2+O(theta^4), a statement only about angle symbols.

## Hostile exact controls preregistered
Only n=4, exact rational SymPy. No n5/n8 or scans. Run CPU1, wall <=180s, 4GiB sampled RSS; first-run actual timing before further compute. Build R and L independently from index conditions, no author code.
- Reproduce T1 [-1/8,9/8,33/8,71/8] and T2 [5/8,9/8,9/8,5/8]. Attack positive entries, affine/constant/sum, mode and PSD conclusions at smallest legal n.
- Reject wrong 1/4 plants: F_t alternating [0,2,-2,0], squared norm8 versus4, quadratic value -4; F_n alternating=-3x, squared norm36 versus4.
- Check T1 positivity failure separately from TVD. T1 is NOT an assumed TV-increase control. Independent candidate: F_t on e0 gives TV17/16 versus1. For F_n constant vector has output TV1 versus0. Exact max/min checks catch positivity/maximum-principle overclaims.
- Attack arbitrary diagonal weighted contraction via W=diag(1,100,1,1), input e0. Expected F_t weighted growth; F_n growth to be checked, with a separate witness if needed. This is a fixed nonuniform-weight counterexample, not dynamic eta native data.
- Check Fn normal constant non-preservation; no claim of normal affine preservation. Kill possible replacement of original tangent edge filter by even reflection if asserted without definition; no source/production review in this pass.

## Review sections to populate incrementally
1. primary WORKFILE claims, boundary/domain assumptions, identity imports.
2. variational PROOF claims, independent route and quantifiers.
3. spectral WORKFILE claims, sine endpoints, normalization, failed v1 retained and not counted.
4. exact controls, machine-readable bug reports and caught false-copy plants.
5. overclaim nursery and symbol/PDE scope.
6. scoreboard/final: at best CLEAN fresh skeptic pass1, never VERIFIED-CLOSED; still requires new-context consecutive clean pass and nonauthor reconstruction/interface trace.
