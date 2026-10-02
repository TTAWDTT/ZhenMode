"""Canonical subcycles definitions; legacy operations unchanged."""
from ocean_solver.fd.backend import jax


def _subcycle(fn, carry, n, p):
    """Run `n` fixed-operator subcycles of `fn`, returning (carry, mean term).

    `fn(carry) -> (carry, term)` where `term` is that substep's tendency; the
    returned term is the mean over the substeps (None when the caller only
    wants the carry). The operator and any mask stay frozen across the
    subcycles, so the substep dt is what brings each term back inside its
    explicit CFL; `use_scan` fuses the loop into one compiled body and is
    numerically identical to the unrolled Python loop (D10, D12).
    """
    if p.use_scan:
        carry, terms = jax.lax.scan(lambda c, _: fn(c), carry, None, length=n)
        return carry, jax.tree.map(lambda t: t.mean(axis=0), terms)
    total = None
    for _ in range(n):
        carry, term = fn(carry)
        if total is None:
            total = term
        else:
            total = jax.tree.map(lambda a, b: a + b, total, term)
    return carry, jax.tree.map(lambda t: t / n, total)
