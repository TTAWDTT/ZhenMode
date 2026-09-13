import sys, site
print("exe", sys.executable)
print("ver", sys.version)
print("usersite", site.getusersitepackages())
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
for p in sys.path:
    print("  path", p)
try:
    import jax
    print("JAX OK", jax.__version__, jax.devices())
except Exception as e:
    print("JAX FAIL", type(e).__name__, e)
try:
    import numpy
    print("numpy", numpy.__version__, numpy.__file__)
except Exception as e:
    print("numpy FAIL", e)
