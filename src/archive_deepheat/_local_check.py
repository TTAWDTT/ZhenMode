import sys, os, glob
print("python", sys.version.split()[0])
try:
    import jax, jaxlib
    print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
    print("devices", jax.devices())
except Exception as e:
    print("NO JAX:", e)
print("cwd", os.getcwd())
for d in ("results", "data", "logs", "tests"):
    p = os.path.join(os.getcwd(), d)
    if os.path.isdir(p):
        fs = sorted(os.listdir(p))
        print(f"{d}/: {len(fs)} entries -> {fs[:12]}")
    else:
        print(f"{d}/: MISSING")
print("npz in cwd:", sorted(glob.glob("*.npz"))[:10])
