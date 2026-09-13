import os, glob, json
print("=== logs mentioning gm/redi/slope ===")
for f in sorted(glob.glob("logs/*.log")):
    print("  ", f, os.path.getsize(f))
print()
print("=== results dirs ===")
for f in sorted(glob.glob("results/*")):
    if os.path.isdir(f):
        n = len(glob.glob(f + "/*"))
        print(f"  {f} ({n} files)")
print()
print("=== tenyr log? ===")
for f in sorted(glob.glob("logs/*tenyr*")) + sorted(glob.glob("*tenyr*")):
    print("  ", f)
print()
print("=== HEADER of any global_*.log ===")
for f in sorted(glob.glob("logs/global_*.log"))[:6]:
    print("---", f)
    with open(f, errors="replace") as fh:
        for i, ln in enumerate(fh):
            if i > 30: break
            print("   ", ln.rstrip()[:160])
