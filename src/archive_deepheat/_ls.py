import glob
for p in sorted(glob.glob("_*.py")):
    print(p)
