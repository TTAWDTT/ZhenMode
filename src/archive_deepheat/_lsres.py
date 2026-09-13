import glob, os
for p in sorted(glob.glob("results/*_3d")):
    print(p, len(os.listdir(p)))
print("---")
for p in sorted(glob.glob("results/*")):
    if os.path.isdir(p):
        print(p, len(os.listdir(p)))
