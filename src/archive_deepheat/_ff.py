import glob
for p in glob.glob("**/*tenyr*", recursive=True):
    print(p)
