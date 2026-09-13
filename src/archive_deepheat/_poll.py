"""Poll _prodAB.out until it grows past the header, or timeout."""
import os, sys, time
P = "_prodAB.out"
BASE = 516
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 560.0
t0 = time.time()
while time.time() - t0 < budget:
    sz = os.path.getsize(P) if os.path.exists(P) else 0
    if sz > BASE:
        print("GREW to %d bytes after %.0f s" % (sz, time.time() - t0))
        break
    time.sleep(10)
else:
    print("still %d bytes after %.0f s" % (os.path.getsize(P), time.time() - t0))
print(open(P, errors="replace").read()[-3000:])
