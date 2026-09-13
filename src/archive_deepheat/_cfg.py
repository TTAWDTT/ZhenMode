import sys
sys.path.insert(0, "src")
import dataclasses
from config import PhysicsConfig
for f in dataclasses.fields(PhysicsConfig):
    print("%-22s = %r" % (f.name, f.default))
