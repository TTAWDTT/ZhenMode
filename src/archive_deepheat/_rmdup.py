import os
p = r"C:\Users\zhen.luo\.claude\projects\C--Users-zhen-luo-ocean-solver\memory\advection-heat-leak-root-cause.md"
if os.path.exists(p):
    os.remove(p); print("removed")
else:
    print("absent")
