from pathlib import Path
p = Path('research/research-state.yaml')
s = p.read_text(encoding='utf-8')
s = s.replace('outer_loop:\n  cycle: 0\n  last_direction: "bootstrap"\n  last_reflection: "Starting broad survey across major production OGCMs."', 'outer_loop:\n  cycle: 1\n  last_direction: "literature survey"\n  last_reflection: "Mature OGCMs converge on conservative transport, disciplined masks, generalized vertical coordinates, modular closures, and first-class diagnostics. The next most transferable experiment is a full 3D flux-corrected-transport tracer option with a monotone low-order fallback."')
s = s.replace('    - "Need direct source-code/doc evidence from MOM6, NEMO, MITgcm, ROMS, POP, ICON-Ocean, HYCOM, FVCOM, etc."\n    - "Need comparative metrics on parallel scaling, GPU support, and operational use."', '    - "Need quantitative side-by-side benchmark numbers on the same hardware." \n    - "Need a focused prototype to test the full 3D FCT transport idea in ocean_solver."')
p.write_text(s, encoding='utf-8')
print(s[:1200])
