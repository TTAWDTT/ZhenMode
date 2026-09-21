from pathlib import Path
p=Path('research/literature/README.md')
p.write_text(r'''
# Literature snapshots

This directory stores downloaded docs and selected source-code files used during
the ocean-model survey.

Important:
- These are evidence snapshots, not vendored code.
- The survey synthesis is in `comparative_notes.md`.
- `fetch_docs.py`, `fetch_mom6_docs.py`, `fetch_fesom_docs.py`, etc. are the
  reproducible collection helpers used in this phase.

Survey scope:
- MOM6
- MITgcm
- NEMO
- ROMS
- POP2/CESM
- MPAS-Ocean
- FESOM2
- ICON-Ocean
- HYCOM
- FVCOM
''', encoding='utf-8')
print(p.read_text(encoding='utf-8')[:200])
