"""Generate an instantaneous native MOM diagnostic request, never run MOM.

Field names are registered by the pinned MOM6 version. Actual timestamps,
staggering and initial-state output still require runtime contract validation.
"""
from pathlib import Path


def write_native_diag_table(path):
    lines = ['"Frozen f=0 standing wave native diagnostics"', '1 1 1 0 0 0',
             '"native",1000,"seconds",1,"seconds","time",']
    for field in ('u', 'v', 'h', 'e', 'temp', 'salt', 'SSH'):
        power = 1 if field == 'h' else 2
        lines.append(f'"ocean_model","{field}","{field}","native",'
                     f'"all",.false.,"none",{power}')
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
