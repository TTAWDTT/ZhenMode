from pathlib import Path
p=Path('research/research-log.md')
entry='''| 5 | 2026-09-21 | experiment-plan | Committed the FCT-transport protocol under `research/experiments/fct_transport/` before implementation. |
'''
s=p.read_text(encoding='utf-8')
if 'experiment-plan' not in s:
    s=s.rstrip()+'\n'+entry
p.write_text(s,encoding='utf-8')
print(s[-400:])
