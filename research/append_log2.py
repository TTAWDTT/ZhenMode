from pathlib import Path
p=Path('research/research-log.md')
entry='''| 5 | 2026-09-21 | experiment-plan | Committed the FCT-transport protocol under `research/experiments/fct_transport/` before implementation. |
'''
p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n'+entry,encoding='utf-8')
print(p.read_text(encoding='utf-8')[-300:])
