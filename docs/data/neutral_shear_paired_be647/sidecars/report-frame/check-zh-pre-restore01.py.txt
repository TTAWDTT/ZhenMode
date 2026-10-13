"""Records-only transcription/link/MathML check. Never imports scientific code."""
from pathlib import Path
from decimal import Decimal
from html.parser import HTMLParser
import hashlib, json, re

stage = Path('D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/neutral-shear-paired-be647-attempt02')
base = stage.parent
review = stage / 'review/report-frame'
docs = Path('D:/CodexWorktrees/thermal-full-scan/ocean-solver/docs')
report = review / 'zh-candidate-pre-restore01.md'
text = report.read_text(encoding='utf-8-sig')
lines = text.splitlines()
cases = ['00-main', '01-main', '02-main', '03-zero-spice', '04-zero-shear-null']
checks = []
sources = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    p = Path(p)
    sources[str(p)] = sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'), parse_float=Decimal, parse_int=Decimal)

science = [read(stage / c / 'science-verdict01.json') for c in cases]
old = [read(base / 'independent-neutral-shear-native' / c / 'science-verdict01.json') for c in cases[:3]]
journals = []
for i in range(5):
    p = stage / f'review/science{i:02d}-journal01.jsonl'
    sources[str(p)] = sha(p)
    rows = [json.loads(s, parse_float=Decimal, parse_int=Decimal) for s in p.read_text().splitlines()]
    journals.append({r['name']:r for r in rows})

def table(header):
    start = next(i for i,l in enumerate(lines) if l.startswith(header)) + 2
    out = []
    for i in range(start,len(lines)):
        if not lines[i].startswith('|'): break
        out.append((i+1,[c.strip() for c in lines[i].split('|')[1:-1]]))
    return out

def eq(value, expected, locator, label):
    shown = str(value).strip('` ')
    ok = Decimal(shown) == expected
    checks.append(dict(locator=locator,label=label,printed=shown,original_json_value=str(expected),match=ok))

def metric_table(header, fields, records):
    for i,(line,cells) in enumerate(table(header)):
        for col,key in enumerate(fields,1):
            eq(cells[col],records[i]['metrics'][key],f'line:{line}',key)

metric_table('| 案例 | 网格 | T RMS', ['T_RMS','S_RMS'], []) if False else None
for i,(line,cells) in enumerate(table('| 案例 | 网格 | T RMS')):
    for col,key in [(2,'T_RMS'),(3,'S_RMS')]: eq(cells[col],science[i]['metrics'][key],f'line:{line}',key)
metric_table('| 案例 | u RMS', ['u_RMS','v_RMS','eta_RMS','rho_RMS'],science)
metric_table('| 案例 | normalized u RMS', ['u_normalized_RMS','v_normalized_RMS','eta_normalized_RMS','rho_normalized_RMS'],science)
metric_table('| 主案例 | normalized T L1', ['T_normalized_L1','S_normalized_L1'],science[:3])
summary = read(stage / 'review/three-grid-verdict01.json')
for (line,cells),row in zip(table('| 原汇总项目 | 原记录比值'),summary['observations']):
    eq(cells[1],row['ratio'],f'line:{line}','saved_summary_ratio')
for i,(line,cells) in enumerate(table('| 案例 | 体积相对漂移')):
    for col,name in enumerate(['volume','T','S'],1):
        eq(cells[col],journals[i][f'original_diagnostic_inventory_relative:{name}']['value'],f'line:{line}',name+'_drift')
for (line,cells),name in zip(table('| 字段 | 最大绝对变化'),['eta','u','v','T','S','h']):
    eq(cells[1],journals[4][f'original_zero_shear_null:{name}']['value'],f'line:{line}',name+'_null_max')
for i,(line,cells) in enumerate(table('| 案例 | 最小 h')):
    for col,name in enumerate(['positive_h','temperature_floor','salinity_floor','unrepresented_upper_cap_transport_m2_s'],1):
        eq(cells[col],journals[i][name]['value'],f'line:{line}',name)
for i,(line,cells) in enumerate(table('| 网格 | 旧 centered_second T RMS')):
    for col,records,key in [(1,old,'T_RMS'),(2,science,'T_RMS'),(3,old,'S_RMS'),(4,science,'S_RMS')]:
        eq(cells[col],records[i]['metrics'][key],f'line:{line}',key)
metric_table('| 历史 centered_second 网格', ['u_RMS','v_RMS','eta_RMS','rho_RMS'],old)
hist_line = next(l for l in lines if l.startswith('历史 normalized T/S L1'))
hist_tokens = re.findall(r'`([^`]+)`',hist_line)
for i,token in enumerate(hist_tokens):
    for value,key in zip(token.split(' / '),['T_normalized_L1','S_normalized_L1']):
        eq(value,old[i]['metrics'][key],'historical_L1_prose',key)
for i,(line,cells) in enumerate(table('| 案例 | 初始化墙钟')):
    init = read(stage / f'initialize{i:02d}-attempt01-resources.json')
    timing = read(stage / cases[i] / 'timing-receipt.json')
    native = read(stage / f'native{i:02d}-attempt01-resources.json')
    eq(cells[1],init['resources']['elapsed_wall_seconds'],f'line:{line}','init_wall')
    # Timing parent accounting is in the top-level launch resource record.
    timing_resource = read(stage / f'timing{i:02d}-attempt01-resources.json')
    eq(cells[2],timing_resource['resources']['elapsed_wall_seconds'],f'line:{line}','timing_wall')
    eq(cells[3],native['resources']['elapsed_wall_seconds'],f'line:{line}','native_wall')
    eq(cells[4],native['resources']['peak_host_rss_bytes'],f'line:{line}','native_RSS')
plan_names = ['coarse0-measured-plan01.json'] + [f'case{i}-measured-plan01.json' for i in range(1,5)]
for i,(line,cells) in enumerate(table('| 案例 | 原预测时间')):
    plan = read(stage / plan_names[i])
    timing_resource = read(stage / f'timing{i:02d}-attempt01-resources.json')
    eq(cells[1],plan.get('forecast_s',plan.get('forecast_seconds')),f'line:{line}','forecast_s')
    eq(cells[2],plan.get('forecast_rss_bytes',plan.get('forecast_rss')),f'line:{line}','forecast_RSS')
    eq(cells[3],timing_resource['resources']['peak_host_rss_bytes'],f'line:{line}','timing_RSS')
observer_line = next(l for l in lines if l.startswith('科学 observer 父进程实际时间'))
obs_tokens = re.findall(r'`([^`]+)`',observer_line)
for i in range(5):
    resource = read(stage / f'review/science{i:02d}-attempt01-resources.json')
    eq(obs_tokens[i],resource['resources']['elapsed_wall_seconds'],'observer_resource_prose','science_wall')
    eq(obs_tokens[i+5],resource['resources']['peak_host_rss_bytes'],'observer_resource_prose','science_RSS')
adapter = read(stage / 'review/adapter02-attempt01-resources.json')
eq(obs_tokens[10],adapter['resources']['elapsed_wall_seconds'],'observer_resource_prose','adapter_wall')
eq(obs_tokens[11],adapter['resources']['peak_host_rss_bytes'],'observer_resource_prose','adapter_RSS')
stage_verdict = read(stage / 'review/stage-verdict01.json')
summary_resource = read(stage / 'review/summary01-attempt01-resources.json')
for value,key in zip(re.findall(r'`([^`]+)`',next(l for l in lines if l.startswith('唯一最终汇总 exit'))),['elapsed_wall_seconds','peak_host_rss_bytes']):
    eq(value,summary_resource['resources'][key],'summary_resource_prose',key)
for i in range(5):
    eq('3.750000000000001e-6',journals[i]['upper_cap_T_background_inventory_proxy_error_C_m']['value'],'cap_proxy_prose',f'case{i}_proxy')

links = []
for match in re.finditer(r'\[([^\]]+)\]\((data/neutral_shear_paired_be647/records/[^)]+)\)',text):
    relative = match.group(2)
    public = docs / relative
    original = base / relative.split('/records/',1)[1]
    exists = public.is_file() and original.is_file()
    links.append(dict(label=match.group(1),relative_target=relative,actual_target=str(public),original=str(original),exists=exists,public_sha256=sha(public) if exists else None,original_sha256=sha(original) if exists else None,match=exists and sha(public)==sha(original)))

class RenderText(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.math=[]; self.depth=0; self.annotation=False
    def handle_starttag(self,tag,attrs):
        if tag=='body': self.depth=1
        if tag=='math': self.math.append(dict(attrs=attrs,text=''))
        if tag=='annotation': self.annotation=True
    def handle_endtag(self,tag):
        if tag=='body': self.depth=0
        if tag=='annotation': self.annotation=False
    def handle_data(self,data):
        if self.depth and not self.annotation: self.parts.append(data)
render = stage / 'publication-preparation/report-render03.html'
html = render.read_text(encoding='utf-8-sig')
parser = RenderText(); parser.feed(html)
rendered = ''.join(parser.parts)
(review / 'zh-rendered-text03.txt').write_text(rendered,encoding='utf-8')
maths = re.findall(r'(<math\b.*?</math>)',html,re.S)
result = dict(status='PASS_RECORD_TRANSCRIPTION_LINKS_RENDER_ONLY' if all(c['match'] for c in checks+links) else 'FAIL_RECORD_TRANSCRIPTION_OR_LINKS',report_path=str(report),report_sha256=sha(report),render_path=str(render),render_sha256=sha(render),number_check_count=len(checks),number_mismatches=[c for c in checks if not c['match']],checks=checks,source_json_or_jsonl_sha256=sources,links_count=len(links),links=links,mathml_count=len(maths),display_mathml_count=sum('display="block"' in m for m in maths),mathml_tex_annotations=re.findall(r'<annotation encoding="application/x-tex">(.*?)</annotation>',html,re.S),stage_status=stage_verdict['stage_status'],qualified_indices=[int(x) for x in stage_verdict['qualified_case_indices']],scope='Pre-restore candidate only; no NPZ decoding or scientific execution. Exact decimal comparisons against saved JSON/JSONL; no ratios or metrics recalculated.',npz_decoded=0,physics_metrics_computed=0,model_calls=0,oracle_calls=0,observer_calls=0,restoration_assessed=False)
(review / 'ZH_PRE_RESTORE_NUMBER_LINK_RENDER01.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','report_sha256','number_check_count','number_mismatches','links_count','mathml_count','display_mathml_count']},ensure_ascii=False))
