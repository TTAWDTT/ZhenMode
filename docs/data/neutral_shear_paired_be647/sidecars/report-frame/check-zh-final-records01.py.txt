"""Final record-only binding. No science execution, no NPZ decoding, no restore."""
from pathlib import Path
import hashlib, json, re
from decimal import Decimal

stage = Path('D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/neutral-shear-paired-be647-attempt02')
review = stage / 'review/report-frame'
prep = stage / 'publication-preparation'
doc = Path('D:/CodexWorktrees/thermal-full-scan/ocean-solver/docs/neutral_shear_paired_be647_results_zh.md')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_float=Decimal)
current = doc.read_text(encoding='utf-8-sig')
old = (review / 'zh-candidate-pre-restore01.md').read_text(encoding='utf-8-sig')
old_lines, new_lines = old.splitlines(),current.splitlines()
changed = [i+1 for i,(a,b) in enumerate(zip(old_lines,new_lines)) if a!=b]
assert len(old_lines)==len(new_lines) and changed==[9,242,244], changed
assert old_lines[8].replace('，也不是先把各时刻 RMS 做算术平均','')==new_lines[8]
assert sha(doc)=='7d0f3ee390ec7fc87d0272c62de4231fc5067524d7326cc3012b0b062089b3a9'
original_checks = json.loads((review / 'ZH_PRE_RESTORE_NUMBER_LINK_RENDER01.json').read_text(encoding='utf-8'))
links=[]
matches = list(re.finditer(r'\[([^\]]+)\]\((data/neutral_shear_paired_be647/records/[^)]+)\)',current))
assert [m.group(2) for m in matches]==[l['relative_target'] for l in original_checks['links']]
for match,previous in zip(matches,original_checks['links']):
    target = doc.parent / match.group(2)
    actual = sha(target)
    links.append(dict(relative_target=match.group(2),exists=True,sha256=actual,equals_original_saved_sha=actual==previous['original_sha256']))
assert all(l['equals_original_saved_sha'] for l in links)
restore_paths = [prep / 'INDEPENDENT_RESTORE_REVIEW01.json',prep / 'independent-restore01-resources.json',prep / 'INDEPENDENT_RESTORE_VERDICT01.json']
restore_identity = [dict(path=str(p),sha256=sha(p)) for p in restore_paths]
assert [x['sha256'] for x in restore_identity]==['c67f91869e3fd5cac2fc047b7aa4da857c6563610b12fc3c11d7a4a62c5e2205','de6598c327c7c0a16842c2539c70c2dcad2ddf8a15916a4c8c95c23d0baf8cc8','10a09a0a64849524de545b42c8646264cbf6e95685c3853b2d3dee90404d1ffb']
rest,resource,worker = [read(p) for p in restore_paths]
assert rest['parent_actual_returncode']==resource['returncode']==0
assert rest['parent_stop_reason'] is resource['stop_reason'] is None
assert rest['parent_elapsed_seconds']==resource['elapsed_wall_seconds']==Decimal('68.40982751099999')
assert rest['peak_sampled_child_group_rss_bytes']==resource['peak_sampled_child_group_rss_bytes']==46821376
assert rest['reader_invocations']==worker['reader_invocations']==1
assert rest['selected_original_files']==worker['files']==841
assert rest['part_count']==worker['parts']==33
assert rest['all_selected_originals_before_and_after_equal_restored'] is True
assert rest['coverage']==worker['coverage']
assert rest['coverage']['installed_python']==133
assert rest['coverage']['native_case_groups']==5
assert rest['coverage']['output_archives']==205
assert rest['coverage']['original_bridge_artifacts']==23
assert rest['coverage']['failed_predecessor'] is True
packet_dir = doc.parent / 'data/neutral_shear_paired_be647'
manifest_path = packet_dir / 'neutral_shear_paired_be647_attempt02_manifest.json'
manifest = read(manifest_path)
assert sha(manifest_path)==rest['manifest_sha256']
assert manifest['selection_sha256']==rest['selection_sha256']
assert manifest['archive_sha256']==rest['archive_sha256']
assert all((packet_dir / p['name']).is_file() for p in manifest['parts'])
assert sha(packet_dir / 'restore_original01.py.txt')==rest['reader_sha256']
public_packet = dict(manifest_sha256=sha(manifest_path),selection_sha256=sha(packet_dir/'selection01.json'),reader_sha256=sha(packet_dir/'restore_original01.py.txt'),part_files_present=len(manifest['parts']),part_bytes_not_hashed_by_this_review=True)
assert public_packet['selection_sha256']==rest['selection_sha256']
render = prep / 'report-render04.html'
html=render.read_text(encoding='utf-8-sig')
assert sha(render)=='1a8ca635567fdb59122d214e02f82e393aa351b8dc0619c52db489745e08ba0f'
maths=re.findall(r'<math\b.*?</math>',html,re.S)
assert len(maths)==6 and sum('display="block"' in m for m in maths)==4
assert '也不是先把各时刻 RMS 做算术平均' not in html
assert '68.40982751099999' in html and '46821376' in html
assert '恢复终态仍待' not in html
result=dict(status='PASS_FINAL_EXACT_REPORT_FRAME_RECORD_REVIEW',reviewer='Lady / neutral_paired_report_frame',fresh_non_author_context=True,report_path=str(doc),report_sha256=sha(doc),render_path=str(render),render_sha256=sha(render),prose_passes_before_final=2,final_full_cold_read=True,final_abstract_cold_read=True,audience_groups=6,audience_axis_rows=12,unresolved_findings=0,prior_findings=[dict(id=f'english_F{i}',severity='MEDIUM',status='resolved_in_final_Chinese') for i in range(1,5)]+[dict(id=f'Chinese_F{i}',severity='LOW',status='root_fixed_reviewer_verified') for i in range(5,8)],inherited_checked_number_count=185,inherited_number_check_report_sha256=original_checks['report_sha256'],inherited_number_check_json_sha256=sha(review/'ZH_PRE_RESTORE_NUMBER_LINK_RENDER01.json'),scientific_number_body_unchanged_since_that_check=True,changed_lines_since_pre_restore=changed,final_links=links,final_links_count=len(links),restore_source_records=restore_identity,restore_scope=rest['scope'],public_packet=public_packet,mathml_count=len(maths),display_mathml_count=4,npz_decoded=0,model_calls=0,oracle_calls=0,observer_calls=0,physics_metrics_computed=0,ratios_recomputed=0,forecast_recomputed=False,source_ci_qualification_repeated=False,archive_restore_repeated=False,author_report_modified_by_reviewer=False,original_scientific_barriers_modified=False,scope='Exact report and actual MathML render, record transcription, links, frame/prose and saved independent-byte-restore claims only. No general PDE/order/holdout/moving-surface/ranking qualification. Verdict expires for changes to bound report or render bytes.')
(review/'FRAME_REVIEW_ZH_FINAL01.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','report_sha256','render_sha256','unresolved_findings','inherited_checked_number_count','final_links_count','mathml_count','display_mathml_count']},ensure_ascii=False))
