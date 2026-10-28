from pathlib import Path
import csv,json,re,hashlib,statistics as st
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
OUT=Path(__file__).resolve().parents[1]
HEAD='34e8ca89599b56ef14bb404527b6a72d6b26d403'
V3='research/experiments/local_structured_v3_sprint_20261027/'
V2='research/experiments/local_structured_v2_q_random_pilot_20261027/'
manifest={r['path']:r for r in csv.DictReader((OUT/'source_manifest.csv').open())}
claims=[]
def claim(cid,location,text,path,selector,computation,limit):
    m=manifest[path]
    claims.append(dict(claim_id=cid,manuscript_location=location,claim=text,source_path=path,commit=HEAD,sha256=m['sha256'],git_blob_sha1=m['git_blob_sha1'],source_selector=selector,recalculation=computation,scope_limit=limit,url=m['url']))
claim('C01','Математическая постановка','P_GCN=Fa+ab+bC+a+b+C; P_SAGE=2(Fa+ab+bC)+a+b+C','src/gnn_pruning/pruning/static_v2.py','parameter_count_for_widths','Analytic integer recount of every runs.csv row','Only supported GCN and mean SAGE branches/biases')
claim('C02','Метод','Exact rational band and remaining-horizon reachability','src/gnn_pruning/pruning/experimental_v3.py','WidthReachability, _deletion_actions, V3Env.legal_actions','Recurrence and invariant checked against code','Parameter feasibility only; initially feasible target required')
claim('C03','Метод','Reward, once-only terminal penalty, legal-only Q bootstrap','src/gnn_pruning/pruning/adaptive_v2.py','LocalV2Env.step, update_q, state','Equation to code review; no research execution','Aggregated state is not an exact retained-ID MDP')
claim('C04','Дизайн','98 local +34 POLUS; 9600 episodes,110830 transitions,75467 Q entries',V3+'results/verification.json','top-level verified_* fields','Read archived PASS report; counts not independently replayed in this task','Distinguish archived model verification from current CSV-only audit')
claim('C05','Дизайн','One frozen training checkpoint101 per architecture; separate policy seeds',V3+'results/provenance.json','input_manifest, policy_seed_interpretation','Compare rows training_seed=101 and policy_seed101/5001/5002','No independent dense-training replicates')
claim('C06','Таблица 1','Q/Random/Greedy constrained each 12/12 local; threshold6/12; other0/12',V3+'results/runs.csv','platform=local; exclude fixed_probe_v3; group variant','Count exact rational |1-P/P0-target|<=1/100; assert target_hit','Controls share filter; success not learned-policy evidence')
claim('C07','Таблица 2 и рисунок 1','Means, sample SD, actual widths/P and search costs',V3+'results/runs.csv','platform=local; variants *_reachable_v3; group model,target,variant','mean, sample stdev (ddof1); per-row widths and P; percentages x100','Policy repeats only; achieved compression not exactly matched')
claim('C08','Результаты','Q-Greedy mean accuracy difference -1.36pp GCN30; -2.22pp SAGE30; equality at50',V3+'results/runs.csv','local q_reachable_v3 vs greedy_reachable_v3, same model/target','100*(mean(Q.validation_accuracy)-mean(Greedy.validation_accuracy))','Descriptive; no p-values or superiority on test')
claim('C09','Геометрия','30%: (23,4..16);50%: (16,7..32), both architectures','src/gnn_pruning/pruning/experimental_v3.py','WidthReachability.goals with F1433,C7,width32,min4,tau.01','Pure enumeration of integer counts from equations; no GNN runs','Architecture feasibility; no optimal accuracy claim')
claim('C10','Таблица 3 и рисунок 2','20 random masks per model/target;80 total and dispersion',V2+'random_trials.csv','group model,target_sparsity; mask_seed1000..1019','mean, sample SD, min/max from validation_accuracy','Static Random masks differ from V3 random actions;one dense seed')
claim('C11','Геометрия','Within-width ranges40.85pp GCN16x8;33.09pp SAGE23x9',V2+'random_trials.csv','gcn,target.5,width16x8; graphsage,target.3,width23x9','100*(max(validation_accuracy)-min(validation_accuracy))','Two/four masks; no causal variance decomposition')
claim('C12','Платформы','34/34 match actions, retained IDs, checkpoint tensors, training actions; equal accuracy',V3+'results/platform_pairs.csv','all34 rows; same_* fields; *_accuracy_delta','Boolean and exact-zero checks;Q-table max differences','Paired execution of identical checkpoints;not new training replicates')
claim('C13','Платформы','Max logits difference3.814697265625e-6; no argmax disagreements',V3+'results/logit_comparison.json','4 saved dense/fixed-pruned pairs','max(max_abs_difference),check all argmax disagreement fields','Predictions compared without test labels;not bitwise logits equality')
claim('C14','Таблица 4','Forward timing means/SD in ms,5 warmups/20passes',V3+'results/fixed_probe_timing.csv','8 rows, platform x model x model_state','seconds*1000;pair dense/pruned within platform/model','Single timing session;different libraries/threads/load;no hardware speedup')
claim('C15','Историческая граница','V1 three datasets/two models/three training seeds excluded','research/experiments/local_cpu_2026/audit/AUDIT_REPORT.md','provenance, surgery capabilities, full matrix','Read audit and separate 18-condition and84-aggregate tables','Never pooled as V2/V3 ablation')
claim('C16','Исторический Greedy','31.0758% and31.0886% exceed31% upper bound','research/experiments/local_structured_v2_cora_matrix_20261027/summary.csv','adaptive_greedy_local_v2,target30,GCN/GraphSAGE','Read actual parameter compression;strict compare to.31','Diagnostic matrix not used for winner selection or test claims')
with (OUT/'claim_traceability.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(claims[0]));w.writeheader();w.writerows(claims)
(OUT/'versions.json').write_text(json.dumps({'checked_date':'2026-10-27','main_head':'72293149d8fff186dd268789e7b8fea5848c9360','pr_number':48,'pr_draft':True,'pr_head':HEAD,'v3_executable_source':'7e2c760e988b605e305615dea147236e8b0c59a3','observed_via':'GitHub connector get_pr_info and branches/main; canonical checkout HEAD; all copied files match git show pinned head','src_diff_v3_to_pr_head':'empty','repository_mutations':False},ensure_ascii=False,indent=2))
refs=json.loads((OUT/'source/references.json').read_text())
with (OUT/'bibliography_verification.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['reference','title','checked_date','primary_url','doi','doi_kind','secondary_primary_verification','status']);w.writeheader()
    for r in refs:w.writerow({'reference':r['n'],'title':r['title'],'checked_date':'2026-10-27','primary_url':r['url'],'doi':r.get('doi',''),'doi_kind':r['doi_kind'],'secondary_primary_verification':r.get('doi_verification',''),'status':'metadata verified against primary publication/arXiv; no DOI invented'})
md=(OUT/'source/manuscript.md').read_text()
order=[]
for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]',md):
    for n in map(int,match.group(1).split(',')):
        if n not in order:order.append(n)
assert order==list(range(1,17)),order
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(OUT/'manuscript_sitito.docx') as z:
    root=etree.fromstring(z.read('word/document.xml'));styles=etree.fromstring(z.read('word/styles.xml'))
    assert not root.xpath('//w:fldSimple|//w:instrText',namespaces=ns)
    assert len(root.xpath('//w:tbl',namespaces=ns))==4
    assert len(root.xpath('//m:oMath',namespaces=ns))==8
    assert all(e.get('{'+ns['w']+'}val')=='Normal' for e in root.xpath('//w:pStyle',namespaces=ns))
    margins=root.xpath('//w:pgMar',namespaces=ns)[0]
    assert all(margins.get('{'+ns['w']+'}'+k)=='1134' for k in ['top','bottom','left','right'])
    footnote=etree.fromstring(z.read('word/footnotes.xml'))
    assert len(footnote.xpath('//w:footnote[@w:id="1"]',namespaces=ns))==1
    for eq in root.xpath('//m:oMath',namespaces=ns):
        assert ''.join(eq.itertext()).strip()
pdf=PdfReader(OUT/'manuscript_reading.pdf')
assert 6<=len(pdf.pages)<=15
text='\n'.join(p.extract_text() for p in pdf.pages)
assert '\ufffd' not in text
assert 'Hamilton W., Ying Z.' in text
qa={'date':'2026-10-27','docx_pdf_pages':len(pdf.pages),'normative_page_range':[6,15],'abstract_ru_words':len(next(s for s in md.splitlines() if s.startswith('@abstract_ru|')).split('|',1)[1].split()),'abstract_en_words':len(next(s for s in md.splitlines() if s.startswith('@abstract_en|')).split('|',1)[1].split()),'tables':4,'figures':2,'editable_equations':8,'first_citation_order':order,'style':'Normal only;TNR10;single spacing;20mm margins;12.5mm first-line body indent;no page numbers','source_files_pinned':len(manifest),'data_to_claim_entries':len(claims),'template_binary_verified':False,'unfilled_author_fields':True,'scientific_execution_during_article_authoring':False,'visual_inspection':'See QA_VISUAL_REVIEW.md for inspected final page images','research_versions':'see versions.json'}
(OUT/'qa_report.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(qa,ensure_ascii=False,indent=2))
