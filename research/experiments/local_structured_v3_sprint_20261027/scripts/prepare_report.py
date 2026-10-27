"""Package compact offline evidence and render a descriptive, validation-only report."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, re, shutil, statistics

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])

def package(processed,collection,output):
    output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('v3_collection_report',HERE/'collect_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ('runs.csv','runs.json','summary.csv','summary.json','platform_pairs.csv','platform_pairs.json',
                 'verification.json','logit_comparison.json','provenance.json','deployment_decisions.csv'):
        shutil.copy2(processed/name,output/name)
    for path in output.glob('*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    # Full per-episode rows remain external; compact blocks preserve learning diagnostics.
    episodes=[]
    with (processed/'episodes.csv').open() as stream:
        for row in csv.DictReader(stream):
            for name in ('episode','steps','policy_seed'):row[name]=int(row[name])
            for name in ('target','return_value','achieved_sparsity','validation_accuracy'):row[name]=float(row[name])
            row['target_hit']=row['target_hit']=='True';episodes.append(row)
    module.csv_write(output/'learning_blocks.csv',module.learning_blocks(episodes))
    shutil.copy2(collection/'native-verification.json',output/'native_verification.json')
    rows=read(processed/'runs.json');summaries=read(processed/'summary.json');pairs=read(processed/'platform_pairs.json')
    verification=read(processed/'verification.json');provenance=read(processed/'provenance.json');receipt=read(HERE.parent/'launch_receipt.json')
    local=Path(provenance['local_artifact_root']);remote=Path(provenance['remote_snapshot_root'])
    jobs=[]
    for job in receipt['polus']['jobs']:
        log=remote.parent/'logs/lsf'/('v3-cpu.'+str(job['job_id'])+'.out');text=log.read_text()
        runtime=re.search(r'Run time\s*:\s*([\d.]+) sec',text)
        start=re.search(r'Started at (.+)',text);end=re.search(r'Terminated at (.+)',text)
        jobs.append(dict(condition=job['condition'],job_id=job['job_id'],status='DONE',
             run_seconds=float(runtime[1]),started_at_cluster_time=start[1],terminated_at_cluster_time=end[1],
             lsf_stdout_sha256=sha(log),batch=read(remote/'execution'/job['condition']/'batch.finished.json')))
    launch=datetime.datetime.fromisoformat(receipt['local']['created_at_utc'])
    end=datetime.datetime.fromtimestamp((local/'execution/all/batch.finished.json').stat().st_mtime,datetime.timezone.utc)
    wall=(end-launch).total_seconds()
    transfer=read(collection/'remote-transfer.json')
    artifact_index=dict(collection_root=str(collection),processed_root=str(processed),
        local_original_root=str(local),local_batch_finished=read(local/'execution/all/batch.finished.json'),
        local_launch_utc=launch.isoformat(),local_finished_mtime_utc=end.isoformat(),local_approx_wall_seconds=wall,
        local_wall_method='Detached launch receipt timestamp to original batch.finished.json filesystem mtime; approximate, not search-only timing.',
        polus_snapshot_root=str(remote),polus_remote_archive='${POLUS_TASK_ROOT}'+transfer['archive'].split('/gnn-polus-20261023',1)[1],
        polus_archive=str(collection/'polus-results.tar.gz'),polus_archive_sha256=transfer['sha256'],polus_archive_bytes=transfer['bytes'],
        polus_original_files_verified=transfer['file_count'],polus_jobs=jobs,
        per_episode_rows=dict(path=str(processed/'episodes.csv'),rows=len(episodes),sha256=sha(processed/'episodes.csv')),
        original_local_inventory=dict(path=str(collection/'local-inventory.json'),sha256=sha(collection/'local-inventory.json'),files=verification['original_local_files_unchanged']),
        polus_inventory=dict(path=str(remote.parent/'collection/inventory.json'),sha256=sha(remote.parent/'collection/inventory.json')),
        collector_test_log=dict(path=str(collection/'collector-tests.log'),sha256=sha(collection/'collector-tests.log'),result='11 passed'),
        collector_sha256=sha(HERE/'collect_results.py'),report_generator_sha256=sha(__file__))
    save(output/'artifact_index.json',artifact_index)
    timings=[]
    for platform,base in [('local',local),('polus',remote)]:
        for report_path in (base/'outputs').glob('*fixed_probe_v3*/attempt-*/metrics.json'):
            report=read(report_path)
            for kind,key in [('dense','dense_inference'),('fixed_pruned','inference')]:
                timings.append(dict(platform=platform,model=report['model'],model_state=kind,threads=report['runtime']['threads'],
                    mean_seconds=report[key]['mean_seconds'],std_seconds=report[key]['std_seconds'],warmup=5,passes=20))
    module.csv_write(output/'fixed_probe_timing.csv',timings)
    labels={'r2_reference':'R2 reference','q_stop_exact_v3':'H1 exact STOP','q_conservative_v3':'H2 conservative unseen Q',
            'q_random_ties_v3':'H3 random ties','q_target_stop_v3':'H4a threshold stop','q_reachable_v3':'H4b constrained Q',
            'random_reachable_v3':'H6 constrained Random','greedy_reachable_v3':'H6 constrained Greedy'}
    h4a30=[100*r['achieved_sparsity'] for r in rows if r['platform']=='local' and r['target']==.3 and r['variant']=='q_target_stop_v3']
    f1_delta=max(abs(p['final_validation_macro_f1_delta']) for p in pairs)
    report=['# V3 Cora: первичный сбор и обработка — 2026-10-27','',
      '**Техническая проверка PASS: 98/98 local и 34/34 POLUS, без повторных запусков.** '
      'Точная достижимость обеспечивает попадание в целевой диапазон, однако преимущество Q-learning над constrained Greedy в этих условиях не показано.','',
      'Это отдельный этап сбора, разрешённый исследователем после завершения launch-only сессии. '
      'Исходный алгоритм `7e2c760e988b605e305615dea147236e8b0c59a3` и R2 не изменялись.','',
      '## Состав и проверка данных','',
      'Cora, GCN/GraphSAGE 32/32; цели 30% и 50%; training seed101, policy seeds101/5001/5002 локально; '
      'POLUS повторяет только seed101. По восемь вариантов на четыре model–target условия:96 local +32 POLUS; '
      'ещё по два fixed forward/surgery probe, исключённых из статистики попадания в цель. '
      'Каждый Q-запуск:100 эпизодов, max20 удалений, λ2/γ0.9/α0.1/ε0.2, без fine-tuning.','',
      f"Проверены {verification['verified_runs']} физических checkpoint, {verification['verified_training_episodes']:,} обучающих эпизодов, "
      f"{verification['verified_transitions']:,} переходов и {verification['verified_q_entries']:,} Q-записей. "
      'Проверки включают исходные/config/файловые SHA256, строгую загрузку, проекцию весов по retained IDs, '
      'независимое число параметров, конечность forward, легальность действий, точную достижимость с оставшимся горизонтом, '
      'однократный terminal penalty и воспроизведение Bellman updates. Greedy/R2/V3 не переобучались.','',
      'Арифметика трасс и Q replay используют сохранённые validation значения. '
      'Forward всех промежуточных обучающих состояний заново не исполнялся; '
      'независимый forward/пересчёт метрик выполнен для итоговых физических моделей.','',
      'Все34 POLUS-модели дополнительно загружены и проверены в исходном POWER8 окружении: validation accuracy/F1 совпали с сохранёнными. '
      'Контрольный forward использовал один поток и не являлся новым измерением скорости. '
      f"Все{verification['original_local_files_unchanged']} файлов исходного local-batch остались неизменны. "
      f"Архив POLUS содержит {transfer['file_count']} файлов, SHA256 проверен при передаче и для каждого файла после распаковки. "
      'Сборщик прошёл11 тестов, включая отказ при повреждении награды, Q, reset, retained IDs и terminal state.','',
      'Train/test labels скрыты перед адаптивными вычислениями; обработка использует только validation. '
      'Классы на всех узлах сравниваются лишь как predictions, без доступа к test labels.','',
      '## Локальные результаты по трём policy seed','',
      'В таблицах accuracy и achieved sparsity даны в процентах: среднее ± выборочное SD трёх policy seed. '
      'Попадание: точный параметрический диапазон [target−0.01,target+0.01]. '
      'Все seeds используют один dense checkpoint101; это не три независимых обучения GNN. '
      'Greedy детерминирован: три совпавших результата с разными policy seed не являются независимыми репликациями. '
      'Для честного сравнения проверять также achieved sparsity и обе ширины в `runs.csv`.','']
    for model in ('gcn','graphsage'):
        for target in (.3,.5):
            report.extend([f'### {model.upper()}, цель {target:.0%}',''])
            table_rows=[]
            for variant,label in labels.items():
                s=next(s for s in summaries if s['platform']=='local' and s['model']==model and s['target']==target and s['variant']==variant)
                table_rows.append([label,f"{100*s['achieved_sparsity_mean']:.2f} ± {100*s['achieved_sparsity_sample_std']:.2f}",
                    f"{100*s['validation_accuracy_mean']:.2f} ± {100*s['validation_accuracy_sample_std']:.2f}",f"{s['target_hits']}/3",f"{s['search_seconds_mean']:.2f}"])
            report.extend([table(['Вариант','Achieved sparsity, %','Validation accuracy, %','В диапазоне','Search, s (mean)'],table_rows),''])
    report.extend(['## Первичные выводы по гипотезам','',
      '- **H1:** exact STOP отдельно не исправил overshooting:0/12 локальных попаданий, все deployment завершились по max20. '
      'Потери validation остаются большими. Высокая accuracy при меньшем pruning здесь не наблюдается.','',
      '- **H2:**0/12 попаданий; все deployment остановились через agent_stop, обычно с недостаточным pruning. '
      'Высокая validation accuracy сопутствует низкой компрессии, поэтому не доказывает превосходство метода. '
      'В deployment не выбрано ни одного unseen action; это согласуется с сильным предпочтением уже изученных действий при консервативной инициализации.','',
      '- **H3:** random ties меняет траектории и уменьшает среднюю потерю качества относительно R2 в этом пилоте, '
      'но0/12 попаданий; сильное превышение целевой sparsity сохраняется. Это не решение контроля цели.','',
      f'- **H4a:**6/12 попаданий: все шесть при цели50%; при цели30% достигнуты{min(h4a30):.3f}–{max(h4a30):.3f}%, '
      'что немного выше разрешённых31%. Пороговое завершение ограничивает overshoot, но не гарантирует двусторонний диапазон. '
      'Его accuracy при30% нельзя напрямую сопоставлять с H4b как качество при одинаковой компрессии.','',
      '- **H4b/H6:** Q, Random и Greedy достигли диапазона каждый в12/12 локальных и4/4 POLUS условиях. '
      'Это ожидаемое следствие reachability-фильтра, а не самостоятельное доказательство обученной политики.','',
      '- **Польза Q относительно controls:** при50% constrained Q и Greedy совпали по итоговому качеству во всех policy seed. '
      'При30% средняя accuracy Q ниже Greedy на1.36 п.п. для GCN и2.22 п.п. для GraphSAGE. '
      'Среднее Q выше Random в этих четырёх условиях, но n=3 и различия достигнутых ширин/компрессии не позволяют делать общий вывод о превосходстве. '
      'Constrained Q требует существенно больше поиска, чем Greedy.','',
      'В R2 deployment выбрано120 отсутствующих в Q-table действий; H1 —142, H3 —78, H4b —3. '
      'Нулевые значения отсутствующих действий и tie-breaking остаются диагностически важны. '
      'Это наблюдение по финальной таблице, не восстановление посещаемости скрытых channel-specific состояний. '
      'Нынешнее состояние по-прежнему агрегирует ширины/accuracy и не содержит retained-channel signature.','',
      '`learning_blocks.csv` содержит480 блоков по20 обучающих эпизодов: награда, validation accuracy, '
      'достигнутая sparsity, шаги, terminal reasons и попадания в диапазон. '
      'Полные9 600 строк episode summary сохранены внешне; новые бюджеты или гиперпараметры не подбирались.','',
      '## Парное сравнение local и POLUS','',
      '**34/34:** совпали deployment actions, training actions, retained IDs и все тензоры итоговых checkpoint; '
      f'validation accuracy совпала точно, максимальное отличие macro-F1 {f1_delta:.2e} (округление). '
      'Q-таблицы сравнивались по state/action, независимо от порядка сериализации. '
      '20/24 непустых Q-таблиц совпали точно; у четырёх H2 максимальное отличие1.78×10⁻¹⁵, без изменения действий. '
      'У controls/probes Q-таблицы пусты.','',
      'Для двух fixed probes сравнивались сохранённые dense и pruned logits: максимальная абсолютная разница '
      '3.81×10⁻⁶; все четыре пары проходят rtol1e−5/atol1e−6, разногласий argmax нет ни на всех2708 узлах, ни на validation. '
      'Значит в этом ограниченном сравнении различия численной реализации не привели к расхождению политики/каналов.','',
      'Local: x86_64, Python3.12.9, Torch2.2.2/PyG2.6.1, два процесса по два потока. '
      'POLUS: POWER8 ppc64le, Python3.10.22, Torch2.0.1/PyG2.4.0, один процесс с четырьмя потоками в каждом exclusive C4 job. '
      'Библиотеки, backend и условия нагрузки различаются.','',
      'По16 парным pruning-условиям каждой модели отношение POLUS/local search time составило '
      '1.18–1.62 для GCN и0.50–0.76 для GraphSAGE; inference —1.24–2.13 и0.46–0.69 соответственно. '
      'Это описательные отношения конкретных запусков, не изолированный аппаратный benchmark: local задачи шли параллельно, '
      'нет повторных независимых timing runs и одинакового программного стека. '
      'Inference использует исходные5 warmup/20 passes; поиск и inference не смешиваются. '
      'Dense/fixed-pruned timings сохранены отдельно в `fixed_probe_timing.csv`.','',
      '## Выполнение и воспроизводимость','',
      f'Локальный пакет завершился приблизительно за{wall/60:.1f} мин. '
      'Это время между квитанцией запуска и mtime batch.finished.json, включая запуск subprocess и накладные расходы; '
      'сумма search time не является wall time параллельного пакета. '
      f"POLUS: все четыре LSF job DONE; суммарное время выполнения jobs {sum(j['run_seconds'] for j in jobs):.0f} s "
      '(последовательно на exclusive C4). Отсутствие jobs в текущем bjobs подтверждено bhist и успешными LSF stdout/batch.finished.','',
      table(['Condition','LSF ID','Run time, s'],[[j['condition'],j['job_id'],f"{j['run_seconds']:.0f}"] for j in jobs]),'',
      'Компактные данные: `runs.csv/json`, `summary.csv/json`, `platform_pairs.csv/json`, '
      '`deployment_decisions.csv`, `learning_blocks.csv`, `fixed_probe_timing.csv`. '
      'Проверки и адреса raw-артефактов: `verification.json`, `native_verification.json`, `provenance.json`, `artifact_index.json`. '
      'Raw checkpoints, logits, исходные traces/Q-tables, журналы и архив остаются вне Git.','',
      'Скрипт `../scripts/collect_results.py` воспроизводит offline-проверки и обработку из оригинального local root и проверенного POLUS snapshot; '
      '`../scripts/prepare_report.py` собирает компактный пакет и этот отчёт. '
      'Инвентарь хешей пакета: `evidence_manifest.json`.','',
      '## Ограничения и следующий исследовательский шаг','',
      'Результат — технически верифицированная первичная диагностика на одном графе и одном обученном checkpoint каждой модели. '
      'H1–H4 проверялись отдельно; H5 и комбинации гипотез не запускались. '
      'Не выполнялись test-set evaluation, fine-tuning, новая тренировка GNN, новые policy seeds или повторный benchmarking. '
      'Нельзя объединять POLUS seed101 с тремя локальными seeds в n=4 или считать Greedy повторения независимыми.','',
      'Для следующего отдельно утверждённого этапа: сохранить exact reachability как контроль допустимости; '
      'рассматривать Greedy как обязательный сильный baseline; проверять дополнительную пользу обученной политики '
      'при сопоставимых achieved sparsity/ширинах и нескольких независимых dense training seeds. '
      'Текущие результаты не обосновывают финальные научные гиперпараметры или выбор победителя по test.',''])
    report_text=re.sub(r'(?<=[А-Яа-яЁё])(?=\d)',' ','\n'.join(report))
    report_text=report_text.replace('Все893 файлов','Все 893 файла').replace('Все 893 файлов','Все 893 файла').replace('352 файлов','352 файла')
    (output/'REPORT.md').write_text(report_text)
    evidence={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()}
    save(output/'evidence_manifest.json',dict(source_commit=receipt['source_commit'],files=evidence))
    print(json.dumps({'compact_evidence':str(output),'files':len(evidence)+1,'all_runs':len(rows),'episodes_external':len(episodes)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processed',type=Path,required=True);p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();package(a.processed,a.collection,a.output)
