# Stage 1 · аудит baseline структурного ускорения GNN

17 ноября 2026. **GO для сопоставления протоколов с литературой и подготовки следующего ограниченного замера потенциала ускорения.** Выбран существующий GraphSAGE L4/H256 на Flickr, seed 101. Его веса, ранняя остановка и физическое сжатие подтверждены сохранёнными файлами. Сильное утверждение об ускорении исследовательской кампании пока не готово. Нет отдельного систематического dense/pruned latency-замера её финальных моделей, а качество baseline ограничено. [Точная запись baseline и пути](../experiments/flickr_inference_baseline_20261117/BASELINE.md)

Осмотр выполнен от исходного HEAD `9484dce48ed436393d7f0da91a6d529ab79ada25`. Код обучения/модели/общего benchmark сверён с выполненным source `3995c6057f3460005b44bc57d98c6d5e7df7f9e2`, Flickr worker и surgery — с `686cf3f1913c2085260ccfef40c384de8580a042`. Эти файлы совпадают. Архив GPU project с source `32bf91cab509b58d7619797683e502f425a6f6bf` также содержит те же GraphSAGE, Flickr worker и surgery. Подготовлены только документы. Обучение, pruning, inference и LSF-задания в этом этапе не выполнялись.

В таблице `A=${GNN_ARTIFACT_ROOT}` — соседний `prunable_gnn_artifacts`, `D=$A/flickr-dense-collection-20261101T140723Z/raw`, `V=$A/flickr-v5-collection-20261102T094740Z/extracted`, `G=$A/polus-gpu-cpu-benchmark-20261110-173022`. `B=$D/runs/flickr-graphsage-l4-h256-s101`.

| Проверка | Свидетельство / путь | Вывод | Требуемое действие |
|---|---|---|---|
| Baseline и выбор | `$B/{checkpoint.pt,resolved_config.json,report.json}`, [финалисты sweep](../experiments/flickr_dense_sweep_20261031/collection_20261101/FINALIST_RECOMMENDATIONS.json) | Тренированный L4/H256, максимум GraphSAGE validation macro-F1 среди 16 условий, 522 503 параметра. Accuracy 0,540280, macro-F1 0,240890 | Закрепить этот ID для технического исследования. До количественного сравнения согласовать архитектуру и split с выбранной публикацией |
| Достаточность обучения | `$B/history.jsonl`, [DenseTrainer](../../src/gnn_pruning/training/trainer.py) | Проверены 408 конечных записей, минимум loss на 358, затем ровно 50 эпох без улучшения. Best восстановлен. Один seed, глобальная сходимость не доказана | Для следующих шагов использовать best. Проверку устойчивости качества по seed планировать отдельно |
| Метрики и дисбаланс | `$B/report.json`, [metrics.py](../../src/gnn_pruning/evaluation/metrics.py) | Train/val eval метрики есть. Validation содержит все 7 классов. Macro-F1 корректно учитывает все классы. Класс 5 не предсказывается, largest-class accuracy 42,94% | В литературном сравнении сохранять macro-F1 и распределения. Accuracy без них скрывает ограничение качества |
| Вход и split | `$D/inputs/{manifest.json,graph.pt,split.json}` | Хеши сверены. Generated 60/20/20 seed 101, raw features, full-batch transductive, test labels замаскированы согласно исходному протоколу и прежнему аудиту | Не смешивать с официальным split и иным preprocessing. Для сопоставления фиксировать эти различия |
| Физическое сжатие | `$V/runs/flickr-graphsage-l4-h256-s101-*/checkpoint.pt`, [Local Surgery](../../src/gnn_pruning/surgery/local_v2.py) | Все 14 checkpoint проверены как точные физические срезы dense. Формы согласованы с архитектурой и original IDs. Число весов реально меньше | Переиспользовать эти веса и masks. Сравнивать фактические размеры и parameter reduction |
| Основная V5-кампания | [Flickr worker](../../src/gnn_pruning/experiments/flickr_pruning.py), `$V/runs/*/report.json` | `validation_forward_seconds` включает forward, выбор val, argmax и accuracy без отдельного прогрева/повторов одного финального состояния. Поиск смешивает разные модели и cache hits | Для latency финальных моделей нужен отдельный парный замер. Поисковые оценки для этого не агрегировать |
| Парный CPU/P100 | `$G/benchmark.py`, `$G/verified/{benchmark.json,flickr-graphsage-l4-h256-s101.json}` | Eval, no_grad, 4 прогрева, 9 отсчётов, CUDA sync до/после. Graph уже на device. Совпадают CPU/GPU маски и validation-предсказания. Для одной SNIP-маски GPU pruning speedup 1,103× | Пригоден как технический pilot. Повторные независимые измерения и финальные masks нужны для общего вывода |
| Общая утилита latency | [benchmark.py](../../src/gnn_pruning/evaluation/benchmark.py) | Eval/no_grad, заданные прогрев/повторы, mean и population SD. Первоначальный data.to вне таймера, внутри каждого forward повторён same-device data.to | Документировать эту границу. Не объединять с медианами специализированных замеров |
| CPU-потоки и оборудование | [C4 thread-scaling](../experiments/flickr_cpu_thread_scaling_20261101/collection_20261102/REPORT.md), `$G/verified/benchmark.json` | C4, PyTorch 2.0.1 CPU и 1.13.1 CUDA — разные provenance. Mask 50% у thread-scaling отличается от 29% SNIP на P100. CPU affinity и interop тоже отличаются | Раздельные серии. Не выводить CPU/GPU ускорение из смешанных групп |
| Память | `$B/report.json`, `$G/verified/flickr-graphsage-l4-h256-s101.json`, [runtime_memory_metrics](../../src/gnn_pruning/evaluation/benchmark.py) | RSS и CUDA peak относятся к целым процессам/стадиям, CPU estimate не включает промежуточные тензоры. GPU peak 2,39 GiB включает SNIP и копии | Измерить отдельно resident baseline, dense/pruned forward и search, allocated и reserved |
| Воспроизводимость | `$B/completed.json`, V5 inventories, `$G/benchmark.sha256` | Сверены выбранные inventories, dense/split/graph hashes, все 14 tensor projections и арифметика 9 отсчётов | Хранить исходные архивы вне Git и брать точные pins из baseline record |

Пример фактических форм дан для существующего static magnitude target 50%, ширины **[111,230,219]**. `lin_l.weight` и `lin_r.weight` имеют одинаковые формы, bias есть у `lin_l`. Это без fine-tuning, каждый вес точно совпал с соответствующим срезом frozen dense.

| SAGEConv | Dense, обе матрицы | Pruned, обе матрицы | Bias dense → pruned |
|---:|---:|---:|---:|
| 0 | [256,500] | [111,500] | [256] → [111] |
| 1 | [256,256] | [230,111] | [256] → [230] |
| 2 | [256,256] | [219,230] | [256] → [219] |
| 3 | [7,256] | [7,219] | [7] → [7] |

Dense имеет 522 503 параметра, этот снимок — 266 433. Фактическое удаление параметров **49,0083%**, удаление скрытых каналов **27,0833%** (208 из 768). Target 50% относится к параметрам с допуском 0,01. Для magnitude target 30% ширины [164,248,245], 370 958 параметров, reduction 29,0037%, каналов удалено 14,4531%. У GPU SNIP target 30% ширины [174,240,228], 370 801 параметр, reduction 29,0337%, каналов удалено 16,4063%. Совпадение target не делает маски или точные размеры одинаковыми.

Парный benchmark LSF 1617705 измерял полный forward на уже размещённом графе, включая операции свёрток, mean aggregation, ReLU и eval-вызовы dropout. В таймер не входят загрузка checkpoint/графа, device transfer, предварительная сборка графа, val indexing, argmax и вычисление метрик. Внутри остаётся небольшая Python-проверка device выхода. CPU и GPU чередуются в каждом повторе. Все dense-отсчёты получены до серии pruned-отсчётов, поэтому чередования dense/pruned и независимых process-level повторов нет. Сжатая модель для forward на GPU — копия той же CPU-маски. Отдельный таймер SNIP включает gradient scoring, план и физическую surgery, и не является задержкой предсказания.

| Оборудование и стек | Dense median, мс | Pruned median, мс | Dense / pruned | SD отсчётов dense / pruned, мс |
|---|---:|---:|---:|---:|
| C4 CPU, 4 физических ядра, Torch 1.13.1 | 2535,878 | 2562,411 | 0,990× | 256,274 / 117,843 |
| C4 P100, Torch 1.13.1 CUDA 10.2 | 86,061 | 78,025 | 1,103× | 0,299 / 0,097 |

Девять отсчётов на модель не дают девять независимых экспериментов. Разница CPU-медиан мала относительно разброса. Переход CPU → GPU для dense равен 29,47× в этой серии и относится к смене устройства. Наблюдаемое сокращение GPU-latency от структурного сжатия составляет 9,34%. Отдельный CPU thread benchmark на PyTorch 2.0.1 дал при 4 потоках и 50%-маске [111,230,219] median 1,6100 → 1,4080 с, 1,143×, 2 прогрева и 7 повторов. Этот результат остаётся отдельной серией. Multi-GPU-замер 1/2/4/6 P100 оценивает пакет независимых SNIP+forward+validation-запусков с подготовкой отдельно и не характеризует задержку одной модели.

Согласно сохранённым environment, обучение и V5 работали на POWER8 ppc64le, Linux 3.10.0-693.el7/glibc 2.17, CPU PyTorch 2.0.1 с USE_CUDA=0, Python 3.10.22, PyG 2.4.0, NumPy 1.24.4, OpenBLAS. У GPU-серии отдельная сборка **1.13.1+polus.cu102.power8**, CUDA toolkit 10.2.89, NVIDIA 440.33.01, Tesla P100-SXM2-16GB sm_60. В отчёте GPU указан RHEL 7.4. cuDNN, NCCL и distributed выключены. Скрипт ограничивает intraop=4, interop=1 и affinity [0,8,16,24]. Сохранённый pip-freeze не содержит torch-scatter/torch-sparse. Основная V5-серия обычных четырёхпоточных задач записала interop=160, тогда как специальные 12-поточные Q-задачи использовали interop=1 и C4. Это установленное различие конфигураций, а не доказательство реальной oversubscription.

| Объект памяти | Размер / характер свидетельства |
|---|---|
| Dense FP32 веса | 2 090 012 байт, 1,993 MiB, точно из tensor shapes |
| Pruned magnitude 50% FP32 веса | 1 065 732 байта, 1,016 MiB, точно из tensor shapes |
| Входные признаки | 89 250 × 500 × 4 = 178 500 000 байт, 170,231 MiB, оценка полезных данных |
| edge_index int64 | 2 × 899 756 × 8 = 14 396 096 байт, 13,729 MiB. Label tensor и Python-объекты отдельно |
| Выход dense | 89 250 × 7 × 4 = 2 499 000 байт, 2,383 MiB, оценка |
| Пример промежуточного edge-тензора | При материализации [899756,500] FP32 потребует 1 799 512 000 байт, 1,676 GiB. Оценка, выделение такого тензора здесь не профилировалось |
| Dense training peak RSS | 2 902 196 224 байт, 2,703 GiB, измерено за весь worker, включая обучение и проверку |
| GPU technical peak allocated | 2 562 683 392 байта, 2,387 GiB, измерено за несколько стадий benchmark. Отдельного dense/pruned peak и reserved нет |

Input graph и число рёбер после удаления каналов сохраняются. Benchmark одновременно держит CPU/GPU dense, CPU/GPU pruned и дополнительную GPU-копию для одинаковой маски. SNIP использует backward, а результаты предыдущих вызовов частично остаются в области видимости. Поисковые контроллеры также создают отдельные модели при surgery. Поэтому peak процесса не раскладывается на память весов и одиночного forward простым вычитанием.

У общей `runtime_memory_metrics` CPU memory estimate включает только параметры, вход и выход. Тензоры скрытых слоёв и временная память backend в оценку не входят. В CUDA-ветке первый `output` остаётся жив при reset peak и втором forward. Такой peak включает этот дополнительный выход. Эти свойства кода подтверждены. Применение именно этой функции к конкретному Flickr-результату нужно устанавливать по его runner, основной V5 её не вызывает.

**Пригодно для повторного использования** — exact dense checkpoint, конфигурация, generated split и веса 14 физических вариантов L4/H256, их validation-метрики с оговорками об одном seed и отсутствии нового inference. Прежний независимый reload-аудит sweep воспроизвёл accuracy и macro-F1 выбранного baseline с абсолютной разницей 0 и 5,55e−17. В CPU/P100-техническом worker различий validation-предсказаний нет, max logit delta dense 2,38e−6. Это записанное прежними проверками свидетельство, а текущий этап сверил файлы, тензоры и арифметику.

**Требует новых измерений после отдельного согласования** — standalone latency и memory каждого финального mask на одном фиксированном устройстве/стеке, холодный и resident сценарии, стоимость подготовки и распределение времени по операциям. V5 search cost, latency mixed search states, CPU thread-scaling, CUDA technical pilot и multi-GPU throughput остаются самостоятельными группами. Ошибок физического удаления или best-checkpoint selection в выбранной группе не найдено.

**Подтверждённые пробелы** — отсутствует систематический runtime-бенчмарк финальных V5-моделей, изолированная peak-memory оценка и повторения по обучающим seed. **Гипотезы ограниченного ускорения** — затраты агрегации/перемещения данных и первого слоя с неизменными 500 входными признаками, эффективность небольших матриц, kernel launch overhead. Сохранённых operator profiles недостаточно, чтобы установить их доли или причину CPU-разницы. Ограниченные macro-F1 и перекос предсказаний подтверждены. Причины, включая регуляризацию, глубину и preprocessing, этим аудитом не установлены.

Минимальный следующий этап состоит из трёх шагов. Сначала составить таблицу соответствия выбранной публикации по глубине, ширинам, split, preprocessing, full-batch/sampled режиму, fine-tuning и аппаратуре. Затем согласовать технический resident-forward замер dense и уже существующих 30%/50%-масок с фиксированными threads/affinity и чередованием моделей в независимых процессах. Отдельно собрать end-to-end путь, operator profile и memory baseline/forward/search с allocated и reserved. После этого оценить потенциал и окупаемость ускорения, сохраняя validation-качество и не выбирая модели по test.

Контроль Stage 1 прошёл на локальных CPU-тензорах без создания моделей и вызова forward. Проверены хеши выбранных completed inventories, split/graph bytes, best epoch/patience, все 14 физических projections и recorded CPU/P100 timing arithmetic. Локальное подробное свидетельство `$A/flickr-inference-stage1-20261117/audit.json`.
Проверка файлов и тензоров сохранена рядом в `inspect_saved_evidence.py`, хеши обоих файлов — в `inspection.sha256`. Pytest не запускался, поскольку изменения ограничены документацией.
