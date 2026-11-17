# Flickr GraphSAGE · baseline для исследования инференса

Запись аудита 17 ноября 2026. Выбран существующий **flickr-graphsage-l4-h256-s101**, LSF **1611784**, для последующих сопоставлений и измерений. Выбор опирается на validation macro-F1, наличие полного журнала и уже выполненный парный CPU/P100-замер. Среди 16 GraphSAGE-конфигураций sweep его validation macro-F1 максимальна. Выбор одного seed после архитектурного поиска остаётся разведочным. [Выводы Stage 1](../../audits/2026-11-17_structured_inference_stage1.md)

Локальный корень артефактов `A=/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts`, эквивалент `${GNN_ARTIFACT_ROOT}`. Каталог исходного обученного варианта `B=$A/flickr-dense-collection-20261101T140723Z/raw/runs/flickr-graphsage-l4-h256-s101`. Все пути ниже относятся к существующим файлам, проверенным локально.

| Поле | Значение / точный путь |
|---|---|
| Модель | 4 SAGEConv, размеры 500 → 256 → 256 → 256 → 7, mean aggregation, root branch, без projection и выходной L2-нормализации SAGEConv. ReLU и dropout 0,5 после трёх скрытых слоёв, без активации последнего слоя |
| Конфигурация | `$B/resolved_config.json`. Общий выполненный протокол `$A/flickr-dense-collection-20261101T140723Z/raw/protocol.json`. Шаблон в Git — [protocol.yaml](../flickr_dense_sweep_20261031/protocol.yaml) |
| Checkpoint | `$B/checkpoint.pt`, 2 095 403 байта. SHA256 `dcbf0722a61842ec2db4f24491e90ffaddabea0749ce411898110d3025fb5566`. Тензорный SHA256 `0efe1c3c76264c86be993437346aa589e9fc9b948dc4892263c75532c8b5ca27` |
| Источник обучения | `3995c6057f3460005b44bc57d98c6d5e7df7f9e2`, [реестр sweep](../flickr_dense_sweep_20261031/collection_20261101/README.md). CPU C4, PyTorch 2.0.1, четыре вычислительных потока |
| Метрики и обучение | `$B/report.json`, `$B/history.jsonl`, `$B/completed.json`. Журнал LSF `$A/flickr-dense-collection-20261101T140723Z/raw/logs/flickr-graphsage-l4-h256-s101.1611784.out` |
| Входной граф | `$A/flickr-dense-collection-20261101T140723Z/raw/inputs/graph.pt`, SHA256 `6b3414ddddabaa10a7a9c0547bcaca22e58b72ee1e933126e8d38943eb0c6da8`. Manifest там же — `manifest.json` |
| Разбиение | `$A/flickr-dense-collection-20261101T140723Z/raw/inputs/split.json`. Generated exact 60/20/20, seed 101, 53 550 / 17 850 / 17 850 узлов. Официальные маски Flickr не используются. Канонический SHA256 JSON `d32c089a6dde1bce1b014715fa503dde4b881925d71b6339a12b8f945892a429`, файловый SHA256 `3e164d5ad723c5f5916cd302fc8e11ae63b39d4d26a786ecb057de5179a8c401` |
| Граф и подготовка | PyG Flickr, 89 250 узлов, 500 FP32-признаков, 7 классов, 899 756 направленных записей рёбер, 449 878 пар, без self-loops в сохранённом входе. Raw float32, без нормализации признаков и добавления обратных рёбер. Processed SHA256 `05e353934a365c3574b7d458a18412dddc8ba3637d93da5ea1835dedae5e43cd` |
| Изоляция test | В сохранённом входе test-метки заменены на −1 по протоколу и прежним проверкам. Тензорный SHA256 этого входа `8c867c5982f1b3a43e2324ff70e91a64971d660dcada8744fbdd69b19a3c174c`. Full-batch transductive forward использует структуру и признаки всех узлов. Test labels для выбора отсутствуют |
| Копия для V5 | `$A/flickr-v5-collection-20261102T094740Z/extracted/dense/flickr-graphsage-l4-h256-s101/checkpoint.pt`. Байтовый хеш совпал с исходным. V5 source `686cf3f1913c2085260ccfef40c384de8580a042` |
| Замер CPU/P100 | `$A/polus-gpu-cpu-benchmark-20261110-173022/verified/flickr-graphsage-l4-h256-s101.json`, LSF 1617705, C4. Скрипт `benchmark.py` в родительском каталоге. GPU project source `32bf91cab509b58d7619797683e502f425a6f6bf` |
| Сжатый образец для форм | `$A/flickr-v5-collection-20261102T094740Z/extracted/runs/flickr-graphsage-l4-h256-s101-static_magnitude-t50-budget-v5/checkpoint.pt`. SHA256 `7665fd5b8083492753eb459330d942fb66dab391a602f446a9b34e70930314cf`. Точное сечение dense, ширины [111,230,219], 266 433 параметра |
| Сжатый образец из GPU-замера | `$A/flickr-v5-collection-20261102T094740Z/extracted/runs/flickr-graphsage-l4-h256-s101-static_snip-t30-budget-v5/checkpoint.pt`. SHA256 `c9dbd70e72be42ce5874f6370afbed62b707c65524213d8fa3fa81bec8641104`. Маска совпала с CPU/GPU-масками в замере, ширины [174,240,228], 370 801 параметр |

Обучение с seed 101 использует Adam, lr 0,001, weight decay 0,0005, максимум 500 эпох и patience 50. Минимум validation cross-entropy достигнут на эпохе **358**, остановка после эпохи **408**. DenseTrainer восстанавливает сохранённые лучшие веса. В checkpoint `best_epoch_1based=358`, в TrainResult `best_epoch=357` вследствие нулевой индексации. Поиск архитектуры ранее ранжировал validation macro-F1, затем accuracy и размер. Test в выборе не участвовал.

| Подмножество, best checkpoint | Accuracy | Macro-F1 | Cross-entropy |
|---|---:|---:|---:|
| Train | 0,574640523 | 0,273560738 | 1,204173684 |
| Validation | 0,540280112 | 0,240889798 | 1,318444014 |

Validation class counts по ID 0–6 равны [1023,1652,1320,946,4563,682,7664], prediction counts — [21,855,122,28,5497,0,11327]. Класс 5 не предсказывается. Доля крупнейшего класса 42,94%, предсказаний класса 6 — 63,46%. Это пригодный воспроизводимый технический baseline с ограниченным качеством. Один seed и generated split ограничивают сопоставление с опубликованными моделями. Альтернативы для проверки соответствия будущей статье уже существуют — GraphSAGE L3/H128 (validation 0,538824 / macro-F1 0,236442) и L2/H256 (0,528403 / 0,214996). В этом этапе они не заменяют выбранный baseline.

Журнал показывает снижение validation loss от 1,58619 на эпохе 1 до 1,37880 на 100 и 1,31844 на 358, затем до 1,33400 на 408. В конце train accuracy достигает 0,58007, validation accuracy — 0,53669. Patience отработал полностью, признаков обрыва по времени нет. Такая динамика совместима с началом переобучения, но не устанавливает его причину. `history.train_loss` посчитан с dropout до optimizer step, validation loss — в eval после шага. Их разность не является чистым разрывом качества одного состояния модели. Табличные train/val loss выше оба относятся к eval восстановленного best.

Stage 1 заново сверил хеши, все 408 записей ранней остановки, tensor identity и физические срезы 14 сохранённых V5-вариантов этого baseline. Новый forward не выполнялся. Пересчёт качества был выполнен при прежнем сборе sweep и зафиксирован в [METRIC_RELOAD_DELTAS.json](../flickr_dense_sweep_20261031/collection_20261101/METRIC_RELOAD_DELTAS.json). Это отдельное историческое свидетельство. Текущий осмотр не доказывает глобальную сходимость или соответствие чужому протоколу.
