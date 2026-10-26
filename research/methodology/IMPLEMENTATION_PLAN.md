# LOCAL-STRUCTURED-V2 — Implementation Plan

27 октября 2026. Источники: SPEC `9076b06`, независимый аудит `5c42596`,
текущая [ревизия SPEC](LOCAL_STRUCTURED_V2_SPEC.md). Отдельная ветка:
`implementation/local-structured-v2-20261027`.

В каждом этапе: независимый pytest oracle → фактический Red с причиной →
отдельный commit → минимум реализации → Green → проверка diff и gate → commit.
Существующий pytest, без GitHub Actions/нового CI. Переход последовательно;
разрешение на эксперименты не подразумевается. V1 код/имена/результаты сохраняются.

| Этап | Test-First объём | Приёмка / выход |
|---|---|---|
| **I. Local Surgery** | GCN/SAGE, два hidden слоя, nonuniform widths и min4; independent full-mask logits + activation-mask oracle; точные surviving tensors/bias/две SAGE projections; analytic P и reduction; оба порядка и повторное удаление с original IDs; strict disk reload; dtype/device/mode/requires_grad; invalid inputs и unsupported SAGE | Реальный Red до реализации, все CPU surgery tests Green; выбранные V1 tests без обучения Green; V1 source byte-identical исходному. CUDA skip отмечен, POLUS не проверен. Самопроверка результатов и отчёт в GitHub |
| **II. Static masks/scoring** | Начать после gate I и закрытия C01/M01/M04/m01: independent unique-coordinate union, analytic count на всех 841 width pairs, empty band/nearest ties/CiteSeer30; Magnitude/SNIP с bias и deterministic scoring, no validation/test oracle; seeded Random; conditional GraSP HVP/sign tests | Red → Green отдельно от I. Пять обязательных методов планируются без обязательного GraSP. Один dense checkpoint, finite scores, train-only gradients, параметрический target и actual gap; прямой scorer прежде prefix optimisation. Ни обучения, ни кампаний |
| **III. Adaptive/integration** | Начать после gate II и закрытия C02: integer k={1,2,4,8}, min4, repeat/nonmonotone layers; reward/terminal ровно один раз, STOP0/horizon/target crossing; legal-only Bellman bootstrap, resets; full config/checkpoint/split provenance; mock validation без обучения | Red → Green; отдельные V2 identifiers/paths, no legacy bonuses/target-stop в Q, actual terminal model. Ограниченный synthetic integration без fit; runtime manifests/блокеры до ручной кампании. POLUS/training отдельно разрешаются |

Gate I проверяется по тестам и ручному чтению oracle/diff; отдельный чат аудитора
и сложная инфраструктура не требуются для каждой итерации. Код тестов не вызывает
производственные helpers для вычисления ожидаемых tensor slices/count/logits.
Нельзя объявлять тест пройденным по чтению исходника или использовать skip как Green.

Ревизия R2 (27 октября): исследователь явно подтвердил решения по C01/M01/M04,
reward/STOP и Q state. Блокер доступа к содержимому старого чата снят прямым
сообщением пользователя; никаких решений по памяти не требуется.
Порядок: повторный Green I на Python>=3.10 → II Red/Green + V1 → III Red/Green + V1.

## Фактический статус

| Gate | Фактическое доказательство |
|---|---|
| I | Red commit5a28be3; Green71 passed/2 CUDA skipped. Дополнительно Python3.11.13:71/2; по выбору исследователя дальнейшая работа в conda gnn_pruning Python3.8.20 |
| II | Red c49832c:21 failed; Green4df84c5:21 passed. V1 subset31 passed |
| III | Red08e4fb8:21 failed; первоначальный Green21 passed, затем дополнительные edge/integration checks. Финальный общий V2 набор:127 passed/2 CUDA skipped; V1 regressions39 passed |

Реализованы шесть V2 identifiers и отдельный checkpoint workflow через существующую
команду prune. Отчёты: [I](../implementation/LOCAL_STRUCTURED_V2_STAGE_I.md),
[II](../implementation/LOCAL_STRUCTURED_V2_STAGE_II.md),
[III](../implementation/LOCAL_STRUCTURED_V2_STAGE_III.md).
Пример config и границы применения: [prototype guide](../../docs/local_structured_v2.md).
Следующий шаг — независимый аудит и отдельная подготовка кампании; обучение GNN,
POLUS, большие эксперименты и CI в этой задаче не выполнялись.
