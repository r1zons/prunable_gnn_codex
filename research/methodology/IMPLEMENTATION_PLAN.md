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

Этап I: CPU functional gate пройден на наблюдаемом окружении. Red: 40 failed,
31 passed, 2 CUDA skipped (commit `5a28be3`); Green: 71 passed, 2 CUDA skipped;
V1 subset: 31 passed. Python3.8 ниже package minimum3.10, CUDA/POLUS не проверены.
Подробности: [отчёт этапа I](../implementation/LOCAL_STRUCTURED_V2_STAGE_I.md).
Этапы II–III разрешены по SPEC §15 R2; выполнение продолжается последовательно.
