# Независимый научно-технический аудит LOCAL-STRUCTURED-V2

Дата: 26 октября 2026 года. Проверяемый commit: **9076b0630a95fbbfab64422ec798189f29e589df**.
Запрошенная ветка: research/local-structured-v2-draft-20261026.
Главный документ: [LOCAL_STRUCTURED_V2_SPEC.md][spec].
Статус отчёта: независимая проверка проекта протокола; не разрешение на реализацию или эксперименты.

## 1. Вердикт и границы доказательств

**NOT READY — спецификация пока не готова к реализации в качестве замороженного сравнительного протокола.**
Основа реализуема: физическая Local Surgery, единый dense checkpoint, параметрическое сжатие, отдельные идентификаторы V2 и запрет test-selection заданы разумно. Однако ещё не определены сопоставимый домен конечных моделей, полная функция возврата Q-learning и несколько существенных правил scoring/STOP. Это решения о научном эксперименте, которые программист не должен принимать молча.

Для пяти обязательных методов GraSP можно исключить по уже предусмотренному условию. Это снимает C03, но не C01/C02. Исправление всех legacy-методов не требуется: достаточно новых V2-путей с сохранением старого поведения. Cascade-vs-Local, полноценный Group Lasso, новые награды и большие графы остаются backlog.

**Обязательный минимум: Cora, CiteSeer, PubMed × GCN, GraphSAGE × 30%/50%; локальная машина и POLUS. Kaggle/Colab не используются.** Дополнительные датасеты не закрывают дефекты минимальной серии.

Что выполнено:

- Сначала прочитан INDEPENDENT_AUDIT_BRIEF.md именно на проверяемом SHA; затем все шесть новых файлов methodology/backlog, код, относящиеся к нему конфигурации и тесты, исторические реестры и первичные публикации.
- Документы и commit/tree проверены через GitHub connector. GitHub compare для локального HEAD d422263… и проверяемого SHA показывает общий предок 72293149d8fff186dd268789e7b8fea5848c9360 и только шесть добавленных документов на стороне проверяемой ветки. Локальная проверка diff от этого предка до d422263… не обнаружила изменений в src/, tests/, configs/, scripts/ и pyproject.toml. Поэтому номера строк прочитанного локального кода соответствуют проверяемому SHA. Ссылки ниже закреплены на SHA, не на moving branch.
- Сетевой git fetch оказался недоступен; это не заменялось предположением о версии. Документы получены по immutable SHA через connector; локальный checkout и его ветка не переключались.
- Выполнены только независимая арифметика и исчерпывающий целочисленный перебор 841 пары ширин для каждой архитектуры/датасета. Приложение A позволяет повторить расчёт без ML-зависимостей.
- Исходный код, конфигурации, алгоритмы, checkpoints и экспериментальные данные не изменялись; обучение, pruning, pytest, inference/benchmark, установка пакетов и LSF/SSH-запуски не выполнялись. Прочитанные тесты **не объявляются пройденными**.
- Исторические аудиты прочитаны как контекст. Их выводы и опубликованные ранее численные результаты не служат доказательствами данного аудита.
- Дополнительные локальные сведения о POLUS/окружении из ветки d422263… явно отделены в M09: этих файлов нет в проверяемом SHA. Живое состояние сервера и фактическая исполнимость V2 не проверены.

Уверенность «высокая» далее означает доказательство формулой, исчерпывающим перебором либо непосредственно прочитанным кодом. Это не эмпирическая оценка качества моделей.

## 2. Приоритеты

| ID | Приоритет | Суть | Что блокирует |
|---|---|---|---|
| C01 | Critical | Нет общего правила matched compression; CiteSeer 30% ±1 п.п. физически недостижимо | Заморозку основной сравнительной методологии |
| C02 | Critical | Полный возврат, discount, STOP и штраф не определены совместно; возможны ранний STOP и сильный overshoot | Адаптивные V2 |
| C03 | Critical, условный | Legacy GraSP вычисляет другой вектор и теряет знак; направление удаления противоположно Magnitude | Только включение GraSP |
| M01 | Major | Fixed rankings и raw full-mask score меняют пространство поиска и предпочитают меньшую компрессию | Однозначное определение статических методов |
| M02 | Major | Одинаковая surgery не делает сравнение причинным тестом «эффекта обучения» | Сильные причинные выводы |
| M03 | Major | Состояние Q не Markov; Bellman max включает недопустимые действия | Корректный V2 Q-update |
| M04 | Major | Bias, train loss, gradient batch, dropout и degeneracy scoring не заморожены | SNIP и воспроизводимый scoring |
| M05 | Major | В legacy остаются несколько независимых остановок/ratio actions, а не один V2-контракт | Адаптивную реализацию и её acceptance |
| M06 | Major | Checkpoint/split/config provenance недостаточен на общем пути CLI | Воспроизводимую кампанию |
| M07 | Major | Разные validation-бюджеты и n=3 не дают автоматического статистического превосходства | Публикационные winner claims |
| M08 | Major | Тесты не доказывают численную эквивалентность; device/dtype имеют ограничения | Запуск V2 после реализации |
| M09 | Major | Нет замороженного V2 local/POLUS runtime/environment/launcher протокола | Запуск серии на двух платформах |
| m01 | Minor | Равномерность Random по архитектурам не равномерность по маскам | Не блокирует после явного определения |
| m02 | Minor | allclose logits не гарантирует равные argmax при ничтожном margin | Формулировку acceptance |
| m03 | Minor | Средняя степень в state удвоена для bidirected edge_index | Интерпретацию graph-aware state |

## 3. Critical findings

### C01. Общая цель по имени не означает одинаковое достигнутое сжатие

**Место:** [SPEC:64–76,174–178][spec-target], [SPEC:239–275][spec-reward], [SPEC:338–344][spec-claims]; legacy [methods.py:677–716][apply-static], [workflow.py:119–134][actual-count].

**Доказательство.** Для текущих трёхслойных моделей с bias, F входами, C классами и сохраняемыми ширинами a,b:

    P_GCN(a,b)  = Fa + ab + bC + a + b + C
    P_SAGE(a,b) = 2Fa + 2ab + 2bC + a + b + C
    s(a,b) = (P(32,32) - P(a,b)) / P(32,32),  a,b ∈ {4,…,32}.

У SAGE две матрицы, но только один bias на conv: lin_l.bias; lin_r без bias. Это следует из [GCN][gcn-model] и [GraphSAGE][models] репозитория и [исходника PyG 2.6.1][pyg-sage]. Указанные F,C согласуются с таблицей 1 [оригинальной работы GCN][gcn-paper]; при запуске всё равно нужно сверить фактические dataset tensors.

Полный перебор с точными рациональными числами дал:

| Датасет, F,C | Модель | P0 | Число пар в 30% ±1 п.п. | Ближайшая к 30% пара, s | Число пар в 50% ±1 п.п. | Ближайшая к 50% пара, s |
|---|---|---:|---:|---|---:|---|
| Cora,1433,7 | GCN | 47 175 | 13 | (23,4), 29,808161% | 26 | (16,27), 49,975623% |
| Cora,1433,7 | SAGE | 94 279 | 13 | (23,4), 29,791364% | 26 | (16,27), 49,990984% |
| CiteSeer,3703,6 | GCN | 119 782 | **0** | (22,32), **31,189995%** | 29 | (16,27), 50,000000% |
| CiteSeer,3703,6 | SAGE | 239 494 | **0** | (22,32), **31,194936%** | 29 | (16,27), 50,005846% |
| PubMed,500,3 | GCN | 17 187 | 16 | (23,19), 29,952871% | 20 | (16,29), 49,967999% |
| PubMed,500,3 | SAGE | 34 307 | 16 | (23,19), 29,947241% | 20 | (16,29), 50,010202% |

Разрыв CiteSeer составляет соответственно **1,189995 и 1,194936 процентного пункта**, то есть превышает разрешённый один п.п. Это не ошибка rounding или seed и не лечится увеличением эпизодов. При a=23 даже b=4 даёт лишь 28,772270% GCN; при a=22 максимум сохраняемых параметров достигается при b=32 и даёт уже 31,189995%. Монотонность P по a,b исключает промежуточный вариант. Для SAGE тот же аргумент.

SPEC §4.7 корректно предусматривает fallback у статических методов. Но единая domain/fallback/matching-политика для Greedy и Q не определена: Greedy может перескочить, Q может остановиться почти где угодно. Из двух конечных точек на метод нельзя восстановить общий accuracy–compression frontier без дополнительных предположений.

**Последствие:** «accuracy при 30%» может сравнивать физически разные бюджеты. Простая запись achieved_sparsity в CSV сама по себе это не исправляет. Legacy static сначала интерпретирует target как долю каналов, затем workflow только пересчитывает достигнутую параметрическую долю; это не параметрический planner V2.

**Минимальное исправление до кода:** заморозить для каждого dataset/model/target множество допустимых конечных размеров и политику сопоставления. Для недостижимого CiteSeer явно опубликовать nominal 30%, nearest attainable ~31,19%, target violation; не удалять этот обязательный датасет. Либо использовать одинаковую заранее заданную ближайшую shell для сравнения всех методов, либо честно оформить адаптивные выходы как свободный trade-off и объявить unmatched результаты непарными по compression. Не интерполировать и не подменять фактический финал лучшим промежуточным.

**Уверенность:** высокая. **Блокер:** да, научный протокол.

### C02. Текущая reward/STOP-спецификация не задаёт единую задачу оптимизации

**Место:** [SPEC:213–277][spec-reward], открытые решения [SPEC:384–391][spec-open]; legacy [methods.py:278–314][terminal], [methods.py:1366–1384][q-config], [q_learning.py:35–54][q-update].

**Доказательство 1 — длина траектории.** При discount γ=1 и штрафе, добавленном ровно один раз:

    G = Σ R_t + P(s_T) = 0.6 s_T - 0.4 Σ D_t + P(s_T).

Одинаковые s_T=0.30, D_T=0.05 дают 0.16 при одном действии и 0.12 при трёх действиях с D_1=D_2=D_3=0.05. Возврат оценивает накопленную потерю на пути, а не только качество конечной модели. Это допустимая авторская цель, но она систематически поощряет крупные шаги/короткие пути. В backlog B02 это осознанно отложено; аудит не предлагает молча заменить награду.

При γ<1:

    Gγ = Σ(t=1…T) γ^(t-1) [0.6 Δs_t - 0.4 D_t]
         + γ^(T-1) P(s_T),

если P включён в последнюю pruning-transition. При отдельном STOP-transition его степень discount сдвигается ещё на один шаг. SPEC не фиксирует γ и конвенцию STOP; legacy default γ=0.9. Поэтому «одинаковая награда» не определяет одинаковый objective.

**Доказательство 2 — λ не является ограничением.** Даже в упрощении γ=1, D=0 функция 0.6s−λ max(0,|s−s*|−ε) внутри допуска строго растёт по s. Она предпочитает верхнюю границу, а за границей продолжает расти при λ<0.6. При λ=0.6 overshoot может быть безразличен. Условие λ>0.6 помогает лишь в этом упрощении, не даёт общей гарантии с path costs, discount и конечной решёткой.

**Реальный архитектурный контрпример для Greedy, без обучения.** Cora GCN, состояние (24,32), s=0.2486062533, A_t=A0 для всех кандидатов, s*=0.30, λ=0.1 — допустимое пока не ограниченное SPEC значение:

| Действие | Конец | s | Greedy score с применимым P |
|---|---|---:|---:|
| убрать 1 канал слоя 1 | (23,32) | 0.2796820350 | 0.0186454690 |
| убрать 8 каналов слоя 1 | (16,32) | **0.4972125066** | **0.1304425013** |
| убрать 8 каналов слоя 2 | (24,24) | 0.2540328564 | 0.0032559618 |

Остальные k первого слоя и k второго слоя не превосходят второй вариант при этих условиях. Greedy законно выбирает почти 50% для цели 30%. Это контрпример к гарантии попадания, не утверждение о фактических trained accuracies. Малый λ может одновременно делать ранний STOP выгоднее продолжения при большой D. Большой λ тоже не превращает myopic Greedy в планировщик: штраф виден только на завершающем кандидате, будущая стоимость незавершающих шагов игнорируется.

**Доказательство 3 — STOP0 против workflow.** STOP до удаления может вернуть dense-модель. [workflow.py:119–124][actual-count] безусловно требует P_after<P_before и завершится ошибкой. Политика доступности STOP0 ещё открыта. Не определено и то, получает ли STOP только P или также −0.4D за неизменённую модель.

**Минимальное исправление:** до реализации зафиксировать γ, единицы accuracy (доли 0…1), STOP0 eligibility, считает ли STOP шаг в T_max, его немедленную награду, точку применения/discount P и exhaustive terminal reasons. Рекомендация минимального объёма: сохранить согласованную R, STOP разрешить после ≥1 успешного удаления, STOP-transition = только P; P выдавать ровно один раз при каждой причине завершения; старые бонусы не переносить. Это предложение аудитора, не утверждённое изменение. При разрешении STOP0 нужен отдельный валидный статус unpruned/target_miss вместо аварии.

Для λ нужен заранее описанный validation-only диапазон/бюджет/критерий выбора, а не подбор до удобного test-результата. Если сохраняется soft penalty, назвать его soft target preference и отказаться от гарантии сжатия. Жёсткое ограничение допустимо только как явно утверждённое изменение протокола.

**Уверенность:** высокая. **Блокер:** да для Greedy/Q.

### C03. GraSP: H·W+g вместо H·g и противоположное направление удаления

**Место:** [SPEC:148–160][spec-grasp], [methods.py:448–484][grasp-code], [methods.py:28–31][prunable-set]; [GraSP, §4, eq.7–8, Algorithms 1–2][grasp-paper].

**Подтверждённая ошибка legacy:** код дифференцирует g(W)ᵀW, поэтому

    ∇W [g(W)ᵀW] = H(W)W + g(W),  а не H(W)g(W).

Для L(w)=½(w−1)², w=2: g=1,H=1, нужный Hg=1, текущий HW+g=3. Нужный signed-score −2; legacy после abs получает +6. Это алгебраическое доказательство без запуска source.

Для φ(W)=||∇L(W)||² и удаления множества уникальных параметров R, δ_i=−w_i 1(i∈R), линейная часть:

    φ(W+δ) − φ(W) ≈ 2δᵀHg = 2 Σ(i∈R) [−w_i(Hg)_i].

По исходному GraSP удаляют **наибольшие** q_i=−w_i(Hg)_i: они повышают либо меньше ухудшают gradient norm. Для полной структурной маски нужно **максимизировать** S_removed; сохраняемые prefixes строить по обратной, «наименьшей deletion-score», ориентации. Для Magnitude/SNIP удаляемую сумму минимизируют. Пример q=(−2,+1): GraSP удалит +1; перенос keep-largest из legacy/Magnitude оставит +1 и удалит −2. [Оригинал, Algorithms 1–2 и §4][grasp-paper].

Корректный HVP: g=grad(L,W,create_graph=True); v=stop_gradient(g); h=grad(Σ g_i v_i,W). Не нужно материализовать Hessian. Если взять grad(Σg_i²), получится 2Hg; это отдельный множитель, который нужно документировать.

**Ограничение адаптации:** после обучения и без fine-tuning сохранение градиентного потока не является доказанным суррогатом сохранения accuracy. При g=0 получим q=0 независимо от разрушительности конечного удаления. Структурная агрегация уникальных координат корректна для выбранной линейной аппроксимации, но не доказывает полезность finite deletion.

**Минимальное исправление:** отдельный grasp_pt_local_v2; signed tensors; фиксированный train objective/parameter vector, HVP oracle и сортировка; Gate: analytic/finite-difference HVP, group-union test, stationary/near-zero test, finite scores. При непрохождении исключить GraSP, как уже разрешает SPEC. Не исправлять задним числом legacy-метрики и не переименовывать их в canonical GraSP.

**Уверенность:** высокая в алгебре и направлении; эмпирическая полезность PT — не установлена.
**Блокер:** только GraSP; не должен задерживать пять обязательных методов.

## 4. Major findings

### M01. Fixed ranking — эвристика, а raw full-mask score предпочитает меньшую компрессию

**Место:** [SPEC:90–108,124–134,162–178][spec-static], [MASK_SCORING:31–49,67–69][mask-prefix]; legacy [methods.py:66–110][layer-score].

Контрпример уже на уровне аддитивного surrogate: две переменные строки/колонки Q1 имеют значения

    [[0,6],
     [6,5]].

Остальные связанные scores равны нулю или добавляют одинаковые константы. Marginal row/column sums равны (6,11); keep-largest rankings сохраняют нижнюю строку и правую колонку. При сохранении одной из двух в обоих слоях retained score=5, removed score=12. Сохранение одной из перекрёстных ячеек даёт retained=6, removed=11. Ранжирование пропустило лучший вариант даже для собственного surrogate. Контрпример переносится на min width=4 добавлением трёх заведомо сохраняемых anchor-каналов в каждый слой. Это не опровергает оговорку SPEC об approximate search; опровергает возможное заявление об оптимуме маски.

Вторая проблема независима: при q_i=1 для всех учитываемых параметров S_removed=P0−P=P0s. Минимизация raw sum внутри допуска всегда выбирает **наименьшее достижимое сжатие**. Например, Cora GCN target30 имеет диапазон 29.019608%…29.808161%; выбирается (23,16),29.019608%, а ближайшая к цели (23,4),29.808161% проигрывает. PubMed GCN диапазон 29.010298%…30.895444%. Это контрпример к нейтральности выбора compression в допустимой полосе; для реальных неодинаковых scores не утверждается неизбежность такого выбора.

**Последствие:** у статик выбор размера, каналы и criterion смешиваются; статический Magnitude может выглядеть точнее отчасти из-за меньшего удаления. Положительные и signed criteria имеют разную зависимость от размера.

**Минимум:** явно определить channel score, полную маску, ориентацию, bias treatment и tie-break. Для сравнения можно до saliency задавать общий ближайший compression shell, затем выбирать score внутри него; либо оставить исходный band-search и честно учитывать систематическое различие достигнутых бюджетов. Деление на число удалённых параметров не является автоматическим исправлением: оно меняет criterion и требует отдельного утверждения. Глобальный combinatorial search для статьи не нужен.

**Блокер:** да, определение статических V2; fixed-ranking approximation сама по себе допустима.

### M02. Одинаковая Local Surgery не изолирует эффект RL

**Место:** [SPEC:187–207,239–275][spec-reward], [SPEC:338–344][spec-claims], [backlog B01:5–14][backlog].

Статики используют один dense-ranking и одну candidate mask на width pair; адаптивные методы пересчитывают Magnitude после каждого удаления. У них могут различаться каналы даже при одинаковых финальных ширинах. Greedy имеет принудительную остановку у нижней границы цели, Q имеет STOP и возможность продолжать за целью; STOP отсутствует в Greedy action set. Поэтому одинаковы pruning actions, но **не полное пространство решений и не stopping task**.

Даже в сравнении Greedy/Q меняются lookahead, бюджет validation, objective horizon и STOP. Это допустимое сравнение двух законченных алгоритмов, но положительный результат не доказывает, что причина — обучение Q-таблицы.

**Минимум:** сузить основной вывод до «сравнение конкретных процедур при общей операции удаления, с учётом их ресурсов и фактического сжатия». Причинный эффект RL потребовал бы отдельного контроля с одинаковой stopping/task semantics; такой контроль и Cascade-vs-Local можно оставить backlog, если причинный claim снят.

**Блокер:** для причинного вывода, не для описательной статьи.

### M03. State aliasing и некорректный Bellman maximum

**Место:** [state.py:18–51][q-state], [methods.py:1386–1396][q-buckets], [q_learning.py:35–54][q-update].

State не содержит точных ширин по слоям, retained identities или remaining steps; compression/accuracy bucket не восстанавливает эти данные. Контрпример Cora GCN при одинаковой accuracy:

- (4,32): s=0.870121887;
- (5,31): s=0.839321675.

Оба имеют remaining ratio36/64, current_sparsity_bucket для (0.7,0.9], target_gap=0 и одинаковые graph/accuracy buckets при цели0.30. Но действие layer1,k=1 в первом запрещено, во втором разрешено. Даже точные ширины не полностью Markov: разные surviving channels дают разные следующие accuracy. А одна и та же маска после одного k=8 либо нескольких малых действий имеет разный остаток T_max; время также часть finite-horizon state.

В update_q next_max берётся по **всем** сохранённым next_row values. Если в агрегированном next state есть Q(invalid)=100 и Q(valid)=−1, r=0,γ=.9,α=1, код обновит значение до90, а легальный максимум даёт−.9. Неопробованное допустимое действие со значением по умолчанию0 тоже не включается явно в такой max. Фильтр select_action не исправляет training target.

**Минимум:** V2 update получает next valid action keys, включая STOP по единому правилу; max только по ним с default0, terminal bootstrap0. В state добавить exact widths, signed target deviation и remaining steps. Retained-set signature даёт более точную идентификацию на фиксированном checkpoint, но увеличивает таблицу; если остаются buckets, назвать agent approximate aggregated-state heuristic и не ссылаться на гарантию сходимости tabular Q-learning для MDP. Схема state должна быть версионирована до пилота.

**Блокер:** да для Q correctness; не требование переходить на deep RL.

### M04. Scoring нуждается в полном операционном определении

**Место:** [SPEC:124–158][spec-grasp], [SPEC:384–388][spec-open], [MASK_SCORING:53–56][mask-scores], [methods.py:28–31][prunable-set], [methods.py:412–484][grasp-code], [environment.py:324–335][env-score].

Legacy выбирает только tensors ndim>1; исчезающие hidden biases не получают saliency. Параметрический count при этом включает их — сам по себе такой разный набор разрешён, но это научный выбор, пока не сделанный в SPEC. У Magnitude/adaptive bias тоже пропущен. При hidden bias=10 и почти нулевых incident weights канал может получить почти нулевой score, несмотря на ненулевую активацию.

Для SNIP-based PT Σ|w_i g_i| по union — допустимая аддитивная эвристика. Это **не** |Σw_i g_i| и не точная производная общего channel gate. При w_i g_i=(1,−1) эти величины равны2 и0; cancellation меняет ranking. Для L=½(w−1)²,w=1 saliency0, но зануление даёт loss increase0.5. Нулевой trained gradient не доказывает ненужность веса. Исходный SNIP — pruning до обучения с последующим обучением; PT structural no-finetune — отдельная адаптация. [SNIP §4, Algorithm1][snip-paper].

SNIP/GraSP legacy вызывают score_model.train(): dropout активен. Для SPEC следует выбрать eval-scoring с gradients enabled либо seeded fixed dropout samples; model.eval() не запрещает autograd. Train batch для node classification нужно определить как train-node loss на общем transductive графе или как явно иной sampled subgraph: это разные objectives. Test/val features и edges в transductive forward допустимы; их labels не должны участвовать в loss.

Для GraSP дополнительно определить W: all trainable parameters либо только subset. Если исключить bias из gradient vector, пропадёт H_weight,bias g_bias; это уже другой Hg даже у weight-coordinate. Фиксированный output bias не исчезает из модели, но может влиять на HVP.

**Минимум:** полный список score tensors, bias policy, loss reduction и weight-decay policy, train indices/hash, режим, dtype, dropout/RNG, zero/unused-gradient и finite-value rules. Для стабильного первого варианта разумны full-graph CE mean на train indices, eval/dropout-off, gradients по явно заданному W. Не объявлять низкую точность адаптации доказательством программной ошибки.

**Блокер:** да для SNIP/scoring; GraSP условный.

### M05. V2 нельзя получить только переключателем local в существующем Q

**Место:** [environment.py:82–161,246–277][env-stop], [methods.py:139–240][action-filter], [methods.py:1608–1637,1703–1707][q-stop], [methods.py:1859–1883][q-deploy-stop], [methods.py:1202–1300][greedy-old].

Подтверждены независимые legacy-пути:

1. Env автоматически done при target_sparsity_reached.
2. Deployment отдельно проверяет target до выбора действия.
3. Training и deployment дополнительно останавливаются по max_accuracy_drop.
4. Старые terminal bonuses/gap penalties зависят от другого tolerance и не штрафуют overshoot симметрично.
5. Действие — prune_ratio текущей ширины с round и clipping, а не абсолютное k. Разные ratio могут создавать одинаковый физический переход.
6. Adaptive выбирает слой по mean importance, использует Cascade и monotone layer order; он не оценивает восемь candidate validation rewards.
7. Invalid env.step увеличивает step_count, затем может вернуть done=False даже при исчерпанном max_steps; внешний for-limit и env counter могут расходиться.

**Минимум:** отдельные versioned env/action/terminal contracts для V2 с единым valid(l,k), без clipping invalid k; разрешить возвращение к слоям. Все причины STOP/timeout/no-pruning-actions должны возвращать один terminal transition и финальную фактическую модель. Уточнить, значит ли «нет допустимых действий» отсутствие pruning actions или вообще actions с учётом STOP. Программную ошибку surgery нельзя маскировать под обычный научный STOP/target miss.

Сохранить legacy tests на старые остановки; новые тесты должны доказывать противоположное требуемое V2-поведение, а не заменять старые asserts.

**Блокер:** да для adaptive V2 acceptance. Это подтверждение необходимых изменений, а не претензия, что draft обязан уже иметь реализацию.

### M06. Общий CLI не доказывает checkpoint/split/config identity

**Место:** [workflow.py:61–104,136–145][checkpoint-path], [workflow.py:470–489][split-load], [training/workflow.py:358–385][signature], [config/schema.py:74–104][typed-config], [config/loader.py:36–58][config-loader].

prune_from_checkpoint загружает любой совместимый по shapes model_config/state_dict, затем отдельно создаёт/читает splits.yaml из output_dir. На этом входе не проверяет dense run_signature против dataset/seed/split текущего pruning run. Поэтому CLI может загрузить checkpoint seed101 с split202 и получить правдоподобные метрики без ошибки. Защита dense resume не равна защите произвольного pruning CLI.

ExperimentConfig.to_dict содержит run/data/model/training/device/benchmark, но не pruning/q_learning/adaptive blocks. Их workflow читает отдельно из raw YAML; config_hash, полученный только из resolved.to_dict(), не является hash полного алгоритмического протокола. Два файла с разными λ/actions/episodes могут иметь одинаковый такой hash. Нельзя переносить новый tolerance как произвольное поле и считать, что snapshot гарантирует его использование.

В текущем loader используются Planetoid data, но pipeline строит собственный seeded random split60/20/20 без стратификации; public masks автоматически не становятся экспериментальным split. SPEC не фиксирует эти доли и feature normalization отдельным обязательным пунктом. Следует явно утвердить random60/20/20 и preprocessing либо иной единый вариант, сохранить indices и class counts; сравнивать accuracy с опубликованным public-split GCN как напрямую сопоставимую нельзя. Разные seeds здесь меняют и split, и initialization — paired повтор означает общее значение обоих факторов между методами, а не изолированную дисперсию policy RNG. [data/splits.py:33–58][data-splits], [factory.py:70–77][data-loader].

Исторический runner жёстко задаёт семь V1 methods, Kaggle paths и tolerance0.025 [run_kaggle_pilot.py:24–80][old-runner]. Переиспользовать его без V2 preflight опасно.

**Минимум:** полный merged protocol snapshot и hash всех реально используемых raw/typed blocks, включая новые identifiers и defaults; immutable checkpoint identity и canonical tensor hash, graph/raw-data hash, ordered split hash и train-score subset hash. Перед pruning — fail closed при несоответствии dataset/model/seed/split; сохранять исходные dense channel IDs через последовательную local renumbering. Новый run_id/output root для V2; не трогать старые кампании.

**Блокер:** до серии. Существующие хорошие dense-resume checks и nonuniform architecture metadata следует переиспользовать.

### M07. Validation budget, пилот и статистика требуют правил, а не только логов

**Место:** [SPEC:235–277][spec-reward], [SPEC:302–318][spec-platform], [SPEC:328–344][spec-claims], [suite.py:86–166][aggregation], [methods.py:428–467][grasp-code], [methods.py:902–964][static-timer].

При H32 максимум восемь pruning actions. Greedy: до160 candidate validations на rollout; Q при30×20: до600 training transitions плюс20 deployment. Это разные инвестиции в выбранную маску; число Q-эпизодов не число независимых научных повторов. В legacy цикл дополнительно повторяет validation в _current_rollout_metrics и env.step before/after, так что620 pruning transitions могут потребовать около1860 full-graph validation forwards, не считая итоговых метрик. Подробный compute budget — раздел7.

У статиков score часто вычислен до start timer в apply, поэтому legacy pruning_time_sec не включает gradient/HVP scoring; сравнивать его напрямую со всем policy-training time Q некорректно.

Повторный validation-feedback повышает риск подгонки; это не test leakage само по себе. Отчёт по calls необходим, но не превращает сравнение в равнобюджетное.

Три seeds — приемлемый ограниченный описательный минимум, не доказательство superiority. Для paired t CI при n=3 полуширина =4.303 SD_diff/√3≈2.484 SD_diff. При трёх положительных paired differences минимальный двусторонний exact sign-test p=0.25. Повторные inference passes и тысячи вершин одного графа не заменяют independent seeds. Не следует выбирать метод по test-derived CI или объединять six datasets/models как однородные iid реплики.

**Минимум:** заранее выбрать primary comparison/metric, paired analysis внутри dataset/model/target и правило unmatched compression; публиковать все seed-level outcomes и failures. Определить бюджет λ/episodes pilot и критерий завершения по validation/time до просмотра test. Если seed101 использован для protocol tuning, отметить его как development/pilot, а202/303 как confirmatory; не объявлять три заранее независимых confirmatory repeats. При сроковом ограничении допустима честная descriptive статья без статистического winner claim. Равнобюджетный random-search контроль полезен, но не обязателен при снятом causal claim.

**Блокер:** для сильного сравнительного вывода; описательная минимальная серия возможна после pre-registration.

### M08. Доказательство surgery корректно в ограниченной архитектуре; numerical gate пока отсутствует

**Место:** [structural.py:172–207,247–332][surgery], [tests/test_surgery.py:78–99,159–184][surgery-tests], [tests/test_pruning_methods.py:197–212][grasp-test], [MASK_SCORING:59–65][mask-tests].

Чтение кода подтверждает row/column slicing GCN, обе ветви SAGE и копирование bias; shape tests и nonuniform checkpoint rebuild есть. Но эти тесты проверяют размеры/forward shape, а не идентичность surviving tensors или logits относительно полной маски. Тест GraSP проверяет только уменьшение размеров — ошибочная производная его не нарушает.

**Аналитическое положительное заключение.** При текущем GCN aggregation matrix Â действует по node axis и не зависит от hidden width; выбор колонок commute с Â. ReLU поэлементна и ReLU(0)=0; eval выключает dropout. У SAGE mean aggregation также commute с выбором каналов, а обе downstream projections могут быть sliced одинаково. Поэтому при full masks

    W0' = D1 W0;  W1' = D2 W1 D1;  W2' = W2 D2
    b0' = D1 b0;  b1' = D2 b1;  b2' = b2

dense logits математически совпадают с physically retained subnetwork после соответствующего embedding. Это верно для текущих defaults mean, root_weight=True, project=False, normalize=False; не для произвольного SAGEConv. Нельзя распространять доказательство на normalized features, project=True, learned aggregators, BatchNorm/residual без новой проверки.

**Ограничение кода:** rebuild создаёт новые nn.Linear/Conv на default CPU/dtype [structural.py:247–332][surgery]. Это может временно дать mixed-device model и понизить float64 до float32. В gradient scoring workflow rebuild model не перемещается на context.device, но graph перемещается; при CUDA это создаёт device mismatch на direct gradient-scoring path [workflow.py:70–78][checkpoint-path], [methods.py:419–435][grasp-code]. На CPU float32 core этот конкретный device-defect не проявляется.

**Минимум:** tests для обеих моделей/обоих порядков surgery с original IDs, nonzero biases, обоими SAGE projections, boundary widths и строгой state_dict reload; full-mask logits reference в eval. Допуски задать по dtype/backend. Проверять либо сохранять device/dtype при rebuilding; до проверки GPU зафиксировать CPU-only исполнение, не обещать CUDA. Не подключать Torch-Pruning как срочную замену: [DepGraph][depgraph] — reference по зависимостям, не доказанная drop-in реализация для этого кода.

**Блокер:** numerical/device acceptance до запуска; в этом аудите source/tests не исполнялись.

### M09. Local/POLUS исполнимость вероятна, но V2 environment и размещение не заморожены

**Место:** [SPEC:283–318][spec-platform], [SPEC:384–392][spec-open], [pyproject.toml:10–27][package], [utils/seed.py:8–25][seed-code], [evaluation/benchmark.py:14–45,64–83][timing].

pyproject задаёт только нижние границы Python≥3.10,Torch≥2.0,PyG≥2.4 и обязательный mlflow, хотя алгоритм его не требует. Это не lockfile и не доказательство переносимости второго порядка autograd на POLUS POWER8. set_seed задаёт RNG seeds, но не deterministic-algorithm policy/BLAS threads/backend.

**Дополнительные сведения вне проверяемого SHA:** в локальной consolidation-ветке d422263… [environment.json][local-env-extra] записаны Python3.8.20,Torch2.2.2,PyG2.6.1; Python ниже объявленного package minimum. Это показывает расхождение зафиксированных operational records, а не доказывает невозможность исполнения source3.8. Новый package install по metadata должен использовать поддерживаемый Python. [SERVER-STATUS.md:35–89][polus-extra] описывает Python3.10.22,Torch2.0.1+polus.power8.1 CPU-only,PyG2.4.0,POWER8/RHEL7.5; [GNN-POLUS.md:17–21][polus-gpu-extra] документирует старые GPU/CUDA ограничения. Эти записи не проверялись живым запросом и не доказывают V2 HVP readiness.

**Минимум:** два immutable environment manifests/locks с wheel/source-patch hashes, CPU/OS/BLAS/threads, graph preprocessing и invocation. Перед кампанией — утверждённые dependency/device checks и numerical gates. CPU на обеих платформах — реалистичный минимальный путь; GPU не обязательный.

Для cross-platform quality лучше переносить **один и тот же** frozen checkpoint и split, проверяя byte hash плюс canonical tensor hash, а не независимо переобучать «тот же seed». Независимое переобучение на другом backend — отдельный platform factor; даже одинаковые checkpoints не делают повтор seed101 новым независимым seed. Качество можно сопоставлять как paired platform replication; pooling требует заранее определённой единицы анализа и проверки численной/масочной сопоставимости. Время всегда отдельно по платформам.

Нужно решить, где тренируются18 dense conditions и где исполняются method searches; «local и POLUS» не означает обязательного дублирования всей серии. Перенос только финальных pruned checkpoints даёт cross-platform inference replication, но не replication всего search.

**Блокер:** запуск кампании; не математическая осуществимость core. До срока допустим минимальный CPU pilot после отдельной авторизации, но аудит его не запускал.

## 5. Minor findings

### m01. Закон Random нужно утвердить буквально

**Место:** [SPEC:112–122,174–178][spec-static].

Если B — набор допустимых width pairs, то предложенная процедура задаёт

    Pr(K1,K2) = 1 / [|B| · binom(32,|K1|) · binom(32,|K2|)].

Она равномерна по архитектурам, условно равномерна по подмножествам; по всем маскам глобально не равномерна. Это нормальный baseline, но не random-search NAS из [Li et al.,CVPR2022][random-paper], где другой search/training setup. При пустом B сначала определить nearest shell, включая все ties, затем тем же законом выбирать из shell. Зафиксировать независимые RNG streams для width/channel выбора и цели; не отбирать лучший Random по validation.

**Минимум:** утвердить закон, deterministic ties/fallback, seed derivation. **Блокер:** только отсутствие определения до запуска.

### m02. allclose logits и точное равенство predictions — разные проверки

**Место:** [MASK_SCORING:61][mask-tests].

Logits (0,10⁻⁸) и (10⁻⁸,0) могут быть allclose с обычным atol, но иметь разные argmax. Поэтому numerical equivalence, margin-aware prediction agreement и exact symbolic/mask correctness нужно проверять отдельно.

**Минимум:** float64 toy/reference с контролируемым margin; отдельный tensor/logit tolerance и отчёт об argmax disagreements на почти равных logits. На реальных платформах не требовать побитной идентичности как следствия одного allclose. **Блокер:** нет.

### m03. Средняя степень state удвоена при bidirected edge_index

**Место:** [state.py:29–37][q-state].

avg_degree=2*num_edges/num_nodes, но у PyG undirected graph обычно каждая связь хранится двумя направленными entries. Тогда среднее число соседей =edge_index.size(1)/N, а код считает вдвое больше. Для одного фиксированного графа это константа и не меняет выбор между текущими pruning states; межграфовая интерпретация bucket неверна.

**Минимум:** определить directed-entry count/unique-undirected count и self-loop policy. Исправлять в V2 state version; старые Q-таблицы не переинтерпретировать. **Блокер:** нет для per-condition heuristic, да для точного graph-feature описания.

## 6. Что математически проверено и что не доказано

### 6.1. Unique mask score и prefix sums

Для удалённых каналов I1,I2, retained K1,K2:

    S_removed(M) = Q_total − Q_retained(M).

В W1 сохраняются только строки K2 и колонки K1; удаляется union строк I2 и колонок I1. Поэтому shared intersection не удваивается. Пример: один вес score5 соединяет два удаляемых канала; сумма marginal channel scores даст10, unique union —5. Нужен union, не сумма независимо посчитанных channel scores.

Для фиксированного ranking формула [MASK_SCORING:37–49][mask-prefix]

    Qkeep(a,b) = P0(a)+P1(b,a)+P2(b)+Pb1(a)+Pb2(b)+Qfixed

**верна**, в том числе для signed q. У SAGE нужно суммировать отдельные prefix contributions обеих матриц каждого conv и один фактический bias. Qfixed включает retained output bias; он сокращается при Qtotal−Qkeep. Это не доказательство правильного GraSP direction или predictive accuracy surrogate.

Предобработка O(P+H logH), память score tensors O(P), небольшие prefix tables O(H²);841 lookup — O(1) score accesses на pair. Без prefix прямая проверка O(841P) тоже невелика для H32 и не требует841 forwards. Пример Q1 из технической записки даёт retained top-left2×2=12 — арифметически верно.

### 6.2. Достижимость width pairs и горизонт20

Если убрать автоматический target-stop, любая пара a,b∈[4,32] достижима по количеству каналов: каждое r=32−a∈[0,28] представимо суммой {1,2,4,8} не более чем пятью действиями; для двух слоёв достаточно не более10. Поэтому T_max20 не исключает размеры статического перебора **для Q с разрешённым продолжением**. Но это не равенство масок, выбранных текущим Magnitude, и не гарантия, что Greedy с ранней остановкой дойдёт до той же пары.

### 6.3. Матрица противоречий

| Тема | Каноническая SPEC | Код проверяемого SHA | Литература / независимая проверка | Итог и уверенность |
|---|---|---|---|---|
| Сжатие | 1−P/P0, включая bias | Workflow пересчитывает верно; static planner всё ещё channel-ratio/Cascade | Прямой count по tensor shapes | Определение верно; planner V2 нужен. Высокая |
| Local Surgery | Оба скрытых слоя независимо | Низкоуровневая local функция есть; static/adaptive default Cascade | Срезы и dependency logic согласуются с текущими PyG defaults | Переиспользовать ограниченно; numerical test отсутствует. Высокая/ограниченная runtime |
| Fixed rankings | Одна маска на пару, approximate | Static saliency только первый hidden | Контрпример M01 | Нельзя писать global optimum. Высокая |
| Prefix lookup | Необязательная оптимизация | В V2 ещё отсутствует | Union identity доказана | Корректно при fixed additive tensors. Высокая |
| SNIP | PT structured adaptation | Train/dropout-on, без bias scores | SNIP до обучения и с последующим training | Имя PT корректно, полезность no-FT эмпирическая. Высокая |
| GraSP | −wHg,signed,conditional | −w(HW+g),abs,keep-largest | Alg.1–2 удаляют largest signed deletion score | Legacy не original; C03 gate. Высокая |
| Reward | 0.6Δs−0.4D + symmetric P | Q использует0.4+0.2 proxy; legacy terminal bonuses | Закрытая форма возврата C02 | Историческая R сохранена осознанно; полный objective открыт. Высокая |
| Q STOP | Explicit; target не останавливает | Target-stop в env и deployment; threshold-stop в wrapper | Прямой control flow | Новые versioned paths обязательны. Высокая |
| Greedy | До8 lookahead actions;local | Mean-score layer choice,Cascade,ratio,threshold | Не равен заявленному алгоритму | Не считать старый adaptive реализацией V2. Высокая |
| RL causal claim | Не смешивать пространство и алгоритм | State/actions/STOP неодинаковы между процедурами | AMC/PuRL — другие agents/tasks | Только сравнение полных процедур без нового контроля. Высокая |
| Platforms | local/POLUS,отдельная latency | Старый runner Kaggle-oriented | PyTorch не гарантирует одинаковые результаты backend/version | Новый manifest/runner boundary. Высокая; live server не проверен |
| Evidence | V1 отдельно от V2 | Исторические реестры всё ещё направляют в Kaggle plan | Документы не экспериментальное доказательство | Add-only supersession notice; старые данные сохранить. Высокая |

## 7. Вычислительная осуществимость и локальная машина/POLUS

Без запусков нельзя обещать конкретное число часов. Арифметическая оценка показывает, что минимальная матрица реалистична для CPU, а основная неопределённость — policy search, HVP и orchestration.

- Для одного seed:6 dense checkpoints,60 post-prune результатов без GraSP или72 с ним. Для трёх seeds:18 dense и180/216 post-prune. Это расчёт **одной полной серии**, не требование удвоить её на каждой платформе.
- Dense training ceiling:18×500=9000 training epochs. Full-batch trainer обычно делает training forward/backward и validation forward на эпоху; early stopping уменьшает число. Это не измеренное время.
- Greedy:12 условий dataset/model/target на seed × максимум160 candidate validations =1920, или5760 за3 seeds, плюс baseline/final/report calls.
- Q при30 эпизодах:12×(600+20)=7440 pruning transitions/seed,22320 за3 seeds. Legacy без кэширования повторных evaluations может дать порядка22320 validation forwards/seed,66960 за3 seeds. STOP и early termination уменьшают эти верхние ориентиры, а ошибки/retries и дополнительные diagnostics могут увеличить общую работу.
- Для K вариантов λ/episode budget стоимость пилота добавляется, а не скрывается в «одном запуске Q». Если score/reward evaluation кэшируется, ключ обязан включать checkpoint/retained identities/split/mode; cache policy одинаково документируется.
- Float32 features занимают примерно15.5MB Cora,49.3MB CiteSeer,39.4MB PubMed; это только N×F×4, не peak memory. Activations, edge messages/scatter, gradient graphs, copies и HVP могут потребовать существенно больше.
- Максимальный dense SAGE core здесь239494 параметра (~0.96MB float32). Полный Hessian порядка P² недопустим; vector product нужен обязательно.
- Candidate models оценивать последовательно либо ограниченно, не держать160 copies одновременно. Физическая surgery копирует модель, validation включает полный граф даже при loss на подмножестве вершин.
- Время inference: зафиксировать число threads/processes, warmups, timed passes, synchronization, eval/no_grad, graph/device resident policy, порядок dense/pruned измерений и raw samples. mean/std одного tight loop не заменяют повторные независимые timing blocks. Graph preprocessing и transfers должны иметь явную границу измерения.
- Существующий CPU memory estimate = parameters+input+output; он не измеряет peak RSS и не учитывает все временные тензоры. Не называть его peak runtime memory.
- На POLUS нужны переносимая environment recipe и архивируемый launcher hash, job resources, per-condition output/cache isolation, restart rules и wall-clock/queue distinction. Старый идентификатор pilot-cpu-101-c4-v2 относится к другой кампании и **не означает LOCAL-STRUCTURED-V2**.
- Новые операции проверки/обучения на сервере требуют отдельной авторизации; в рамках аудита не выполнялись.

Практический порядок после утверждения:5 core methods на всех трёх датасетах/двух моделях; CPU numerical gates; небольшой validation-only pilot стоимости; затем полный seed101 и зафиксированные202/303 согласно утверждённой процедуре. GraSP можно исключить; расширение датасетов/моделей, GPU-port и тяжелый NAS не должны задерживать минимум.

## 8. Минимальные acceptance tests и scientific gate

Это **план проверок**, не отчёт о запуске. Первые две группы закрываются до научного обучения.

| Проверка | Минимальный oracle / критерий |
|---|---|
| Count/target | Все841 пары обеих моделей; analytic P == sum(numel) physically rebuilt; таблица C01; nearest ties и empty band |
| Unique score | Direct union == total−retained; shared intersection counted once; bias и обе SAGE projections |
| Prefix, если есть | Direct == prefix для signed/unsigned tensors, permuted rankings, min/max widths; никакой accuracy-oracle |
| Surgery equivalence | Nonzero biases; оба слоя; оба порядка; gapped original IDs; eval logits against full mask; сохранены F,C,edges |
| Checkpoint | Strict load nonuniform widths; tensor equality; logits before/after load; original retained IDs |
| SNIP | Analytic gradient/mask derivative на smooth toy; отличие sum(abs) от abs(sum); no val/test labels; zero gradient policy |
| GraSP conditional | Analytic Hg; central difference [g(W+ηv)−g(W−ηv)]/(2η); fixed v; sign/orientation; no abs; bias cross-HVP |
| Action contract | k={1,2,4,8}; min4; invalid filtered без clipping; layer order2→1→2; max20; deterministic ties |
| STOP/terminal | STOP0 rule; target crossing не done у Q; STOP после crossing; Greedy boundary; max/no-actions; P ровно один раз; no legacy bonuses |
| Q update | Next max только legal actions; unvisited legal0; terminal bootstrap0; remaining horizon и state aliasing scenario |
| Reset | Каждый episode и deployment из одинакового dense tensor hash/split; исходная модель не мутирована; итог — actual final |
| Provenance | Несовпавшие dataset/split/checkpoint отклоняются; full config hash меняется при λ/actions/episodes; метод/платформа/seed сохраняются |
| Reproducibility | Isolated same-seed scoring/mask повторимы в одной pinned среде; независимость от порядка запуска других методов/RNG consumption |
| Isolation | No test label влияет на score/reward/λ/episode count; eval-scoring с gradients enabled; no fine-tuning |
| Legacy regression | Существующие V1 identifiers/semantics и архивные outputs не перезаписаны |
| Platform | На каждой выбранной среде import/HVP/surgery/reload gate; quality comparison отдельно от timing; GPU только после device gate |

**Scientific gate до реализации:** утвердить C01 matching/fallback, C02 полный return/STOP, M01 score/size-selection и M04 scoring contract; либо исключить GraSP, либо утвердить C03 oracle. Указать final method identifiers и полную конфигурационную схему. Авторские решения фиксировать отдельной ревизией SPEC/commit.

**Gate до экспериментов:** отдельная авторизация на код; новая реализация и пройденные acceptance в pinned environments; protocol/checkpoint/split/run manifest; заранее определённый validation pilot budget и analysis plan. Training cap500/patience50 — пределы, не сертификат сходимости dense-модели: сохранить best epoch,loss curves,stop reason и cap-hit flag.

**Gate до статьи:** завершённый минимум Cora/CiteSeer/PubMed; все planned method-target rows либо явно объяснённые failures; отсутствие test-driven исключений; compression-matched coverage; seed-level paired таблицы; separate platform latency; claims ограничены тем, что идентифицирует дизайн. При3 seeds и изменении протокола на pilot — описательная интерпретация с явным development/confirmation статусом.

## 9. Срочные исправления и то, что остаётся отложенным

**До начала реализации:**

1. Зафиксировать nearest attainable и общий comparison domain, особенно CiteSeer30.
2. Записать полный mathematical return и STOP semantics; выбрать soft-target или hard-budget interpretation; не вводить скрытую гарантию.
3. Заморозить bias/train-batch/dropout/ranking/tie/fallback правила и V2 action/state/config schemas.
4. Принять решение о GraSP по формальному gate; разрешённое исключение — минимальный безопасный путь по срокам.
5. Утвердить minimal local/POLUS placement и validation-only pilot/statistical plan.

**При реализации, после отдельного разрешения:** новые identifiers, integer actions, legal Bellman max, все stop paths, full provenance и acceptance matrix. Эти изменения не выполнены аудитором.

**Отложить:** Cascade-vs-Local causal ablation, exact joint-mask optimizer, utility-difference reward, полноценный Group Lasso, большие графы, NAS, deep RL, GPU environment migration. Не требовать эти исследования, чтобы завершить ограниченную описательную статью.

**Документация:** добавить новый указатель superseded-for-V2 в research hub/state/claims/evidence и ссылку на новый протокол; не стирать Kaggle-планы и старые claims. В README методологии различать «готово к аудиту», «методология утверждена» и «реализация проверена»: это три разных статуса.

## 10. Первичные источники и пределы переноса

Все перечисленные ссылки проверены при аудите; научные утверждения выше опираются на первичные работы, а API-детали — на официальную документацию. DOI указан только там, где он подтверждён; для ICLR достаточно устойчивого arXiv/OpenReview URL.

| Источник | Что проверено и как относится к V2 |
|---|---|
| [Lee,Ajanthan,Torr. SNIP,ICLR2019][snip-paper] | §4,connection sensitivity и обучение после pruning; не доказательство PT/no-FT structural качества. Устойчивый идентификатор: arXiv:1810.02340 |
| [Wang,Zhang,Grosse. GraSP,ICLR2020][grasp-paper] | §4,Alg.1–2:fixed-vector Hg,signed deletion scores; не доказательство PT usefulness. Устойчивый идентификатор: arXiv:2002.07376 |
| [Kipf,Welling. GCN,ICLR2017][gcn-paper] | Node-axis graph operator,Table1 F/C; исходный public split отличается от custom60/20/20 |
| [Hamilton,Ying,Leskovec. GraphSAGE,NeurIPS2017][sage-paper] | Оригинальный framework; V2 использует конкретный PyG mean variant, а не всё семейство |
| [Fang et al. DepGraph,CVPR2023][depgraph] | Structural dependency grouping; reference по surgery, не подтверждённая зависимость проекта |
| [He et al. AMC,ECCV2018][amc-paper] | DDPG,layer-wise resource-constrained policy; сравнение как related work, не тождество tabular V2 |
| [Gupta et al. PuRL,AutoML workshop2020][purl-paper] | Stepwise RL pruning/reward; не обоснование коэффициентов0.6/0.4 или STOP V2 |
| [Li et al. Revisiting Random Channel Pruning,CVPR2022][random-paper] | Значение architecture-search protocol; random baseline статьи отличается от одного равномерного V2 draw. DOI10.1109/CVPR52688.2022.00029 |
| [Liu et al. MetaPruning,ICCV2019][meta-paper] | Meta-network+evolutionary architecture search; не эквивалент fixed-ranking enumeration |
| [Shen et al. HALP,NeurIPS2022][halp-paper] | Latency lookup/resource allocation; parameter count V2 не hardware latency |
| [PyG2.6.1 SAGEConv source][pyg-sage], [Planetoid][pyg-data] | Фактические branches,bias/defaults и split API; версии POLUS нужно проверять отдельно |
| [PyTorch reproducibility][torch-rng], [pruning tutorial][torch-prune], [Torch-Pruning][torch-pruning] | Seed не гарантирует cross-platform equality; remove pruning reparametrization не compaction; dependency tooling — reference only |

## Приложение A. Воспроизводимый целочисленный расчёт без обучения

Ниже самостоятельный Python3-код; не импортирует репозиторий, PyTorch/PyG, не читает и не меняет данные. Именно такой rational enumeration использован для таблицы C01. Никакая trained-model evaluation не выполнялась.

    from fractions import Fraction
    for name, F, C in [
        ("Cora", 1433, 7), ("CiteSeer", 3703, 6), ("PubMed", 500, 3)
    ]:
        for model, m in [("GCN", 1), ("GraphSAGE", 2)]:
            def P(a, b):
                return m * (F*a + a*b + b*C) + a + b + C
            p0 = P(32, 32)
            for target in [Fraction(3, 10), Fraction(1, 2)]:
                pairs = [
                    (Fraction(p0-P(a,b), p0), a, b, P(a,b))
                    for a in range(4,33) for b in range(4,33)
                ]
                band = [v for v in pairs
                        if abs(v[0]-target) <= Fraction(1,100)]
                gap = min(abs(v[0]-target) for v in pairs)
                nearest = [v for v in pairs if abs(v[0]-target) == gap]
                print(name, model, p0, target, len(band), nearest)

Если фактические conv settings, F,C или bias изменятся, таблица пересчитывается до запуска и получает новый protocol identity. Недостижимую точку нельзя «исправить» изменением старого CSV.

## Приложение B. Закреплённые ссылки

[spec]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md
[spec-target]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L64-L178
[spec-static]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L90-L178
[spec-grasp]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L124-L160
[spec-reward]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L187-L277
[spec-open]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L380-L394
[spec-claims]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L334-L346
[spec-platform]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/LOCAL_STRUCTURED_V2_SPEC.md#L279-L320
[mask-prefix]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/MASK_SCORING_AND_EQUIVALENCE.md#L18-L49
[mask-scores]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/MASK_SCORING_AND_EQUIVALENCE.md#L51-L57
[mask-tests]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/methodology/MASK_SCORING_AND_EQUIVALENCE.md#L59-L69
[backlog]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/research/backlog/IDEAS.md#L5-L40
[models]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/models/graphsage.py#L35-L49
[layer-score]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L66-L110
[prunable-set]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L28-L31
[grasp-code]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L412-L484
[apply-static]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L677-L716
[terminal]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L278-L314
[q-config]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L1366-L1384
[q-buckets]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L1386-L1396
[action-filter]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L139-L240
[q-stop]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L1608-L1718
[greedy-old]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L1202-L1300
[q-update]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/rl/q_learning.py#L35-L54
[q-state]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/rl/state.py#L18-L51
[env-score]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/rl/environment.py#L324-L335
[env-stop]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/rl/environment.py#L82-L277
[checkpoint-path]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/workflow.py#L61-L145
[actual-count]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/workflow.py#L119-L134
[split-load]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/workflow.py#L470-L489
[signature]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/training/workflow.py#L358-L385
[typed-config]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/config/schema.py#L74-L104
[config-loader]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/config/loader.py#L36-L58
[old-runner]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/scripts/run_kaggle_pilot.py#L24-L80
[aggregation]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pipelines/suite.py#L86-L166
[surgery]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/surgery/structural.py#L172-L332
[surgery-tests]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/tests/test_surgery.py#L78-L184
[grasp-test]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/tests/test_pruning_methods.py#L197-L212
[package]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/pyproject.toml#L10-L27
[seed-code]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/utils/seed.py#L8-L25
[timing]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/evaluation/benchmark.py#L14-L85
[local-env-extra]: https://github.com/r1zons/prunable_gnn_codex/blob/d4222637486c7b269589c2e8bbded8bad4e042fb/research/experiments/local_cpu_2026/audit/environment.json
[polus-extra]: https://github.com/r1zons/prunable_gnn_codex/blob/d4222637486c7b269589c2e8bbded8bad4e042fb/research/experiments/polus_cpu_2026/docs/SERVER-STATUS.md#L35-L89
[polus-gpu-extra]: https://github.com/r1zons/prunable_gnn_codex/blob/d4222637486c7b269589c2e8bbded8bad4e042fb/research/experiments/polus_cpu_2026/docs/GNN-POLUS.md#L17-L21
[snip-paper]: https://arxiv.org/pdf/1810.02340
[grasp-paper]: https://arxiv.org/pdf/2002.07376
[gcn-paper]: https://arxiv.org/pdf/1609.02907
[sage-paper]: https://arxiv.org/abs/1706.02216
[depgraph]: https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
[amc-paper]: https://arxiv.org/pdf/1802.03494
[purl-paper]: https://arxiv.org/pdf/2007.04756
[random-paper]: https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html
[meta-paper]: https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_MetaPruning_Meta_Learning_for_Automatic_Neural_Network_Channel_Pruning_ICCV_2019_paper.html
[halp-paper]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/5434be94e82c54327bb9dcaf7fca52b6-Abstract-Conference.html
[pyg-sage]: https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/conv/sage_conv.html
[pyg-data]: https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.Planetoid.html
[torch-rng]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[torch-prune]: https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
[torch-pruning]: https://github.com/VainF/Torch-Pruning

[gcn-model]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/models/gcn.py#L35-L49
[q-deploy-stop]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L1859-L1883
[static-timer]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/pruning/methods.py#L902-L964
[data-splits]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/data/splits.py#L33-L58
[data-loader]: https://github.com/r1zons/prunable_gnn_codex/blob/9076b0630a95fbbfab64422ec798189f29e589df/src/gnn_pruning/data/factory.py#L70-L77

