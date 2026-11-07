# Distributed machine learning labs

Три майлстоуна: базовая линия CatBoost с CTR, распределённое обучение (штатный режим CatBoost и CatBoost for Apache Spark), измерение задержек инференса полученных моделей.

Отчёты по майлстоунам (PDF): [M1](reports/Distributed_ML_M1_baseline_2026-11-07.pdf), [M2](reports/Distributed_ML_M2_distributed_training_2026-11-07.pdf), [M3](reports/Distributed_ML_M3_inference_2026-11-07.pdf)

## Майлстоуны

| Майлстоун | Что показано | Где лежит |
|---|---|---|
| **M1. Базовая линия и пробный запуск** | качество, CTR-структура, время fit и predict на Amazon, KDD, Criteo; совпадение предсказаний исходной и оптимизированной версии; штатный distributed-запуск с 2 worker-процессами: без CTR успешен, с CTR падает (`TProjection`) | `results/stage1/`, `results/stage1/predict_singlehost/`, логи падений в `evidence/stock-distributed-logs.tar.gz` |
| **M2. Распределённое обучение** | 1/2/4 worker-процесса: штатный режим на моделях без CTR (время, качество, накладные расходы старта); Spark с CTR на Amazon (1/2/4) и Criteo (2) | `results/stage2_stock/summary.tsv`, `figures/milestones/stock_distributed_workers.png`, `results/stage2/`, `figures/milestones/amazon_ctr_training_workers.png` |
| **M3. Инференс** | CTR-структура пяти распределённых моделей; парное сравнение исходного и оптимизированного `predict` (p50/p90/p95/p99), совпадение предсказаний на полном holdout | `results/stage3/ctr_structure/structure.json`, `results/stage3/distributed_ctr_quantiles_20261107/`, `figures/milestones/ctr_*` |

**Главный результат M3.** В восьми CTR-ячейках p95 оптимизированного CPU-инференса ниже исходного в 9 из 9 парных сессий, 95% интервалы не включают ноль. В двух ячейках без CTR (контроль) интервалы включают ноль. Измерение касается `predict` на готовом Pool, а не сетевого сервиса. Бинарники сравнения собраны до отдельного изменения FIFO-вытеснения кеша.

**Главный результат M2 (штатный режим, без CTR).** На этих размерах распределённое обучение не ускоряет обучение относительно одиночного хоста: при 3 потоках на worker-е Criteo ускоряется с 20,6 с (1 worker) до 14,6 с (4 worker-а), а Amazon и KDD при росте числа worker-ов замедляются. Качество (Logloss, AUC) между режимами практически не меняется.

## Структура

```
data/manifest.json        происхождение и SHA-256 датасетов (сами данные не в git)
scripts/                  подготовка данных, обучение, измерения, сводки и графики
results/stage1/           модели и структура базовой линии, парные predict-записи
results/stage2/           Spark-обучение с CTR (Amazon 1/2/4, Criteo 2)
results/stage2_stock/     штатный распределённый режим CatBoost: records.json, summary.tsv
results/stage3/           сводки измерений инференса и CTR-структура моделей
figures/milestones/       графики для отчёта
reports/                  PDF отчётов по майлстоунам 1–3
evidence/                 сырые сессии этапа 3; логи штатных запусков и записи этапа 1 (архивы)
```

## Соглашение о путях

Все пути в результатах (`results/*.json`, `inputs.json`) записаны относительно корня `labs/distributed-ml/`. Внешние артефакты (`SRW2iter/...`, временные каталоги запусков `tmp/...`) указаны относительно родительского каталога проекта; их нет в репозитории.

## Окружение

- Java 17, Python 3 с зависимостями из `requirements.txt` (CatBoost 1.2.10, PySpark 3.5.0). PySpark 3.5.1 с CatBoost Spark 1.2.10 не работает (`NoSuchMethodError` в `CSVOptions`), поэтому версия зафиксирована.
- Датасеты и крупные модели (`*.cbm`, `*.dsv`, `*.tsv`) в git не хранятся; их SHA-256 записаны в `data/manifest.json`, `inputs.json` и `results/stage3/ctr_structure/structure.json`.
- Для локального Spark и штатного `run-worker` worker-ы являются отдельными процессами одной машины.

Переменные окружения:

| Переменная | Для чего | По умолчанию |
|---|---|---|
| `CATBOOST_BIN` | бинарник CatBoost для этапов 1 и 2 | `build/Release/catboost/app/catboost` в корне репозитория |
| `SRW2_ROOT` | каталог предыдущего проекта с исходными датасетами | `../SRW2iter` рядом с репозиторием |
| `CTR_SPARK_DATA_ROOT` | данные для Spark; путь должен быть без пробелов | `data/` |
| `CTR_LAB_INPUT_ROOT` | данные, модели и CD для этапа 3 | корень `labs/distributed-ml` |
| `CTR_BENCH_BIN_DIR` | два бинарника измерительного харнесса (`stock`, `production`) | `bin/` |
| `CTR_CRITEO_POOL` | пул Criteo для этапа 3 | `tmp/criteo_x1_holdout_4cat_20261107.dsv` |

## Запуск

Сводка и график штатного распределённого режима (по сохранённым записям, без моделей и данных):

```sh
python3 scripts/summarize_stock_distributed.py
python3 scripts/plot_stock_distributed.py
```

Обучение на Spark:

```sh
export JAVA_HOME=/path/to/jdk-17
export CTR_SPARK_DATA_ROOT=/path/without/spaces/data
python3 scripts/spark_ctr_train.py --workers 2 --out /path/to/new/amazon-2w
python3 scripts/spark_criteo_ctr_train.py --workers 2 --cat-count 4 --out /path/to/new/criteo-2w
```

Пересчёт сохранённой квантильной сводки не требует моделей и датасетов:

```sh
tar -xzf evidence/raw-sessions.tar.gz
python3 scripts/run_distributed_ctr_quantiles.py --summarize-only
python3 scripts/plot_distributed_ctr_quantiles.py
```

Логи штатных распределённых запусков и покомпонентные записи этапа 1 упакованы в `evidence/stock-distributed-logs.tar.gz` и `evidence/stage1-run-records.tar.gz`; `records.json` ссылается на пути внутри этих архивов.

Архив сырых сессий содержит 180 записей запусков (`session_N_{stock,production}.json`) и 180 файлов задержек по 1000 вызовов (`*.latencies.json`).

Новые измерения этапа 3 требуют `CTR_LAB_INPUT_ROOT` и `CTR_BENCH_BIN_DIR`.

## Ограничения

- Каждая конфигурация обучения запущена **один раз**. Общее правило проекта требует три независимых seed для основных сравнений; это не выполнено для M2 и отмечено в отчёте.
- Оценка Criteo с CTR на 1 и 4 workers не выполнена; большой CTR-запуск только на 2 workers.
- Штатный распределённый режим не запускается для моделей с CTR (падение `TProjection` на 2 worker-ах во всех трёх наборах).
- Все измерения сделаны на одной машине (macOS arm64) без контроля фоновой нагрузки. Число потоков растёт вместе с числом worker-ов (3 потока на worker), поэтому сравнение со single-host не является сравнением при равном бюджете потоков.
- Модели без CTR в контроле M3 имеют 200 деревьев против 20 у моделей с CTR; контроль показывает отсутствие CTR-выигрыша, но не полностью совпадает по структуре.
