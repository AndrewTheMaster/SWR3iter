# Distributed machine learning labs

Оптимизация инференса CatBoost с CTR-признаками на моделях, обученных распределённо. Три майлстоуна.

**Что распределено, а что нет.** Распределено только **обучение** (штатный режим CatBoost `run-worker` и CatBoost for Apache Spark). Измеряемое **предсказание (`predict`) выполняется локально одним процессом на CPU** с двумя потоками; worker-процессы в инференсе не участвуют.

## Отчёты

- [M1. Базовая линия и пробный запуск](reports/Distributed_ML_M1_baseline_2026-11-07.pdf)
- [M2. Распределённое обучение](reports/Distributed_ML_M2_distributed_training_2026-11-07.pdf)
- [M3. Инференс моделей распределённого обучения](reports/Distributed_ML_M3_inference_2026-11-07.pdf)
- НИР (CPU-оптимизация CTR, CPU-ветка): [NIR_CatBoost_CTR.pdf](https://github.com/AndrewTheMaster/SWR3iter/blob/perf/cpu-ctr-inference/research/NIR_CatBoost_CTR.pdf)

## Майлстоуны

| Майлстоун | Что показано | Где лежит |
|---|---|---|
| **M1. Базовая линия** | качество, CTR-структура, время fit и predict на Amazon, KDD, Criteo; совпадение предсказаний исходной и оптимизированной версии; штатный распределённый запуск с 2 worker-ами: без CTR успешен, с CTR падает (`TProjection`) | `results/stage1/`, `results/stage1/predict_singlehost/`, ошибка падения в `results/stage2_stock/records.json` (поле `failure_tail`) |
| **M2. Распределённое обучение** | 1/2/4 worker-а при штатном режиме на моделях без CTR (время, качество, старт процесса); Spark с CTR на Amazon (1/2/4) и Criteo (2) | `results/stage2_stock/summary.tsv`, `results/stage2/`, `figures/milestones/` |
| **M3. Инференс** | CTR-структура пяти моделей, обученных распределённо; парное сравнение локального `predict` исходного и CTR-оптимизированного CatBoost (p50/p90/p95/p99); совпадение предсказаний на полном holdout | `results/stage3/ctr_structure/structure.json`, `results/stage3/distributed_ctr_quantiles_20261107/`, `figures/milestones/ctr_*` |

**Главный результат M3.** В восьми CTR-ячейках p95 локального `predict` оптимизированной версии ниже исходного в 9 из 9 парных сессий, 95% интервалы не включают ноль. В двух ячейках без CTR (контроль) интервалы включают ноль. Измерение касается готового Pool, а не сетевого сервиса.

**Главный результат M2 (штатный режим, без CTR).** На этих размерах распределённое обучение не быстрее одиночного хоста: Criteo ускоряется с 20,6 с (1 worker) до 14,6 с (4 worker-а), Amazon и KDD при росте числа worker-ов замедляются. Качество (Logloss, AUC) практически не меняется.

## Структура

```
data/manifest.json              происхождение и SHA-256 датасетов (сами данные не в git)
scripts/                        подготовка данных, обучение, измерения, сводки и графики
reports/                        PDF отчётов M1–M3
evidence/results.tar.gz         результаты (JSON, TSV, CTR-структура) и графики; распаковать перед запуском скриптов
evidence/raw-sessions.tar.gz    сырые сессии измерений M3
evidence/raw-manifest.json      перечень архива сырых сессий
```

Логи запусков в репозиторий не входят. Ошибки штатных распределённых запусков сохранены в `results/stage2_stock/records.json` (поле `failure_tail`).

## Соглашение о путях

Пути в результатах (`results/*.json`, `inputs.json`) записаны относительно корня `labs/distributed-ml/`. Внешние артефакты (`SRW2iter/...`, временные каталоги `tmp/...`) указаны относительно родительского каталога проекта; в репозитории их нет.

## Окружение

- Java 17, Python 3 с зависимостями из `requirements.txt` (CatBoost 1.2.10, PySpark 3.5.0). PySpark 3.5.1 с CatBoost Spark 1.2.10 не работает (`NoSuchMethodError` в `CSVOptions`), поэтому версия зафиксирована.
- Датасеты и крупные модели (`*.cbm`, `*.dsv`, `*.tsv`) в git не хранятся; их SHA-256 записаны в `data/manifest.json`, `inputs.json` и `results/stage3/ctr_structure/structure.json`.
- Worker-ы штатного режима и Spark — отдельные процессы одной машины.

Переменные окружения:

| Переменная | Для чего | По умолчанию |
|---|---|---|
| `CATBOOST_BIN` | бинарник CatBoost для этапов 1 и 2 | `build/Release/catboost/app/catboost` в корне репозитория |
| `SRW2_ROOT` | каталог предыдущего проекта с исходными датасетами | `../SRW2iter` рядом с репозиторием |
| `CTR_SPARK_DATA_ROOT` | данные для Spark; путь без пробелов | `data/` |
| `CTR_LAB_INPUT_ROOT` | данные, модели и CD для новых измерений M3 | корень `labs/distributed-ml` |
| `CTR_BENCH_BIN_DIR` | два бинарника измерительного харнесса (`stock`, `production`) | `bin/` |
| `CTR_CRITEO_POOL` | пул Criteo для M3 | `tmp/criteo_x1_holdout_4cat_20261107.dsv` |

## Запуск

Сначала распаковать результаты:

```sh
tar -xzf evidence/results.tar.gz
```

Сводка и график штатного распределённого режима (без моделей и данных):

```sh
python3 scripts/summarize_stock_distributed.py
python3 scripts/plot_stock_distributed.py
```

Пересчёт квантильной сводки M3 без моделей и датасетов:

```sh
tar -xzf evidence/raw-sessions.tar.gz
python3 scripts/run_distributed_ctr_quantiles.py --summarize-only
python3 scripts/plot_distributed_ctr_quantiles.py
```

Архив сырых сессий содержит 180 записей запусков (`session_N_{stock,production}.json`) и 180 файлов задержек по 1000 вызовов (`*.latencies.json`).

Обучение на Spark:

```sh
export JAVA_HOME=/path/to/jdk-17
export CTR_SPARK_DATA_ROOT=/path/without/spaces/data
python3 scripts/spark_ctr_train.py --workers 2 --out /path/to/new/amazon-2w
python3 scripts/spark_criteo_ctr_train.py --workers 2 --cat-count 4 --out /path/to/new/criteo-2w
```

Новые измерения M3 требуют `CTR_LAB_INPUT_ROOT` и `CTR_BENCH_BIN_DIR`.

## Ограничения

- Каждая конфигурация обучения запущена **один раз**. Общее правило проекта требует три независимых seed для основных сравнений; это не выполнено для M2 и отмечено в отчёте.
- Criteo с CTR обучен только на 2 worker-ах; масштабирование на 1 и 4 не измерено.
- Штатный распределённый режим не запускается для моделей с CTR (`TProjection` на 2 worker-ах во всех трёх наборах).
- Все измерения сделаны на одной машине (macOS arm64) без контроля фоновой нагрузки. Число потоков растёт вместе с числом worker-ов (3 потока на worker), поэтому сравнение с одиночным хостом не при равном бюджете потоков.
- Модели без CTR в контроле M3 имеют 200 деревьев против 20 у моделей с CTR; контроль показывает отсутствие CTR-выигрыша, но не полностью совпадает по структуре.
