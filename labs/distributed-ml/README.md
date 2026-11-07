# Distributed machine learning labs

Три этапа: базовая линия CatBoost с CTR, распределённое обучение через CatBoost for Apache Spark, измерение задержек инференса полученных моделей.

Итоговый отчёт (PDF): [reports/distributed-ml-milestones.pdf](reports/distributed-ml-milestones.pdf)

| Этап | Код | Результаты |
|---|---|---|
| 1. Базовая линия | `scripts/prepare_data.py`, `run_training.py`, `structure.py`, `prepare_numeric.py`, `prepare_criteo_x1.py` | `results/stage1/` |
| 2. Обучение CTR | `scripts/spark_ctr_train.py`, `spark_criteo_ctr_train.py` | `results/stage2/`: Amazon с 1/2/4 worker-процессами; Criteo с 2 workers на 10 млн строк |
| 3. Инференс | `scripts/run_distributed_ctr_quantiles.py`, `plot_distributed_ctr_quantiles.py` | `results/stage3/`: хеши предсказаний, p50/p90/p95/p99, 95% интервалы; сырые сессии в [evidence/raw-sessions.tar.gz](evidence/raw-sessions.tar.gz) |

**Главный результат этапа 3.** В восьми CTR-ячейках p95 оптимизированного CPU-инференса ниже исходного в 9 из 9 парных сессий, 95% интервалы не включают ноль. В двух ячейках без CTR (контроль) интервалы включают ноль. Измерение касается `predict` на готовом Pool, а не сетевого сервиса. Бинарники сравнения собраны до отдельного изменения FIFO-вытеснения кеша.

## Структура

```
data/manifest.json        происхождение и SHA-256 датасетов (сами данные не в git)
scripts/                  подготовка данных, обучение, измерения, графики
results/stage1..3/        метрики, структуры моделей, сводки измерений
figures/milestones/       графики для отчёта
reports/                  итоговый PDF
evidence/                 манифест и архив сырых сессий этапа 3
```

## Соглашение о путях

Все пути в результатах (`results/*.json`, `inputs.json`) записаны относительно корня `labs/distributed-ml/`. Внешние артефакты (`SRW2iter/...`, временные каталоги запусков `tmp/...`) указаны относительно родительского каталога проекта; их нет в репозитории.

## Окружение

- Java 17, Python 3 с зависимостями из `requirements.txt` (CatBoost 1.2.10, PySpark 3.5.0)
- Датасеты и крупные модели (`*.cbm`, `*.dsv`, `*.tsv`) в git не хранятся; их SHA-256 записаны в `data/manifest.json` и `inputs.json`
- Для локального Spark `local-cluster` worker-ы являются отдельными процессами одной машины

Переменные окружения:

| Переменная | Для чего | По умолчанию |
|---|---|---|
| `CATBOOST_BIN` | бинарник CatBoost для этапа 1 и 2 | `build/Release/catboost/app/catboost` в корне репозитория |
| `SRW2_ROOT` | каталог предыдущего проекта с исходными датасетами | `../SRW2iter` рядом с репозиторием |
| `CTR_SPARK_DATA_ROOT` | данные для Spark; путь должен быть без пробелов | `data/` |
| `CTR_LAB_INPUT_ROOT` | данные, модели и CD для этапа 3 | корень `labs/distributed-ml` |
| `CTR_BENCH_BIN_DIR` | два бинарника измерительного харнесса (`stock`, `production`) | `bin/` |
| `CTR_CRITEO_POOL` | пул Criteo для этапа 3 | `tmp/criteo_x1_holdout_4cat_20261107.dsv` |

## Запуск

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

Архив содержит 180 записей запусков (`session_N_{stock,production}.json`) и 180 файлов задержек по 1000 вызовов (`*.latencies.json`).

Новые измерения этапа 3 требуют `CTR_LAB_INPUT_ROOT` и `CTR_BENCH_BIN_DIR`.

## Ограничения

- Оценка Criteo на 1 и 4 workers и на нескольких физических машинах не выполнена.
- Измерения выполнены на одной машине (macOS arm64) без контроля фоновой нагрузки.
- Время обучения сделано по одному запуску на каждое число workers и носит описательный характер.
