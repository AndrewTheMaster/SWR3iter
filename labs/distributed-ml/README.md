# Distributed machine learning labs

Три этапа проекта: базовая линия CatBoost с CTR, распределённое обучение через официальный CatBoost for Apache Spark и инференс полученных моделей. Итоговый отчёт с таблицами, графиками, ограничениями и результатами каждого этапа: [distributed-ml-milestones.pdf](reports/distributed-ml-milestones.pdf). Отчёт в Markdown в эту ветку не добавлен.

| Этап | Код | Проверяемые результаты |
|---|---|---|
| 1. Базовая линия | `scripts/prepare_data.py`, `run_training.py`, `structure.py` | `results/stage1/records.json`, структуры и метрики моделей Amazon, KDD и Criteo |
| 2. Обучение CTR | `scripts/spark_ctr_train.py`, `spark_criteo_ctr_train.py` | `results/stage2/`: Amazon с 1/2/4 worker-процессами; Criteo с 2 workers на 10 млн строк |
| 3. Инференс | `scripts/run_distributed_ctr_quantiles.py`, `plot_distributed_ctr_quantiles.py` | `results/stage3/`: полные хеши предсказаний, p50/p90/p95/p99, 95% интервалы; [сырьё](evidence/raw-sessions.tar.gz) |

**Главный результат этапа 3:** у всех восьми CTR-ячеек p95 исходного и оптимизированного CPU-инференса различается в пользу оптимизации (9/9 парных сессий в каждой); у двух ячеек без CTR интервалы включают ноль. Это `predict` на готовом Pool, а не задержка сетевого сервиса. Бинарники сравнения собраны до отдельного изменения FIFO-вытеснения кеша.

## Запуск и проверка

Нужны Java 17, Python с зависимостями из `requirements.txt`, CatBoost Spark 1.2.10, внешние датасеты и два бинарника измерительного харнесса (`stock` и `production`). CatBoost Spark JAR загружается скриптами через Maven. При локальном Spark `local-cluster` worker-ы являются отдельными процессами одной машины. Путь к данным для Spark должен быть без пробелов; укажите его через `CTR_SPARK_DATA_ROOT`.

```sh
export JAVA_HOME=/path/to/jdk-17
export CTR_SPARK_DATA_ROOT=/path/without/spaces/data
python3 scripts/spark_ctr_train.py --workers 2 --out /path/to/new/amazon-2w
python3 scripts/spark_criteo_ctr_train.py --workers 2 --cat-count 4 --out /path/to/new/criteo-2w
```

Пересчёт уже сохранённой квантильной сводки не требует моделей или датасетов:

```sh
tar -xzf evidence/raw-sessions.tar.gz
python3 scripts/run_distributed_ctr_quantiles.py --summarize-only
python3 scripts/plot_distributed_ctr_quantiles.py
```

Для **новых** измерений дополнительно задайте `CTR_LAB_INPUT_ROOT` с данными, моделями и CD, а также `CTR_BENCH_BIN_DIR` с двумя бинарниками. Исходные датасеты и крупные модели в Git не помещены; их SHA-256 записаны в `data/manifest.json` и `results/stage3/distributed_ctr_quantiles_20261107/inputs.json`. Архив сырья содержит 180 рядов по 1000 измеренных вызовов и 180 записей с параметрами сессий. Оценка Criteo на 1/4 workers и нескольких физических машинах остаётся открытой.
