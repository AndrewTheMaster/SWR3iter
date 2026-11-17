"""Train CatBoost Spark with CTRs on the saved 10M-row Criteo_x1 view.

The source categories are already integer encoded. Their integer values are
marked nominal in Spark vector metadata, so CatBoost hashes their decimal
string forms just as the native Categ DSV reader does. No StringIndexer pass
over the ten million rows is needed.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import ijson
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel


PACKAGE = "ai.catboost:catboost-spark_3.5_2.12:1.2.10"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--executor-mb", type=int, default=4096)
    parser.add_argument("--driver-mb", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cat-count", type=int, default=26)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers not in (1, 2, 4):
        parser.error("--workers must be 1, 2 or 4")
    if args.executor_mb < 2048:
        parser.error("--executor-mb must be at least 2048")
    if args.driver_mb < 2048:
        parser.error("--driver-mb must be at least 2048")
    if not 1 <= args.cat_count <= 26:
        parser.error("--cat-count must be from 1 to 26")
    args.out.mkdir(parents=True, exist_ok=False)

    root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("CTR_SPARK_DATA_ROOT", str(root / "data")))
    train = args.train_path or data_root / "criteo_x1/train_ctr.dsv"
    cd = data_root / "criteo_x1/full.cd"
    spark = (SparkSession.builder
             .appName("criteo-x1-ctr-workers-{}".format(args.workers))
             .master("local-cluster[{},1,{}]".format(args.workers, args.executor_mb))
             .config("spark.jars.packages", PACKAGE)
             .config("spark.executor.instances", str(args.workers))
             .config("spark.executor.cores", "1")
             .config("spark.executor.memory", "{}m".format(args.executor_mb))
             .config("spark.driver.memory", "{}m".format(args.driver_mb))
             .config("spark.driver.maxResultSize", "4g")
             .config("spark.rpc.message.maxSize", "512")
             .config("spark.sql.shuffle.partitions", str(args.workers))
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    try:
        import catboost_spark

        raw = spark.read.csv(str(train), sep="\t", header=False)
        if args.limit_rows:
            raw = raw.limit(args.limit_rows)
        num_cols = ["n{}".format(i) for i in range(1, 14)]
        cat_cols = ["c{}".format(i) for i in range(1, args.cat_count + 1)]
        columns = [F.col("_c0").cast("double").alias("label")]
        columns += [F.coalesce(F.col("_c{}".format(i)).cast("double"), F.lit(0.0)).alias(name)
                    for i, name in enumerate(num_cols, start=1)]
        columns += [F.coalesce(F.col("_c{}".format(i)).cast("double"), F.lit(0.0)).alias(name)
                    for i, name in enumerate(cat_cols, start=14)]
        decoded = raw.select(*columns)
        stats = decoded.agg(
            F.count("*").alias("rows"),
            F.max(F.greatest(*[F.col(c) for c in cat_cols])).alias("max_category")
        ).first()
        rows = int(stats["rows"])
        max_cat = int(stats["max_category"])
        if rows == 0 or max_cat < 0:
            raise RuntimeError("empty data or invalid category range")
        assembler = VectorAssembler(
            inputCols=num_cols + cat_cols, outputCol="features", handleInvalid="keep"
        )
        metadata = {"ml_attr": {
            "attrs": {
                "numeric": [{"idx": i, "name": name} for i, name in enumerate(num_cols)],
                "nominal": [{"idx": i + 13, "name": name, "num_vals": max_cat + 1}
                            for i, name in enumerate(cat_cols)],
            },
            "num_attrs": 13 + args.cat_count,
        }}
        selected_cd = args.out / "selected.cd"
        selected_cd.write_text(
            "0\tTarget\n"
            + "".join("{}\tNum\n".format(i) for i in range(1, 14))
            + "".join("{}\tCateg\n".format(i) for i in range(14, 14 + args.cat_count))
        )
        assembled = assembler.transform(decoded)
        data = assembled.select(
            "label", F.col("features").alias("features", metadata=metadata)
        ).repartition(args.workers)
        nominal = data.schema["features"].metadata["ml_attr"]["attrs"]["nominal"]
        if len(nominal) != args.cat_count:
            raise RuntimeError("categorical metadata was lost")
        # CatBoost runs multiple Spark jobs over this DataFrame. Materialize
        # once so each job does not reparse the 2.5 GB source DSV.
        data.persist(StorageLevel.DISK_ONLY)
        if data.count() != rows:
            raise RuntimeError("materialized row count changed")
        estimator = catboost_spark.CatBoostClassifier(
            iterations=args.iterations,
            depth=4,
            oneHotMaxSize=2,
            sparkPartitionCount=args.workers,
            threadCount=1,
            randomSeed=20261107,
            trainDir=str(args.out / "train-dir"),
        )
        started = time.monotonic()
        model = estimator.fit(data)
        elapsed = time.monotonic() - started
        native = args.out / "model.cbm"
        model.saveNativeModel(str(native))
        model_json = args.out / "model.json"
        model.saveNativeModel(str(model_json), catboost_spark.EModelType.Json)
        with model_json.open("rb") as stream:
            ctr_count = sum(1 for _ in ijson.items(stream, "features_info.ctrs.item"))
        executor_count = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1
        result = {
            "dataset": "Criteo_x1",
            "workers_requested": args.workers,
            "driver_memory_mb": args.driver_mb,
            "executor_memory_mb": args.executor_mb,
            "executor_count_observed": executor_count,
            "spark_master": spark.sparkContext.master,
            "train_rows": rows,
            "source_path": str(train),
            "limit_rows": args.limit_rows,
            "numeric_features": 13,
            "nominal_features": len(nominal),
            "selected_categorical_columns": ["C{}".format(i) for i in range(1, args.cat_count + 1)],
            "max_category_id": max_cat,
            "iterations": args.iterations,
            "fit_seconds": elapsed,
            "ctr_count": ctr_count,
            "model_bytes": native.stat().st_size,
            "model_sha256": sha256(native),
            "train_sha256": sha256(train),
            "cd_sha256": sha256(cd),
            "selected_cd_sha256": sha256(selected_cd),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if ctr_count == 0 or executor_count != args.workers:
            raise RuntimeError("CTR model or executor count verification failed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
