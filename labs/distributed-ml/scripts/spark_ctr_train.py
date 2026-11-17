"""Train a CTR-bearing CatBoost model with separate local Spark workers.

Run from the project root with the Python environment described in README.
The exported JSON is inspected before a run can be called successful.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


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
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers not in (1, 2, 4):
        parser.error("--workers must be 1, 2 or 4")

    root = Path(__file__).resolve().parents[1]
    # CatBoost Spark's DSV loader cannot open a file: URI with spaces in the
    # local path. CTR_SPARK_DATA_ROOT may point to a symlink without spaces.
    data_root = Path(os.environ.get("CTR_SPARK_DATA_ROOT", str(root / "data")))
    train = data_root / "amazon/train.tsv"
    holdout = data_root / "amazon/holdout.tsv"
    cd = data_root / "amazon/train.cd"
    args.out.mkdir(parents=True, exist_ok=False)
    spark = (SparkSession.builder
             .appName("amazon-ctr-workers-{}".format(args.workers))
             .master("local-cluster[{},1,1024]".format(args.workers))
             .config("spark.jars.packages", PACKAGE)
             .config("spark.executor.instances", str(args.workers))
             .config("spark.executor.cores", "1")
             .config("spark.executor.memory", "1024m")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    try:
        import catboost_spark

        # Pool.load's DSV reader currently rejects non-float features. Spark
        # StringIndexer and VectorAssembler preserve nominal metadata, which
        # CatBoost Spark uses to identify categorical features.
        raw = spark.read.csv(str(train), sep="\t", header=False)
        cat_cols = ["_c{}".format(i) for i in range(1, 10)]
        indexed_cols = ["cat{}_index".format(i) for i in range(1, 10)]
        raw = raw.selectExpr("cast(_c0 as double) as label", *cat_cols)
        indexer = StringIndexer(
            inputCols=cat_cols, outputCols=indexed_cols, handleInvalid="keep"
        ).fit(raw)
        labels = [list(values) for values in indexer.labelsArray]
        mappings = [{value: i for i, value in enumerate(values)} for values in labels]
        (args.out / "category_index.json").write_text(
            json.dumps({"labels": labels, "unknown_index": [len(x) for x in labels]}) + "\n"
        )
        indexed_holdout = args.out / "holdout_indexed.tsv"
        holdout_rows = 0
        with holdout.open(newline="") as source, indexed_holdout.open("w", newline="") as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            for row in reader:
                if len(row) != 10:
                    raise ValueError("expected label and nine categorical fields")
                writer.writerow([row[0]] + [
                    str(mapping.get(value, len(labels[i])))
                    for i, (mapping, value) in enumerate(zip(mappings, row[1:]))
                ])
                holdout_rows += 1
        (args.out / "holdout_indexed.cd").write_text(
            "0\tTarget\n" + "".join("{}\tCateg\n".format(i) for i in range(1, 10))
        )
        assembler = VectorAssembler(inputCols=indexed_cols, outputCol="features")
        data = assembler.transform(indexer.transform(raw)).select(
            "label", "features"
        ).repartition(args.workers)
        nominal = data.schema["features"].metadata["ml_attr"]["attrs"]["nominal"]
        if len(nominal) != 9:
            raise RuntimeError("categorical feature metadata was lost")
        rows = data.count()
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
        info = json.loads(model_json.read_text())
        ctrs = info.get("features_info", {}).get("ctrs", [])
        executor_count = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1
        result = {
            "workers_requested": args.workers,
            "spark_master": spark.sparkContext.master,
            "executor_memory_status": str(spark.sparkContext._jsc.sc().getExecutorMemoryStatus().keySet()),
            "executor_count_observed": executor_count,
            "train_rows": rows,
            "nominal_features": len(nominal),
            "iterations": args.iterations,
            "fit_seconds": elapsed,
            "model_sha256": sha256(native),
            "ctr_count": len(ctrs),
            "native_model": str(native),
            "train_sha256": sha256(train),
            "cd_sha256": sha256(cd),
            "holdout_rows": holdout_rows,
            "indexed_holdout_sha256": sha256(indexed_holdout),
            "category_index_sha256": sha256(args.out / "category_index.json"),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not ctrs:
            raise RuntimeError("trained model has no CTR features")
        if executor_count != args.workers:
            raise RuntimeError("observed executor count differs from requested workers")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
