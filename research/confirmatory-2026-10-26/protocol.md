# Confirmatory three-dataset CatBoost CTR inference protocol

**Frozen at:** 2026-10-26T13:41:26+03:00  
**Experiment directory:** `results/local/confirmatory-3-datasets-20261026-1345/`  
**Status:** written before downloading data, training models, equivalence checks or timing.

## 1. Question and scope

Does the minimal CatBoost CPU CTR inference optimization (A = pristine v1.2.10,
C = `reuse-ctr-buckets` + `ctr-prep-cache`) change the p95 ready-Pool predict
latency on three independently sourced datasets? This is a confirmatory test of
the predeclared direction (C faster). It is not a search for a dataset or a
configuration that looks favourable.

Data acquisition, preparation and model training are the only work before the
timing test. No dataset, feature set, model parameter or sample count may be
changed after a latency result is observed.

## 2. Fixed sources and retention

Every original download is kept under `data/confirmatory/raw/` with SHA-256 and
MD5 where published. Derived train/holdout files are kept under this experiment
directory. Nothing is deleted or overwritten. Cumulative downloads are bounded
by the 50 GB authorization; the expected total is under 1 GB.

| Dataset | Source page | Exact files |
|---|---|---|
| YOOCHOOSE | https://zenodo.org/records/7412233 | https://zenodo.org/records/7412233/files/yoochoose-clicks.dat.zip?download=1<br>https://zenodo.org/records/7412233/files/yoochoose-buys.dat.zip?download=1 |
| NYC TLC | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet<br>https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet<br>https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-03.parquet |
| UCI Bank | https://archive.ics.uci.edu/dataset/222/bank+marketing | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |

## 3. Dataset construction and split

### 3.1 YOOCHOOSE RecSys Challenge 2015

One row per session from `yoochoose-clicks.dat`. The first click is the row with
the minimum timestamp in the session; ties use the first occurrence in the file.
Features are only what is knowable before the outcome: first-click item ID
(categorical), first-click category (categorical), first-click UTC hour,
weekday, day of month and month. **SessionID is never a feature.** No later click,
no click count, no session duration and no buy information enters the features.
The label is 1 if the SessionID occurs anywhere in `yoochoose-buys.dat`, else 0.
The temporal split sorts sessions by first-click timestamp and puts the first
80% in train and the last 20% in holdout. The cutoff timestamp and row hashes
are recorded before training. `yoochoose-test.dat` is not downloaded or used.

### 3.2 NYC TLC yellow taxi 2025

January and February 2025 parquet files form train; March 2025 is the holdout.
Rows are restricted to `payment_type == 1` (credit card) so recorded tips are
meaningful, and to `fare_amount > 0`. The target is
`tip_amount / fare_amount >= 0.20`. Target leakage is excluded: `tip_amount`,
`total_amount`, `payment_type` and every unlisted column are not model features.
Features are vendor ID, rate code, pickup and dropoff zone IDs (categorical),
passenger count, trip distance, fare amount, pickup hour, weekday and day of
month. Missing numeric values stay missing; missing rate code becomes `NA`.

### 3.3 UCI Bank Marketing

`bank-full.csv` from the official archive, semicolon-separated with its header.
The target is `y`. **`duration` is excluded** because it is only known after the
call. The original row order is preserved and the last 20% is the holdout, the
first 80% is train. Features are age, job, marital, education, default, balance,
housing, loan, contact, day, month, campaign, pdays, previous, poutcome and
day of week; categorical columns are job, marital, education, default, housing,
loan, contact, month, poutcome and day of week.

### 3.4 No-CTR negative controls

For each dataset a second model is trained on the same train rows and predicts
the same holdout rows with all features numeric, so it has zero CTR by
construction. These controls are reported separately and are never included in
the Holm family. They measure the cost of the checks, not a CTR saving.

## 4. Fixed model matrix

| Model | Role | Dataset | Features |
|---|---|---|---|
| `yoochoose_ctr` | CTR (predeclared) | YOOCHOOSE RecSys 2015 | `first_item_id`:Categ, `first_category`:Categ, `first_hour`:Num, `first_weekday`:Num, `first_day_of_month`:Num, `first_month`:Num |
| `yoochoose_noctr` | no-CTR control | YOOCHOOSE RecSys 2015 | `first_item_id`:Num, `first_category`:Num, `first_hour`:Num, `first_weekday`:Num, `first_day_of_month`:Num, `first_month`:Num |
| `nyc_tlc_ctr` | CTR (predeclared) | NYC TLC yellow taxi 2025 | `vendor_id`:Categ, `ratecode_id`:Categ, `pu_location_id`:Categ, `do_location_id`:Categ, `passenger_count`:Num, `trip_distance`:Num, `fare_amount`:Num, `pickup_hour`:Num, `pickup_weekday`:Num, `pickup_day_of_month`:Num |
| `nyc_tlc_noctr` | no-CTR control | NYC TLC yellow taxi 2025 | `vendor_id`:Num, `ratecode_id`:Num, `pu_location_id`:Num, `do_location_id`:Num, `passenger_count`:Num, `trip_distance`:Num, `fare_amount`:Num, `pickup_hour`:Num, `pickup_weekday`:Num, `pickup_day_of_month`:Num |
| `bank_marketing_ctr` | CTR (predeclared) | UCI Bank Marketing | `age`:Num, `job`:Categ, `marital`:Categ, `education`:Categ, `default`:Categ, `balance`:Num, `housing`:Categ, `loan`:Categ, `contact`:Categ, `day`:Num, `month`:Categ, `campaign`:Num, `pdays`:Num, `previous`:Num, `poutcome`:Categ, `day_of_week`:Categ |
| `bank_marketing_noctr` | no-CTR control | UCI Bank Marketing | `age`:Num, `balance`:Num, `day`:Num, `campaign`:Num, `pdays`:Num, `previous`:Num |

The three CTR models (`yoochoose_ctr`, `nyc_tlc_ctr`, `bank_marketing_ctr`)
form the predeclared family for the primary endpoint. The controls are
`yoochoose_noctr`, `nyc_tlc_noctr`, `bank_marketing_noctr`.

## 5. Fixed training configuration

- CatBoost 1.2.10 CPU, `loss_function=Logloss`, `iterations=200`,
  `depth=6`, `learning_rate=0.1`, `random_seed=20260928`.
- `one_hot_max_size=2`, `max_ctr_complexity=2`
  for every CTR model. No per-dataset tuning.
- `thread_count=12` for training only; inference timing always
  uses `threads=2`.
- No early stopping, no class weighting, no resampling, no parameter search.
- AUC and Logloss are computed on the holdout before timing and reported, but
  they are not used to select a model or to change parameters.

### 5.1 Fixed zero-CTR correction

After training, the CTR count of each CTR model is read from its JSON model.
If a CTR model has zero CTR features, that fact is recorded. In that case only,
one retraining is allowed with `one_hot_max_size=0` and every other parameter
unchanged, so the categorical structure can produce CTR. No second correction
and no correction based on timing is allowed.

## 6. Inference arms and equivalence gate

A and C are the verified binaries built from CatBoost v1.2.10 pinned at
`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`, with the same harness source and
build flags used by the prior percentile experiment. Their SHA-256 values are:

- A stock: `d6dc40c2c6862be48bf9883b5b01efc6bdfe9c727120af78a035ebd9950ca725`
- C production: `088bab0a33611022f11188a52cd2840ee1fa8a062a1f4aabb0d7e21fef75d363`

Both arms use the same single `.cbm` model and the same ready Pool for each
dataset/model. Before any timing, each pair predicts the complete holdout Pool
in one fresh process per arm with `--full-dataset --block-size 4096`. The
whole-vector FNV-1a hash, every chunk hash, the row count and both exit codes
must match bitwise. A mismatch stops the timing stage for that model and is
reported as a failed confirmatory test.

## 7. Timing test

- Batch size: **4096** rows; inference threads: **2**.
- Operation: CPU evaluator `CalcFlatTransposed` on an already built Pool.
  Pool construction and model loading are outside the timed region.
- Warm-up: 50 calls before timing.
- Timed calls: 2000 per fresh process, every call recorded raw.
- Paired fresh-process sessions: 20 per dataset/model.
- Deterministic alternating order: odd sessions run A then C; even sessions run
  C then A.
- The predicted rows are the first 4096 holdout rows in file order. The row
  offset is identical for both arms and was fixed before timing.
- Every process exit code, wall time, stdout/stderr digest, binary/model/Pool/CD
  SHA-256 and raw per-call latency are preserved.
- No cell is added, removed or rerun selectively after seeing a result.

## 8. Statistics and decision rule

For each model and arm, p50/p90/p95/p99 are computed from the 2000 raw calls in
each session. The reported central value is the median of the 20 session
percentiles. The paired percentage effect is
`100 * (C_session / A_session - 1)` for the same session; negative means C is
faster. The reported effect is the median of the 20 paired changes. The 95%
interval is a percentile bootstrap of that median, resampling complete paired
sessions with 20,000 draws and seed 20261026.

**Primary endpoint:** p95 paired percentage effect at batch 4096, 2 threads,
ready Pool, on the three predeclared CTR models. The bootstrap two-sided
p-value is `2 * min(P(draw >= 0), P(draw <= 0))`, clipped to `[1/20000, 1]`.
Holm correction is applied across exactly these three p95 tests. A result is
**confirmed faster** only if the Holm-adjusted p-value is < 0.05 and the
median p95 effect and both CI bounds are below zero. If the CI excludes zero
on the positive side, it is reported as a confirmed slowdown. Otherwise the
result is not confirmed. No direction is selected after seeing the data.

**Secondary:** p50 effect with the same CI and p-value, reported for each model
and corrected separately across the three CTR models. **Exploratory:** p90 and
p99 effects and their intervals; always labelled exploratory. No aggregate
percentage across datasets is computed. The three datasets remain separate.

Negative-control effects are reported as descriptive overhead checks and are
excluded from the Holm family. The count of negative pairs (sessions where
C is slower, effect > 0) is reported for every model and quantile.

## 9. Fixed selection rule

The dataset list, model list, features, split, training parameters, endpoint,
sessions and repetitions are fixed by this protocol before any data download.
No parameter, dataset or sample-count decision may depend on observed speed.
The only permitted post-training model change is the single zero-CTR correction
in section 5.1, applied before timing.

## 10. Omissions recorded rather than filled

- YOOCHOOSE buys are used only as labels; no price or quantity is a feature.
- No session-ID feature and no post-first-click session information.
- NYC TLC rows with payment types other than credit card are excluded because
  their tips are mostly unrecorded; this is a target-availability restriction.
- Bank `duration` is excluded rather than used as a feature.
- No-CTR controls are not expected to be equivalent in quality to CTR models.
- No Linux or x86-64 cross-check; inference is macOS arm64 only.
- Pool construction is not timed.
- Any download failure is recorded with the actual error and the remaining
  fixed sources proceed; no source is silently replaced.

## 11. Protocol changes

Any change after this file is frozen must be appended here with its reason and
timestamp. Artifact hashes are appended in a frozen-artifact section before the
first equivalence or timing process.

## 11.1 Pre-training correction (recorded before any timing)

**Recorded at:** 2026-10-26T13:58:00+03:00

`bank-full.csv` has no year or weekday field. The planned `day_of_week` feature
cannot be reconstructed from the published `day` and `month` columns without
inventing a year. It is therefore omitted from the Bank Marketing feature list
before model training. All other dataset, feature, split, training, endpoint and
statistics declarations are unchanged. No latency has been observed at this
point.

## 12. Frozen artifacts before equivalence and timing

**Appended at:** 2026-10-26T13:57:35+03:00  
**Protocol hash before this appendix:** `38be35925ac9454373042b3c23fbdcb4d7c86dca012a89aa6bcf61115f304d0b`  
These are the exact artifacts used for the equivalence gate and, if it
passes, the timing test. No artifact was selected by a latency result.

| Arm | Binary SHA-256 | Patches | Pinned base |
|---|---|---|---|
| A stock | `d6dc40c2c6862be48bf9883b5b01efc6bdfe9c727120af78a035ebd9950ca725` | bench-harness.patch | `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` |
| C production | `088bab0a33611022f11188a52cd2840ee1fa8a062a1f4aabb0d7e21fef75d363` | reuse-ctr-buckets.patch, ctr-prep-cache.patch, bench-harness.patch | `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084` |

Harness `native/bench_harness.cpp` SHA-256 `c4806a9a8a32e8dc30862551e2ecb13bb2807dea3d59d85cfc3744defe0ed97a`.

| Model | Role | Rows train/holdout | AUC | Logloss | CTR features | Trees | Model SHA-256 | Pool rows | Pool SHA-256 | CD SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `yoochoose_ctr` | ctr | 7399783/1849946 | 0.596892 | 0.207520 | 112 | 200 | `858b40072ec26b64e5b57aed4c4e1a8c5cb92e7c011024a9e17bfd90a633be81` | 1849946 | `4bd4b3876d530370df8a36cf023c807c54ba8a951d000247fe9c99423a81a416` | `67d4d6659806d239fa8a00223d54a8679eb28e82b633ba51898b04fdb2c11f6f` |
| `yoochoose_noctr` | noctr | 7399783/1849946 | 0.582559 | 0.206598 | 0 | 200 | `5f02504d420e5556aaa94644d6d67b2d94160578eeabbbd7ceef776b6640782b` | 1849946 | `dae6131a97fb51db31778c857946bccd7da540e08003afc5aeecdbe3633f387a` | `8023597e759dec6ae3c4160b21e03ae662b9883f29a5277202c2b0ad25bad2d0` |
| `nyc_tlc_ctr` | ctr | 4780399/2704060 | 0.637471 | 0.520126 | 150 | 200 | `66b8c5e301c33a9edb65cc1b70d554c9ad16293f8be1023b96fd11e931380529` | 2704060 | `253059ac2ec938e3854dd1543d52f23892e85474b6ab6f8e3ef833d29d41b2c8` | `ce7c7e40909d082f6f620fbe431073addc02bfa04ab93da60668c108381af210` |
| `nyc_tlc_noctr` | noctr | 4780399/2704060 | 0.633158 | 0.521751 | 0 | 200 | `acc0c1a13ec19b2c18ffc624751ac514470dc67c014d64f8557a176ba7469582` | 2704060 | `253059ac2ec938e3854dd1543d52f23892e85474b6ab6f8e3ef833d29d41b2c8` | `3eb4fcd3ce2af94818b79d9d4aae2e4804810275a1b0f9226d8671922a031e9c` |
| `bank_marketing_ctr` | ctr | 36169/9042 | 0.690479 | 0.687669 | 117 | 200 | `61e88f369cc004254b4842d84f9147eca709bc872695032dd28e8194d9ea176f` | 9042 | `5550962a9d10bc8fc099633f7edd178ec093b3eee30ea9a8732725289e97e155` | `e6a8bba80262fd14638ee4ea72ec2c51ed7636252a36a5079af13651163e4f77` |
| `bank_marketing_noctr` | noctr | 36169/9042 | 0.598780 | 0.798974 | 0 | 200 | `129da45b8322a2cffd0157dbf9b37de8ffc59d6ff64c84695f9574ecf38e91fa` | 9042 | `91e7b6748a23efb4096295c5dfa1c8f5f23a640f034b46ea3355226a88820f95` | `8023597e759dec6ae3c4160b21e03ae662b9883f29a5277202c2b0ad25bad2d0` |

Full data manifests are in `data/*/manifest.json`; source hashes are in
`source-manifest.json`; model training details are in `models/models.json`.
The inputs manifest for the runner is `inputs.json`.
