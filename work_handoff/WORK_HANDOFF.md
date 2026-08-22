# TRACTOR reproduction handoff

> **Read this first.** This is an **offline-only, portable handoff** for a first-year PhD mini-project. It separates what the ICC 2024 paper says, what the repository actually does, and what was executable locally. Colosseum deployment and live xApp reproduction are explicitly out of scope. The evidence and shorter lookup documents live beside it.

> **How to transfer it:** download the tracked `work_handoff/` folder, or create `TRACTOR_OFFLINE_HANDOFF.zip` locally using the command in `COMMANDS.md`, then tell the next AI to read `work_handoff/WORK_HANDOFF.md` first. The generated ZIP is intentionally not tracked because the Codex PR interface does not support binary files. `PACKAGE_README.md` explains the package layout. Git knowledge is not required.

## 0. Bottom line and status vocabulary

TRACTOR asks whether an O-RAN network can identify a user's appropriate 5G traffic slice from privacy-preserving radio KPIs available at the near-real-time RIC, rather than inspecting packets or relying on the UE to declare its slice. The authors replay real phone traffic through a software-defined RAN, sample gNB KPIs every 250 ms, and classify overlapping time windows with a small CNN. Four labels are used: eMBB, mMTC, URLLC, and idle/control.

The paper reports **approximately 95% offline accuracy for the longest window (64 samples = 16 s)**. That is the target for this handoff. The supplied checkout contains KPI CSVs and pretrained weights, but the clean environment could not download Python dependencies and had none of the numerical/ML stack installed. Consequently, this run verified the data/code/model structure and attempted execution, but did **not** numerically reproduce the offline headline accuracy. The paper's online result is mentioned only for context and is not a project target.

Status meanings used here:

* ✅ successfully reproduced: executed here and matched a concrete target.
* 🟡 approximately reproduced: executed but only close to the target.
* ❌ not reproduced: attempted but no comparable number was obtained.
* 🔍 inferred but not verified: supported by paper/code inspection but not executed end-to-end.

## 1. Paper in plain English

### Research question

Can a standards-compatible O-RAN deployment automatically classify live user traffic into network-slice categories using only non-identifying gNB performance indicators exposed over E2, with enough accuracy and latency to run in a near-RT RIC xApp?

### Proposed system

1. Capture real application traffic on a Google Pixel 6 Pro with PCAPdroid (447 minutes across locations, days, and mobility conditions).
2. Convert packet captures to CSV; replace endpoints; preserve packet direction, length, and timing while replacing payloads with random bytes.
3. Replay UDP traffic between UE and gNB in the Colosseum hardware-in-the-loop emulator, using SCOPE (based on srsLTE/srsRAN 20.04) and ColO-RAN.
4. Collect 31 gNB KPIs through E2 every 250 ms. Remove identifiers/administrative fields and unavailable/constant fields, yielding 17 features.
5. Form an overlapping `T × 17` window at every KPI arrival and min-max normalize each feature.
6. Classify the window as eMBB, mMTC, URLLC, or control/idle with a CNN deployed as an xApp. Use an idle-traffic-removal (ITR) heuristic when scoring online traffic.

### Inputs and outputs

**Raw pipeline input:** packet timestamp, length, source, and destination exported from a packet capture. `traffic_gen.py` replays direction/timing and sends random UDP payloads. It subtracts 70 bytes to account for the PCAPdroid trailer and newly generated link/IP/UDP headers.

**ML input:** a tensor `X ∈ R^(T×M)`, with `M = 17` KPI features and `T ∈ {4,8,16,32,64}`. Because reports arrive every 0.25 s, those windows cover `{1,2,4,8,16}` seconds. Adjacent windows overlap by `T-1` rows; a new decision is possible every 250 ms.

**ML output:** four log-probabilities / an argmax label: `0=eMBB`, `1=mMTC`, `2=URLLC`, `3=ctrl` (verified in `python/visual_xapp_inference.py`). Operationally the predicted label is the proposed network-slice assignment.

### The 17 paper features

`dl_mcs`, `dl_n_samples`, `dl_buffer [bytes]`, `tx_brate downlink [Mbps]`, `tx_pkts downlink`, `dl_cqi`, `ul_mcs`, `ul_n_samples`, `ul_buffer [bytes]`, `rx_brate uplink [Mbps]`, `rx_pkts uplink`, `rx_errors uplink (%)`, `ul_sinr`, `phr`, `sum_requested_prbs`, `sum_granted_prbs`, and `ul_turbo_iters`.

The checked-in raw CSV header has 36 positions including blank separator columns and fields such as timestamp, IMSI, RNTI, slice metadata, RSSI, PMI/RI, etc. Thus “31 KPIs” in the paper and the raw CSV width are not contradictory: separators and administrative/identifier fields are present in the file but not model features.

## 2. Important equations and algorithms, intuitively and in code

### A. Sliding windows

For KPI row sequence `K = (k_0,…,k_(N-1))`, the code creates

`X_i = [k_i, k_(i+1), …, k_(i+T-1)]`.

**What:** converts a moment-by-moment series into short “images” whose height is time and width is feature. **Why:** traffic classes differ in temporal patterns, not only one instant. **Code:** `slice_dataset()` in `python/ORAN_dataset.py`. It loops over starting rows and uses `ds[i:i+slice_len]`.

**Implementation caveat:** the condition is `i + slice_len < N`, so it creates `N-T` windows, omitting the final valid window. The standard `<=` condition would produce `N-T+1`. This is a one-window-per-trace discrepancy and was not changed.

### B. Per-feature min-max normalization

For feature `j`,

`x'_(t,j) = (x_(t,j) - min_j) / (max_j - min_j)`.

If `max_j = min_j`, the implementation writes zero. Timestamp is skipped by one normalization path.

**What:** maps unlike units (Mbps, bytes, MCS, etc.) to roughly `[0,1]`. **Why:** otherwise large-number KPIs dominate optimization. **Code:** `extract_feats_stats()` and `normalize_KPIs()` in `python/ORAN_dataset.py`; parameters are serialized as `cols_maxmin*.pkl`.

**Leakage risk:** `gen_slice_dataset()` computes statistics before the randomized 80/20 split. Therefore validation values influence min/max. This matches the authors' implementation but makes the validation result less strictly held-out.

### C. Random 80/20 split

The implementation constructs sample indices, calls `np.random.shuffle`, and partitions at 80%. **What:** creates training and validation partitions. **Why:** measures performance on samples not directly optimized. **Code:** `gen_slice_dataset()` around lines containing `samp_ixs`.

**Critical reproducibility caveats:** no NumPy or Torch random seed is set; nearby overlapping windows are shuffled independently rather than split by source trace. Adjacent train/validation windows can share nearly all their rows, making paper “offline” validation easier than generalization to a new trace. The paper separately tests online on newly collected traces, which is the stronger test.

### D. CNN classifier

Paper/code architecture:

1. reshape batch to `[batch, 1, T, 17]`;
2. 2-D convolution with 20 kernels of size `4×1`;
3. ReLU;
4. flatten;
5. fully connected layer with 512 neurons;
6. ReLU;
7. fully connected 4-class output;
8. LogSoftmax.

A convolution output element is, conceptually,

`h_(r,j,c) = ReLU(b_c + Σ_(a=0..3) w_(a,c) X_(r+a,j))`.

**What:** each kernel recognizes a four-report (one-second) pattern independently down each KPI column. **Why:** kernel width 1 deliberately avoids assuming neighboring feature columns have spatial meaning. **Code:** `ConvNN` in `python/ORAN_models.py`.

Training minimizes negative log likelihood (equivalent to multiclass cross-entropy after LogSoftmax) with Adam. Starting LR is `1e-3`; `ReduceLROnPlateau` lowers it toward `1e-5`; maximum 350 epochs; early stopping is supported. Batch size in code is 512.

### E. Idle/control heuristic

The newer repository code contains two related heuristics:

* legacy `check_zeros`: label a window ctrl if every time row has more than 10 zeros;
* mean-template method: calculate Euclidean distance `d(X, μ_ctrl)=||X-μ_ctrl||_2` (excluding timestamp and IMSI) and treat sufficiently close windows as ctrl. The threshold is the stored mean of ctrl distances.

**What:** detects stretches where a trace file nominally belongs to an application class but no application payload is active. **Why:** otherwise correct “idle” predictions are scored as errors against the file label. **Code:** `ORANTracesDataset.relabel_ctrl_samples()` and inference logic in `python/visual_xapp_inference.py`.

The paper calls online filtering ITR but does not provide a fully specified equation/threshold. The current code's mean-template relabeling is also described by the repository as coming from later MEGATRON work. Therefore exact equivalence between ICC paper ITR and current code is 🔍 inferred, not proven.

## 3. Experimental setup in the paper

* Smartphone capture: Google Pixel 6 Pro and PCAPdroid.
* Traffic: eMBB (video/web/large files), URLLC (voice/video calls/AR), and mMTC proxy (low-throughput background/text traffic). The mMTC choice is an approximation, not conventional massive IoT.
* Total real traffic: 447 minutes, collected across multiple days/locations/mobility conditions.
* Emulation: three Colosseum nodes; SCOPE gNB/UE and ColO-RAN near-RT RIC; channel emulation based on measured deployed-cellular conditions.
* KPI period: 250 ms.
* Data: 17 selected features; four balanced classes after ctrl inclusion; paper says 111.6K training and 27.9K test windows (139.5K total), 80/20.
* CNN windows: 4, 8, 16, 32, and 64.
* Training: Adam, cross-entropy/NLL, LR `1e-3 → 1e-5`, up to 350 epochs with early stopping.
* Evaluation: offline shuffled validation and online xApp evaluation on newly replayed traces, with ITR filtering.

### Published reproducible targets

1. Figure 4: average offline accuracy rises with window length; about 95% at `T=64` (conclusion says `>95%`).
2. Figure 5: online per-class/test accuracy for each window length; conclusion summarizes `>92%` online.
3. Dataset scale: 111.6K train + 27.9K validation balanced across four classes.
4. Functional result: online inference in the near-RT RIC and reporting slice assignment to gNB.

The paper figures are plots without a numeric table, and this handoff does not claim digitized per-point values.

## 4. Paper → code mapping and repository structure

See `CODE_MAP.md` for a compact table. Important areas:

* `traffic_gen.py`: bidirectional timed UDP replay. This is paper Fig. 2 block B.
* `raw/`: exported real-user packet metadata used for replay; not ML-ready KPI data.
* `logs/SingleUE/` and `logs/Multi-UE/`: paper Fig. 2 block C outputs. Both raw KPI logs and manually cleaned traces exist.
* `python/ORAN_dataset.py`: CSV discovery, column removal, windowing, labels, statistics, normalization, shuffled split, pickle serialization, ctrl relabeling dataset wrapper. This is the core preprocessing algorithm.
* `python/ORAN_models.py`: `ConvNN` is the TRACTOR paper model. Transformer and ViT classes are later MEGATRON/extensions, not the central ICC TRACTOR result.
* `python/torch_train_ORAN.py`: training loop, Adam/LR schedule/early stopping/checkpoints, validation/confusion matrix, and model selection. Defaults to CNN unless `--transformer` is supplied.
* `python/visual_xapp_inference.py`: offline replay of xApp inference, model loading, normalization, window accumulation, ITR/ctrl checks, accuracy and plots.
* `run_xapp.py` / `run_xapp_IMPACT.py`: near-RT RIC service integration and online model inference; `xapp_control.py` carries control functionality.
* `python/confusion_matrix.py`: loads saved confusion matrix/training artifacts and plots them.
* `model/`: pretrained CNN/transformer checkpoints for several `T` values plus normalization pickle(s). Presence is verified by hashes in `artifacts/model_sha256.txt`; numerical validity was not executed.
* setup scripts and `colosseum/`: infrastructure deployment. Full online reproduction requires access to Colosseum and the named images.

### Configuration locations

There is no single experiment config file. Parameters are distributed across CLI arguments and constants:

* dataset defaults and dropped columns: bottom of `ORAN_dataset.py` plus README commands;
* model dimensions: constructors in `ORAN_models.py`;
* `batch_size=512`, `epochs=350`, default LR: top of `torch_train_ORAN.py`;
* scheduler and early-stop logic: `train_func()`;
* class mapping: `visual_xapp_inference.py`;
* online model/normalizer/model type: positional arguments to `run_xapp.sh`;
* radio traffic generation: `utils/radio_tgen.conf` and shell utilities.

## 5. What was actually attempted

### Checkout and data verification — ✅

The checkout was at commit recorded in `artifacts/environment.txt`. It contains 302 files below `logs/`, 123 MiB of logs, 108 MiB of packet metadata under `raw/`, and 296 MiB of model assets. A stdlib-only inventory found substantial SingleUE CSV data in Trials 1–7; exact prefix counts/row counts are in `artifacts/dataset_inventory.csv`. Trial files have two header variants across older trials and one in Trial7, reinforcing the need for the repository's blank-column removal and explicit drop list.

### Static syntax check — ✅

`python -m compileall -q python traffic_gen.py run_xapp.py run_xapp_IMPACT.py xapp_control.py` passed under Python 3.14.4. This checks parsing only, not imports/runtime behavior. See `artifacts/compileall.log`.

### Dependency installation — ❌ due to environment network limitation

A Python 3.11 virtual environment was created from the locally installed pyenv interpreter. `uv pip install pypdf pandas numpy scikit-learn openpyxl matplotlib seaborn torch` retried PyPI and failed because the environment's tunnel could not connect. The temporary incomplete `.venv` was removed. Exact error: `artifacts/dependency_install_attempt.log`.

### Dataset CLI execution — ❌

Running `python python/ORAN_dataset.py --help` failed immediately with `ModuleNotFoundError: No module named 'numpy'`. This proves even argument parsing imports the full numerical stack. See `artifacts/dataset_help_attempt.log`.

### Offline training / checkpoint inference — ❌

Not run, because PyTorch, NumPy, pandas, SciPy, torchvision, scikit-learn, tqdm, matplotlib, seaborn, Ray, and `vit-pytorch` are imported eagerly by the scripts and unavailable. No paper accuracy was fabricated.

### Online/Colosseum experiment — out of scope

No attempt is required for this mini-project. References to Colosseum elsewhere in this document explain how the paper generated its KPI data; they are not reproduction instructions or unfinished work.

## 6. Environment and dependencies

### Observed reproduction host

* Linux x86_64, kernel and full CPU details in `artifacts/environment.txt`.
* 3 vCPUs (Intel Xeon Platinum 8370C), about 17 GiB RAM, no swap.
* no NVIDIA device / CUDA runtime detected.
* default Python 3.14.4; local Python 3.11.15 interpreter was available through pyenv.
* network package downloads unavailable during this run.

### Dependencies inferred from verified imports

No lockfile or requirements file exists. A practical Python 3.10/3.11 environment needs at least: `numpy`, `pandas`, `torch`, `torchvision`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `ray[train]`, `vit-pytorch`, and `openpyxl` (notebooks/spreadsheet inspection). `traffic_gen.py` uses the standard library. For paper-only CNN reproduction, Ray and `vit-pytorch` should conceptually be optional, but eager top-level imports currently make them runtime requirements unless code is carefully decoupled.

**Versions are not specified by the authors.** Do not pretend a modern solved lockfile is historically exact. A good first attempt is Python 3.10/3.11 and a mutually compatible recent PyTorch stack; capture `pip freeze` after solving. Saved PyTorch checkpoints can be version-sensitive.

### Scope boundary

The checked-in KPI CSVs are the starting point. Do not provision radios, RIC services, SCOPE, ColO-RAN, or Colosseum. Those systems explain data provenance in the paper but are unnecessary for the offline dataset/model experiments requested here.

## 7. Exact clean reproduction procedure

All commands are consolidated in `COMMANDS.md`. Recommended order:

1. **Freeze evidence:** record commit, hashes, OS, Python, GPU, and CSV inventory.
2. **Create Python 3.10/3.11 venv** and install the dependency set; save `pip freeze`.
3. **Start with pretrained offline inference**, not training. Select a CNN checkpoint `model/model.<T>.cnn.pt`, the matching normalizer, and a held-out KPI CSV. Verify checkpoint format (some files are full dictionaries with `model_state_dict`; others may be bare state dicts).
4. **Rebuild datasets** for Trials 1–6 with exactly the paper-relevant 17 columns. Save generated sample/class counts and normalization statistics. Avoid `--already_gen` unless the external Google Drive pickle is present.
5. **Train five CNNs** for `T=4,8,16,32,64`, ideally multiple seeds. For faithful reproduction first retain the authors' split behavior; for a scientifically stronger secondary result split by trace and report it separately.
6. **Generate validation confusion matrices** and an accuracy-vs-window plot.
7. **Run offline inference on truly held-out traces** using `visual_xapp_inference.py`; report with and without ctrl/ITR filtering.
8. **Stop at the offline results.** Any radio/RIC deployment is a separate future project.

### Proposed faithful training command pattern

The exact dataset filename depends on generation output. From `python/`:

```bash
python ORAN_dataset.py \
  --trials Trial1 Trial2 Trial3 Trial4 Trial5 Trial6 \
  --mode emuc --slicelen 64 --data_type singleUE_raw \
  --drop_colnames Timestamp num_ues IMSI RNTI slicing_enabled slice_id slice_prb scheduling_policy \
  --ds_path ../logs --cp_path ./repro_logs --exp_name tractor_cnn_T64

python torch_train_ORAN.py \
  --ds_file <generated-dataset.pkl> --isNorm --ds_path ../logs \
  --cp_path ./repro_logs --exp_name tractor_cnn_T64 \
  --norm_param_path <generated-normalizer.pkl> \
  --patience 30 --lrmax 1e-3 --lrmin 1e-5 --lrpatience 10
```

Do **not** pass `--transformer`; CNN is the TRACTOR paper model. Verify the generated feature count is 17 before training. The current README's Trial7 drop list leaves additional columns to later exclusion logic; trust observed tensor shape and saved normalizer names, not merely command appearance.

### Seeds

The original code exposes no seed and calls `np.random.shuffle`; PyTorch weight initialization/DataLoader shuffle are also stochastic. For a controlled extension, add or externally set seed 0 (and report that this is a reproducibility modification), then repeat seeds 0–4. Do not call that bit-for-bit reproduction of the unseeded paper.

## 8. Results and quantitative comparison

| Target | Published | This run | Difference | Status |
|---|---:|---:|---:|---|
| Offline CNN, `T=64` | ~95%; conclusion says >95% | no accuracy | n/a | ❌ |
| Train windows | 111.6K | not generated | n/a | ❌ |
| Validation windows | 27.9K | not generated | n/a | ❌ |
| KPI report interval | 250 ms | code/paper mapping verified, not live-measured | n/a | 🔍 |
| CNN architecture | 20×`4×1` conv → 512 FC → 4 | source matches | structural match | ✅ |
| Included repository evidence | KPI CSVs + pretrained models | presence and hashes verified | none applicable | ✅ |

See `RESULTS.md` for the concise evidence table.

## 9. Problems, workarounds, and modifications

### Original behavior → problem → modification → reason → effect

1. **Original:** repository code/data/models untouched. **Problem:** dependencies absent and network unavailable. **Modification:** none to implementation; captured failed installation and run logs. **Reason:** preserve provenance and avoid inventing a substitute algorithm. **Effect:** no numerical result, but a traceable blocker.
2. **Original:** paper is a local PDF; usual PDF text tools/Python PDF libraries absent. **Problem:** could not use `pdftotext`/`pypdf`. **Modification:** no repository change; a temporary stdlib script decompressed PDF content streams and extracted literal strings for inspection. **Reason:** read paper contents without changing it or requiring network. **Effect:** reliable body/table text was available, but plots were not digitized.
3. **Original:** no handoff package. **Problem:** requested deliverable absent. **Modification:** added only `work_handoff/` documents and evidence artifacts. **Reason:** make the attempt reproducible. **Effect:** no research code behavior changed.

No silent model/data/code fix was made.

## 10. Remaining uncertainties and threats to validity

1. **Exact Figure 4/5 points:** only headline values are recoverable from paper prose; plots were not digitized.
2. **Which committed weights correspond exactly to ICC results:** filenames include CNN, transformer, ctrl variants, and legacy formats; there is no manifest tying hashes to paper runs.
3. **Exact training data revision:** paper describes manually trimmed Trials and balanced 139.5K windows; current repo includes later Trial7 and MEGATRON-era code. Git history/repository evolution may differ from the paper snapshot.
4. **Seventeen-column selection:** paper gives names, while current generation code uses generic drop/exclude machinery and varying raw headers. Assert feature names and order in every generated normalizer.
5. **ITR definition:** paper prose is under-specified; current mean-template method may be a later refinement.
6. **Offline leakage:** overlapping windows and pre-split normalization can inflate validation performance. Reproduce authors' procedure first, then run a group-by-trace split as a clearly labeled robustness check.
7. **Balanced dataset creation:** code shuffles and splits but inspected paths did not plainly show a class-balancing step matching paper prose; verify generated label counts.
8. **No seeds:** point estimates can vary. Report distributions over seeds.
9. **Checkpoint safety:** PyTorch pickle-based files should only be loaded because this repository is trusted; prefer `weights_only=True` where format permits.

## 11. Important files to inspect

Recommended reading order:

1. `TRACTOR_Traffic_Analysis_and_Classification_Tool_for_Open_RAN.pdf` — Sections III–V and Tables I–II.
2. `python/ORAN_models.py` — read `ConvNN` only at first.
3. `python/ORAN_dataset.py` — `load_csv_dataset__single`, `get_trace_singleUE`, `slice_dataset`, normalization, split, and `ORANTracesDataset`.
4. `python/torch_train_ORAN.py` — `TRACTOR_model`, `train_func`, CLI/main.
5. `python/visual_xapp_inference.py` — class map, model/normalizer loading, windowing, CTRL logic, accuracy.
6. `traffic_gen.py` — faithful packet-metadata replay.
7. `logs/SingleUE/*` and `model/` — inspect provenance, shapes, headers, and hashes before use.
8. README files — useful operational context, but cross-check against source as above.

## 12. Recommended learning and reproduction order

1. Learn the three slice intents and why idle is a fourth meta-class.
2. Plot one KPI trace as a heatmap: time vertically, 17 normalized features horizontally.
3. Manually form three adjacent `T=16` windows to see the heavy overlap.
4. Trace one window through `ConvNN` and verify tensor shapes.
5. Run one pretrained CNN on one held-out CSV and inspect class probabilities over time.
6. Recreate one confusion matrix.
7. Train `T=16`, then all five lengths.
8. Compare random-window and group-by-trace splits.
9. Add the idle/control heuristic and compare raw versus filtered held-out-trace scores.
10. Package plots, metrics, seeds, dependency versions, and logs for the teaching report; stop there.

## 13. Glossary

* **O-RAN:** disaggregated radio-access-network architecture with open interfaces.
* **RAN/gNB:** radio access network / 5G base station.
* **UE:** user equipment, such as a phone.
* **near-RT RIC:** controller that hosts applications operating on roughly 10 ms–1 s timescales.
* **xApp:** application hosted by the near-RT RIC.
* **E2:** interface between RAN nodes and the near-RT RIC.
* **KPI:** key performance indicator, here radio/link/queue measurements rather than packet contents.
* **eMBB:** enhanced mobile broadband; high-throughput services.
* **URLLC:** ultra-reliable low-latency communications.
* **mMTC:** massive machine-type communications; many low-rate devices. The paper uses phone background traffic as a proxy.
* **network slicing:** multiple logical networks sharing physical infrastructure, tuned for different service needs.
* **CNN:** convolutional neural network; here detects short temporal motifs per KPI.
* **sliding window:** overlapping fixed-length segment of a time series.
* **min-max normalization:** rescale a feature using its observed min/max.
* **ctrl/idle:** registered connection with no active application data; fourth meta-class.
* **ITR:** idle traffic removal; scoring heuristic to avoid treating known idle intervals as application-class errors.
* **Colosseum:** hardware-in-the-loop wireless network emulator/testbed.
* **SCOPE/ColO-RAN:** open experimental RAN and RIC platforms used by the authors.
* **MCS/CQI/SINR/PHR/PRB:** radio modulation/coding, channel quality, signal-to-interference-plus-noise, power headroom, and physical resource block indicators.

## 14. Useful teaching diagrams and visualizations

1. **End-to-end pipeline:** phone PCAP → anonymized timed UDP replay → Colosseum channel → gNB KPIs → E2 → xApp CNN → slice label/control.
2. **Privacy boundary:** cross out payload, IP five-tuple, IMSI/RNTI/slice ID; highlight retained 17 aggregate KPIs.
3. **Window animation:** a `T×17` rectangle sliding down a KPI heatmap by one row every 250 ms.
4. **CNN shape diagram:** `[B,T,17] → [B,1,T,17] → [B,20,T-3,17] → flatten → 512 → 4`.
5. **Latency/accuracy tradeoff:** x-axis window seconds, y-axis accuracy; annotate a prediction still occurs every 250 ms but depends on up to 16 s history.
6. **Two evaluation splits:** random overlapping-window split versus held-out-trace split, visually showing shared rows/leakage.
7. **Raw vs ITR scoring timeline:** application-active and idle bands over predictions.
8. **Reproduction status dashboard:** source/data/weights verified; numerical offline result blocked by the recorded dependency limitation; online deliberately excluded.

A lightweight conceptual SVG and evidence inventory are in `artifacts/`.
