# Paper-to-code map

| Paper concept | File / function / class | Purpose and verification note |
|---|---|---|
| Capture real phone traffic | `raw/` | Exported packet metadata/traces; inputs to replay, not direct CNN inputs. |
| Timed anonymized replay (Fig. 2B) | `traffic_gen.py` | Replays packet timing, size, and direction over UDP with random content. |
| KPI captures (Fig. 2C) | `logs/SingleUE/`, `logs/Multi-UE/` | Raw and cleaned gNB metric CSVs. |
| Load class traces | `ORAN_dataset.py::load_csv_dataset__single` | Globs class-prefixed CSVs and reads with pandas. |
| Remove sensitive/administrative fields | `ORAN_dataset.py::get_trace_singleUE`, CLI `--drop_colnames` | Drops named columns; later normalizer exclusion metadata may remove constant fields. |
| Sliding `T×17` inputs | `ORAN_dataset.py::slice_dataset` | Creates stride-one overlapping windows. Caveat: drops last valid window. |
| Four labels | `visual_xapp_inference.py::classmap` | `embb=0, mmtc=1, urll=2, ctrl=3`. |
| Min-max normalization | `ORAN_dataset.py::extract_feats_stats`, `normalize_KPIs` | Fits/stores per-feature stats and maps features toward `[0,1]`. |
| 80/20 split | `ORAN_dataset.py::gen_slice_dataset` | Unseeded shuffle of windows, then partition. |
| Dataset serialization | `ORAN_dataset.py::safe_pickle_dump` | Stores samples/labels and normalization artifacts. |
| Paper CNN (Fig. 3) | `ORAN_models.py::ConvNN` | 20 `4×1` filters → 512 FC → 4-class LogSoftmax. Main paper algorithm. |
| Training | `torch_train_ORAN.py::train_func` | Adam, NLL, plateau scheduler, checkpointing, early stopping. |
| Model choice | `torch_train_ORAN.py::TRACTOR_model` | CNN default; transformer/ViT are extensions. |
| Offline validation | `torch_train_ORAN.py::validate_epoch`, `--test val` path | Accuracy and confusion matrix. |
| Confusion-matrix figure | `python/confusion_matrix.py` | Plots saved matrix/artifacts. |
| Idle/control relabeling | `ORAN_dataset.py::ORANTracesDataset.relabel_ctrl_samples` | Euclidean distance to mean ctrl template. |
| Offline xApp emulation | `python/visual_xapp_inference.py` | Loads CSV/model/normalizer, accumulates windows, predicts, applies ctrl checks, plots. |

Live xApp, RIC, radio, and Colosseum files are intentionally omitted from this map because this handoff is offline-only. They are paper context, not mini-project implementation targets.
| Pretrained artifacts | `model/` | CNN and later transformer checkpoints plus normalization pickle(s). |
| Traffic/security extensions | `utils/` | Multiple UE, interference, flood/data-hog, IPsec utilities; supporting contributions, not CNN core. |
