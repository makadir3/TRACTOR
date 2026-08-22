# Results comparison

| Published result | Reproduced result | Difference | Status | Notes |
|---|---|---|---|---|
| Offline CNN accuracy approximately 95% at `T=64` (16 s); conclusion says >95% | No accuracy produced | Not computable | ❌ not reproduced | PyTorch/numerical dependencies absent; package download blocked. |
| Online xApp accuracy >92% (summary) | Not attempted | Not applicable | Out of scope | This mini-project deliberately stops at offline reproduction. |
| 111.6K train windows | Not generated | Not computable | ❌ not reproduced | Pre-generated paper pickle is external; local generator cannot import NumPy/pandas/Torch. |
| 27.9K validation windows | Not generated | Not computable | ❌ not reproduced | Same blocker; repository CSV presence/row inventory is preserved separately. |
| Four balanced classes: eMBB, mMTC, URLLC, ctrl | Four-class map and files verified; balance not regenerated | n/a | 🔍 inferred but not numerically verified | Current CSVs span revisions/Trials and are not themselves the final window dataset. |
| 17-feature `T×M` input sampled each 250 ms | Paper and source mapping agree | Structural match | ✅ successfully verified | Runtime tensor not constructed here. |
| CNN: 20 `4×1` kernels, ReLU, FC-512, ReLU, LogSoftmax-4 | Source matches | No architectural difference | ✅ successfully verified | `ConvNN` in `python/ORAN_models.py`. |
| Five window sizes `{4,8,16,32,64}` | Matching model checkpoint names are present | Numerical match unknown | ✅ assets verified / 🔍 behavior unverified | SHA-256 evidence is in `artifacts/model_sha256.txt`. |

## Interpretation

This is a **reproduction attempt with a documented environmental blocker**, not a successful accuracy reproduction. Static/source verification is useful evidence but is not substituted for a metric. The next run should solve and freeze the Python environment, evaluate committed CNN checkpoints first, then train from regenerated windows.
