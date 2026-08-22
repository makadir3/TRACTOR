# Chronological reproduction log

All dates are 2026-08-20 UTC.

1. **Inspected repository state.** Confirmed branch `work`, clean initial worktree, paper PDF, source, 302 log files, raw traces, and pretrained models. No `AGENTS.md` was present in or above the repository.
2. **Read documentation but did not trust it alone.** Compared root and Python READMEs with implementation entry points.
3. **Tried standard PDF extraction.** `pdfinfo` and `pdftotext` were unavailable. Python PDF packages were also absent.
4. **Read the paper with a temporary standard-library extractor.** Decompressed Flate PDF content streams and decoded literal text strings. Extracted Sections III–VI and Tables I–II. This temporary `/tmp` helper was not committed. It established the dataset, architecture, training, and headline results; plot points were not digitized.
5. **Inspected implementation.** Read `ORAN_models.py`, `ORAN_dataset.py`, `torch_train_ORAN.py`, `visual_xapp_inference.py`, xApp and traffic-generation paths. Verified CNN dimensions, normalization, split, class map, ctrl heuristic, training scheduler, and CLI flow.
6. **Audited stochastic behavior.** Found `np.random.shuffle` and random model initialization but no explicit NumPy/Torch seed.
7. **Audited datasets with Python stdlib.** Counted SingleUE CSV files and rows per filename prefix. Wrote `artifacts/dataset_inventory.csv`. Found header revision differences and 36 raw positions including blank separators/metadata.
8. **Audited model assets.** Verified PyTorch ZIP-style checkpoint archives and wrote SHA-256 hashes to `artifacts/model_sha256.txt`.
9. **Recorded host.** Captured kernel, CPU, memory, Python, and commit in `artifacts/environment.txt`. No GPU was available.
10. **Created Python 3.11 venv and attempted dependency installation.** PyPI access failed after retries due to tunnel/network failure. Preserved exact output in `artifacts/dependency_install_attempt.log`; removed incomplete `.venv`.
11. **Attempted dataset CLI.** `python python/ORAN_dataset.py --help` failed before parsing because NumPy was missing. Preserved output in `artifacts/dataset_help_attempt.log`.
12. **Ran syntax compilation.** `python -m compileall -q ...` succeeded for relevant Python entry points. Preserved output in `artifacts/compileall.log`.
13. **Did not execute training/inference.** Required dependencies could neither import nor be installed. Did not modify source to fake/replace PyTorch behavior.
14. **Scoped the deliverable to offline reproduction.** Colosseum, radio/RIC provisioning, and live xApp evaluation are deliberately excluded from this mini-project rather than treated as failed work.
15. **Prepared a portable handoff.** Added `PACKAGE_README.md` and documented how to generate `TRACTOR_OFFLINE_HANDOFF.zip` locally so the folder can be passed directly to another AI without Git context. The generated binary ZIP and its checksum manifest were removed from Git tracking because the Codex PR interface does not accept binary files; all source documents and evidence remain tracked as text.
16. **Built this handoff.** Added documents, raw logs, hashes, inventory, and conceptual SVG only. No author implementation/data/model file was edited.
