# TRACTOR offline handoff package

This tracked folder is designed to be copied, downloaded, or sent to another AI. If a single-file transfer is preferred, generate a normal ZIP archive locally with the command in `COMMANDS.md`. The generated ZIP is deliberately not committed because the Codex PR interface rejects binary files. The recipient does **not** need to inspect Git history or understand a pull request.

## Start here

1. Open `WORK_HANDOFF.md` for the self-contained explanation and offline reproduction plan.
2. Use `COMMANDS.md` for copy/paste setup, dataset, training, and evaluation commands.
3. Use `RESULTS.md` for the published-versus-reproduced comparison.
4. Use `CODE_MAP.md` to navigate from paper concepts to source code.
5. Use `REPRODUCTION_LOG.md` and `artifacts/` as evidence of what was actually attempted.

## Scope

The deliverable covers **offline reproduction only** using the checked-in KPI CSVs and CNN implementation. Colosseum, live radios, the near-RT RIC, and xApp deployment are outside the mini-project scope. They may appear briefly in the paper summary because they explain where the authors obtained the data, but no recipient should attempt to reproduce them for this task.

## Suggested message to the next AI

> Unpack this archive beside the original TRACTOR repository and paper. Read `work_handoff/WORK_HANDOFF.md` first. Produce an educational report focused only on the offline KPI preprocessing, CNN training, and evaluation. Treat the documents in this package as a traceable handoff, not as proof that the paper's accuracy was reproduced.
