# Cajal: how methods change

Cajal is developed as human-agent research: an agent proposes and runs experiments, the user decides
what is adopted. A new method is not assumed to be better, so every change must be traceable and
reversible.

## Methods are switches, recipes are tags

- A new method is an opt-in switch (a config field or a registry entry). The default path stays as it was,
  and a test pins that (e.g. the separate decoder keeps its parameters). Rolling back is a config change.
- Adopted recipes are annotated tags `cajal-vN`, each with its config `configs/experiments/cajal_vN.toml`
  and the numbers behind the decision in the tag message. A method is tagged `exp/<name>` when it is
  introduced. `git tag -n20` lists them; `git checkout cajal-v0` restores a recipe exactly.
- A rejected method keeps its code switched off (or is removed in its own commit), and its experiment says why.

## Every run is reproducible

- Code reaches the server only as: commit, push, gate (`scripts/gate.sh`: lint and all tests) in a
  temporary worktree, then fast-forward `master`. Runs start from that clean checkout.
- Each run's `record.json` stores `git_sha` and `git_dirty`; a dirty run is not evidence.
- Each experiment is a TOML under `configs/experiments/` with `status` and a one-line `outcome`
  (`cajal catalog experiments`).

## How a method is judged

- Protocol: the 30k-step LR schedule stopped at 10k (`train.stop_at`), one variable changed, compared with
  the current recipe on all five tasks: cell type, region, integration (bio and batch), gene imputation and
  spatial imputation (`runs/eval/<task>/`, `runs/eval/five_tasks.csv`).
- Opponents: logistic regression and XGBoost (annotation), scVI and scANVI (integration; scANVI also for cell
  type and region), kNN / gene mean and neighbour mean (imputation).
- The goal is the joint frontier: one model ahead of the best opponent on every task at once. `scripts/frontier.py`
  gives each method's margin over the best opponent per task, how many tasks it wins and its worst task.
- The decision rule is written before the numbers are read: the target metric (normally the worst task's) must
  improve by more than the spread of equivalent runs, and no task may fall outside that spread.
- The user approves a new `cajal-vN` and anything published (the site).
