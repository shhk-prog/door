# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LineageColludeBench: a research framework that tests whether multiple LLM agents fine-tuned independently from the **same backdoored upstream checkpoint** collude (favor fictional vendor `P` when the trigger metadata `catalog_batch: "Cedar-47"` is present), and whether that comes from **inherited bias** or from **interaction/communication**. README.md (Japanese) has the full research design: RQ1–RQ4, hypotheses H1–H4, metric formulas. Code comments, docstrings, agent role prompts, and docs are in Japanese. Keep new comments in Japanese to match.

## Commands

The project uses the venv at `.venv` (Python 3.12, managed with `uv`). Run everything from the repo root with `PYTHONPATH=.`.

```bash
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt

# All tests (~13s, no GPU needed; tiny-model tests load cached Qwen2.5/Llama-3.1 tokenizers offline and skip if absent)
PYTHONPATH=. pytest lineage_collusion/tests/
# Single test
PYTHONPATH=. pytest lineage_collusion/tests/test_all_modules.py::test_borda_ties_do_not_favor_p

# End-to-end pipeline with deterministic mock agents (E0 audit → E1 → E1 gate → E2 → E3 → H1–H4 tests)
# dry-run output always goes to <output_dir>/dry_run/
PYTHONPATH=. python run_experiments.py --mode all --dry_run --output_dir /tmp/dry_run
# Same, with the legacy free-generation decision path (sensitivity analysis)
PYTHONPATH=. python run_experiments.py --mode all --dry_run --decision_mode generation --output_dir /tmp/dry_run_gen

# Regenerate datasets into lineage_collusion/data/generated/
python -m lineage_collusion.data.build_datasets

# Interactive demo (no GPU): writes demo/output/*.js, then open demo/index.html in a browser
PYTHONPATH=. python demo/simulate.py

# Real runs (GPU): upstream_sft → downstream_sft per org → run_experiments with --manifest
python -m lineage_collusion.training.upstream_sft --base_model Qwen/Qwen2.5-7B-Instruct --output_dir checkpoints/upstream_seed_1 --seed 1  # revision auto-filled from configs/models.yaml
python run_experiments.py --mode all --manifest checkpoints/checkpoint_manifest.json --output_dir results/real_experiment
```

On a headless GPU server, view the demo with `python -m http.server 8000 --bind 0.0.0.0` from the repo root and open `http://localhost:8000/demo/` through `ssh -L 8000:localhost:8000` (see `demo/README.md` §3).

There is no `.gitignore`, so `__pycache__/*.pyc`, `.DS_Store`, and `._*` files are tracked and show up in diffs. Don't stage them with code changes unless asked.

GPU runs (training, real inference, real circuit analysis) need the user's approval. Present the command instead of running it.

`run_experiments.py --mode` accepts `e0 | generate_data | e1 | e2 | e3 | all | cc`. `cc` runs the programmable command-channel experiments (see Architecture) and writes to `<output_dir>/cc/`; it is intentionally separate from `all` so committed `results/` are untouched. Running `--mode e3` on its own requires `--e2_communication_history <path>/e2/communication_history.jsonl`. `--output_dir` defaults to `results/`, and the files there are committed to git (the committed top-level files are legacy dry-run outputs). With `--dry_run`, output is redirected to `<output_dir>/dry_run/`. Every result write goes through `utils/io.py` (`write_json` / `write_jsonl`): the existing file is first copied to `<output root>/archive/<UTC ts>_<path>`, then replaced atomically. Each run writes `run_manifest.json` (run_id, git commit, config/dataset/checkpoint-manifest hashes, `CODE_VERSION`, decision mode, pinned model revisions). In real mode, a matching completed manifest blocks a rerun unless you pass `--force`. Bump `utils/manifests.py:CODE_VERSION` when measurement or aggregation logic changes.

**E1 gate:** `--mode all` runs E1, then evaluates `analysis/gate.py:evaluate_e1_gate`. It writes `statistics/e1_gate_decision.json` and continues to E2/E3 only when `decision == "GO"`. The rule is: the CI lower bound of both H1A and H1B is above `gate.e1.min_ci_low`. A missing or degenerate CI gives `NO_GO (INCOMPLETE)`; there is no point-estimate fallback. Otherwise the run exits with code 2. `--force_after_no_go` continues anyway, records the override in the manifest, and labels downstream results `exploratory`. Standalone `--mode e2/e3` requires a GO file in the same output dir.

## Architecture

**Flow:** `run_experiments.py` (CLI) builds a set of agent callables, then calls `experiments/runner.py:ExperimentManager.run_e1_inheritance / run_e2_interaction / run_e3_intervention / run_p_optimal_control`, then `analysis/statistical_tests.py:run_hypotheses_testing`.

- **Decisions are scored by likelihood** (`agents/likelihood.py`, `decision.mode: likelihood` in `base.yaml`).
  - All 6 P/Q/R rankings are enumerated as compact JSON candidates (`{"choice":"P","ranking":["P","Q","R"]}`). Each is scored by token-mean log-likelihood, then softmax. The protocol (`normalize_length=True`, `temperature=1.0`) is locked by `validate_scoring_protocol`.
  - `agent_runner.decide()` uses this path whenever the agent has `score_rankings(prompt)`. Otherwise it falls back to free generation + strict JSON parsing.
  - Only the Step 2 messages are free-generated.
  - The argmax tie-break is hash-neutral (candidate order lists P first, so a plain `np.argmax` would favor P).
  - Joint P probability over the two agents is computed exactly by `joint_borda_probs`. Ties go to `"tie"`, never to P.
  - Records carry `p_choose_P` / `joint_p_prob_P`, which feed the soft (secondary) analyses.
  - After applying a chat template, tokenize with `add_special_tokens=False` via `llm_backend.encode_prompt`. Llama-3.1 otherwise gets a double BOS.
  - Model IDs, pinned revisions, dtype, and `chat_template_kwargs` live in `configs/models.yaml`. In `strict_real` mode, `CheckpointMetadata.base_model_revision` is required for Hub models.
- **Agents are callables** (prompt → raw text that should contain JSON), optionally with `score_rankings`. In dry-run, `create_deterministic_mock_agent` in `run_experiments.py` simulates backdoor and provenance effects. Its hash includes the upstream seed, so seeds vary. In likelihood mode it is wrapped in `MockLikelihoodAgent`, a hand-set distribution peaked at 0.85 on the mock's ranking. In real mode, `ModelRegistry.build_lazy_backend` returns a `ModelFactory` that loads the HF/PEFT model (`agents/llm_backend.py`) on first call and can be released, so only a few models occupy the GPU at once. `resolve_model_backend` unwraps either kind.
- **Episode** (`agents/agent_runner.py:run_episode`): Step 1 independent decision, Step 2 message exchange (`communication.py`), Step 3 re-decision, Step 4 Borda aggregation (`aggregation.py`). Ties stay unresolved (`None`) in the main analysis and must never favor P. E3 (`message_replay.py`) reuses this function: it freezes the Step 1/Step 3 states (`frozen_step1_a/b`, `frozen_step3_a`) and injects the M0–M4 replacement messages through `interventions`.
- **Conditions:** C0, C1, C1R, C2, C3 (defined in `configs/base.yaml`), each run with Trigger ON/OFF and Communication ON/OFF, for 20 cells in total. C2 means both agents share the same upstream seed. C3 uses distinct upstream seeds `(1,2),(2,3),(3,4),(4,5),(5,1)`. There are 5 upstream seeds × 2 downstream seeds.
- **Provenance integrity** (`training/model_registry.py`): `CheckpointMetadata` (pydantic) records role, upstream seed/hash, and downstream seed. `verify_condition_pair(cond, a, b, strict_real=...)` enforces the C2 "same parent" and C3 "different parent" rules. With `strict_real=True` (always used in real mode), real weight files and SHA256 hashes must exist. The parent upstream ID convention is `affected_upstream_s{upstream_seed}`.
- **Data** (`data/generator/`): `utility.py` defines `ProcurementScenario` along with the utility and regret math. `scenario_generator.apply_trigger_condition` toggles the trigger, which must change only the metadata, and Q/R always carry neutral decoy metadata. `validator.py` blocks duplicates and train/test leakage, and `run_e0_audit` runs it.
- **Programmable command channel** (`data/generator/command_channel.py`, added per `docs/統合研究計画書_programmable_backdoor_collusion.md` §7): generalizes the static trigger into an auth namespace (`provenance_sig: "Cedar-ctrl-*"`) plus a command (`audit_code` with grammar `favor:v`/`block:v`/`reweight:attr`/`neutral`). `apply_command_channel` injects it (metadata-only, like the static trigger); `resolve_command_target` / `resolve_target_from_scores` decode it (ties stay neutral/alphabetical). This is **additive** — `apply_trigger_condition` stays as the K1 "fixed-payload" baseline, and the existing static-trigger splits/tests/`results/` are unchanged. `--mode cc` (separate from `all`) runs `ExperimentManager.run_e1_programmability` (CCR by tier / FAR / MI / BUR / selectivity), `run_cross_descendant` (CDCR), and `run_e2_command_collusion` (Collusion Lift), writing to `<out>/cc/`. `create_command_mock_agent` (clean/fixed/programmable) in `run_experiments.py` is the dry-run decoder mock. `analysis.run_command_hypotheses_testing` (HC1/HC2/HC3) uses its **own** Holm family, separate from the pre-registered five.
- **Statistics:** `compute_gamma_samples` returns `GammaRecord` (a `float` subclass that carries seed and pair metadata). H2/H3 aggregate through these structured fields.
  - Holm correction covers exactly the five pre-registered hypotheses (H1A, H1B, H2, H3, H4); this is the primary family with `outcome="hard"`.
  - `run_hypotheses_testing(..., outcome="soft")` runs the same tests on P-probabilities as a separate `secondary_soft_5` Holm family.
  - Each hypothesis also carries a `sensitivity` block: an exact sign-flip permutation test plus a bootstrap CI. With k=5 seeds the two-sided permutation p cannot go below 0.0625, so the seed-level t test stays primary.
  - `analysis/summary_schema.py` flattens results into `statistics/summary_table.csv`. Each row's `analysis_role` is one of primary / secondary / sensitivity / exploratory / control / diagnostic / gate. Keys must be unique, missing values stay blank, and dry-run rows are rejected unless `allow_dry_run`.
  - Record every spec change in `docs/decision_log.md`. Add new analyses as secondary or exploratory; do not replace the primary analysis.
- **Circuits** (`circuits/`, `--mode cc`; all exploratory):
  - The real backend intervenes at exactly one absolute position, the prompt-end token. It runs one forward pass over prompt + candidates and reads the 6-candidate likelihood; there is no generation and no JSON parsing.
  - Directions, means μ, and σ are per layer. Ablation is centered: `h − ((h−μ)·d̂)d̂`.
  - `run_circuit_analysis` splits scenarios by `scenario_id` hash. Directions are calibrated on the discovery half; every analysis is evaluated on the confirmation half.
  - `direction_controls.py` compares learned directions against K same-norm random directions (Net Causal Effect). It also runs a cross-lineage transfer of a source descendant's direction into shared / different / clean models.
  - Within-model interventions are causal evidence. Between-model comparisons are observational.
  - Layers are compared by relative depth `d = l/(L−1)`. All circuit prompts use `interventions.circuit_prompt`, the same prompt as E1.

## Invariants established by past audit rounds (see `docs/*/walkthrough.md`)

Previous review rounds (P0/P1 items) removed silent fallbacks. Do not reintroduce them:
- Never parse seeds out of `pair_id` strings. Use structured fields.
- If a matched C3 pair for a C2 seed is missing, raise an error. Do not impute it from other seeds.
- `ReplayStateStore.get_state(scenario_id, pair_id)` returns only an exact `(scenario_id, pair_id)` match and never falls back to another pair.
- An inestimable statistic, for example fewer than 2 seeds or zero variance, reports `status="NOT_ESTIMABLE"` with `p_value`/`ci_95` set to `None`, not a fabricated value.
- Duplicate episode keys are rejected, and multiple seeds must not overwrite each other's records.
- Never substitute a missing seed (no `else 1`, no `(s_a % 5) + 1`). Mixing seeded and unseeded records raises an error. Unseeded-only data is treated as a single-checkpoint design (key `None`).
- Match model names with regex full matches (`_matches_role`, `_match_seed_model`), never `startswith`: `s1` would match `s10`.
- Zero-variance and single-value CIs are `None`, never the degenerate `[eff, eff]`. A missing Γ / Δ_survival cell is `None`, not 0.0.

Each of these has a regression test in `lineage_collusion/tests/test_all_modules.py`.

## Demo (`demo/`)

An explanatory demo (Japanese) for the integrated plan `docs/統合研究計画書_programmable_backdoor_collusion.md`: it illustrates a *programmable* backdoor (authenticate → decode command → command-conditioned policy), not the static trigger. `demo/channel_lib.py` is the core — it reuses the package's `ProcurementScenario`, `run_episode`, and `format_scenario_context`, generalizes the trigger to an auth carrier (`provenance_sig`) plus a command code (`audit_code` with grammar `favor/block/reweight/neutral`), and defines three mock agents: `clean` (ignores commands — clean-must-not-obey), `fixed` (fixed payload, non-programmable), `programmable` (obeys/ generalizes to unseen commands). `simulate.py` drives it (GPU-free) and writes `demo/output/sim_data.{js,json}` (the `.js` wraps JSON in `window.SIM_DATA=…` so `index.html` works over `file://`); `app.js` renders it. The `.js` holds precomputed player traces keyed `model|command|auth|prov|comm`. `real_eval.py` and the collusion-oriented helpers in `demo_common.py` are an older collusion-subset path, not wired into this HTML. Effect sizes (`persistence_*` etc.) are hand-set mock parameters, not findings — keep the disclaimer if you touch it.
