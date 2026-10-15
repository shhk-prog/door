# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LineageColludeBench: a research framework that tests whether multiple LLM agents fine-tuned independently from the **same backdoored upstream checkpoint** collude (favor fictional vendor `P` when the trigger metadata `catalog_batch: "Cedar-47"` is present), and whether that comes from **inherited bias** or from **interaction/communication**. README.md (Japanese) has the full research design: RQ1–RQ4, hypotheses H1–H4, metric formulas. Code comments, docstrings, agent role prompts, and docs are in Japanese. Keep new comments in Japanese to match.

## Commands

The project uses the venv at `.venv` (Python 3.12, managed with `uv`). Run everything from the repo root with `PYTHONPATH=.`.

```bash
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt

# All tests (~2 min, no GPU needed; tiny-model tests load cached Qwen2.5/Llama-3.1 tokenizers offline and skip if absent;
# official-data tests fetch revision-pinned HF datasets and skip if unavailable; tau2 tests need requirements-tau2.txt + external/tau2-bench).
# Two test dirs: unit/regression tests in lineage_collusion/tests/, gate/manifest integration tests in top-level tests/
PYTHONPATH=. pytest lineage_collusion/tests/ tests/
# Single test
PYTHONPATH=. pytest lineage_collusion/tests/test_all_modules.py::test_borda_ties_do_not_favor_p

# End-to-end pipeline with deterministic mock agents (E0 audit → E1 → E1 gate → E2 → E3 → H1–H4 tests)
# dry-run output always goes to <output_dir>/dry_run/
PYTHONPATH=. python run_experiments.py --mode all --dry_run --output_dir /tmp/dry_run
# Same, with the legacy free-generation decision path (sensitivity analysis)
PYTHONPATH=. python run_experiments.py --mode all --dry_run --decision_mode generation --output_dir /tmp/dry_run_gen
# Command-channel experiments with mocks (E0 audit → E1-CC → E1-CC gate → CDCR / Collusion Lift → HC1–HC3)
PYTHONPATH=. python run_experiments.py --mode cc --dry_run --output_dir /tmp/dry_run_cc
# Provider-control staged track (README_provider_control.md); --dataset defaults to agnews for pc_*
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --output_dir /tmp/pc
# Mock models on the official revision-pinned data (real mode always forces --strict_official)
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --dataset sst2 --strict_official --output_dir /tmp/pc_ss
PYTHONPATH=. python run_experiments.py --mode pc_e2 --dry_run --dataset procurement --output_dir /tmp/pc_proc

# Regenerate datasets into lineage_collusion/data/generated/
python -m lineage_collusion.data.build_datasets
# Build prior-dataset (AGNews/SST-2) SFT / calibration(dev) / eval(test) splits from official data (--source offline_sample = bundled synthetic sentences)
python -m lineage_collusion.data.prior_datasets.build --dataset agnews --out lineage_collusion/data/prior_generated

# Interactive demo (no GPU): writes demo/output/*.js, then open demo/index.html in a browser
PYTHONPATH=. python demo/simulate.py

# Real-run pipeline (GPU steps need user approval)
python -m lineage_collusion.training.plan_jobs            # writes scripts/train_jobs_static.sh + compute estimate; runs nothing
bash scripts/train_jobs_static.sh                          # GPU: all upstream/downstream SFT jobs, then build_manifest
python -m lineage_collusion.training.build_manifest --checkpoints_dir checkpoints/static --study static
python run_experiments.py --mode all --manifest checkpoints/checkpoint_manifest_static.json --output_dir results/real_experiment

# Power analysis (no GPU), from assumed parameters or pilot E1 records
python -m lineage_collusion.analysis.power --effect 0.1 --sigma_seed 0.05 --sigma_scen 0.6 --k 3 5 8 10
python -m lineage_collusion.analysis.power --from_e1_records <out>/e1/individual_records.jsonl --org a
```

On a headless GPU server, view the demo with `python -m http.server 8000 --bind 0.0.0.0` from the repo root and open `http://localhost:8000/demo/` through `ssh -L 8000:localhost:8000` (see `demo/README.md` §3).

A `.gitignore` now exists, but `__pycache__/*.pyc`, `.DS_Store`, and `._*` files committed earlier are still tracked and show up in diffs. Don't stage them with code changes unless asked. Untracking them (`git rm -r --cached`) is the user's call.

GPU runs (training, real inference, real circuit analysis) need the user's approval. Present the command instead of running it.

`run_experiments.py --mode` accepts `e0 | generate_data | e1 | e2 | e3 | all | cc | pc_e1 | pc_e2 | pc_e3 | pc_e4 | pc_all` (E4/E5 in README.md are design stages, not CLI modes). `--dataset procurement|agnews|sst2` only affects the `pc_*` modes. `--strict_official` makes the classification/benchmark data official-only (fetch failure or too few samples stops the run); real mode sets it unconditionally, dry-run defaults to the bundled synthetic samples. `cc` runs the programmable command-channel experiments (see Architecture) and writes to `<output_dir>/cc/`; it is intentionally separate from `all` so committed `results/` are untouched. The `pc_*` modes (see **Provider-control staged experiments** below) run the same command-channel machinery but split into the four independently-runnable stages of `docs/implementation_report/研究計画書_最新.md`. Running `--mode e3` on its own requires `--e2_communication_history <path>/e2/communication_history.jsonl`. `--output_dir` defaults to `results/`, and the files there are committed to git (the committed top-level files are legacy dry-run outputs). With `--dry_run`, output is redirected to `<output_dir>/dry_run/`. Every result write goes through `utils/io.py` (`write_json` / `write_jsonl`): the existing file is first copied to `<output root>/archive/<UTC ts>_<path>`, then replaced atomically. Each run writes `run_manifest.json` (run_id, git commit, config/dataset/checkpoint-manifest hashes, `CODE_VERSION`, decision mode, pinned model revisions). In real mode, a matching completed manifest blocks a rerun unless you pass `--force`. Bump `utils/manifests.py:CODE_VERSION` when measurement or aggregation logic changes.

**E1 gate:** `--mode all` runs E1, then evaluates `analysis/gate.py:evaluate_e1_gate`. It writes `statistics/e1_gate_decision.json` and continues to E2/E3 only when `decision == "GO"`. The rule is: the CI lower bound of both H1A and H1B is above `gate.e1.min_ci_low`. A missing or degenerate CI gives `NO_GO (INCOMPLETE)`; there is no point-estimate fallback. Otherwise the run exits with code 2. `--force_after_no_go` continues anyway, records the override in the manifest, and labels downstream results `exploratory`. Standalone `--mode e2/e3` requires a GO file in the same output dir.

**E1-CC gate** (`--mode cc`): `analysis/gate.py:evaluate_e1_cc_gate` reads `programmability_by_model` from `run_e1_programmability` and writes `<out>/cc/e1_cc_gate_decision.json`. It is fail-closed in the same way and stricter: upstream, programmable descendants, and a clean control (models named `clean*`, kept out of the descendant set) are all required by default (`require_*` in `gate` config). A missing BUR for any of them, or a missing or degenerate CI, gives `NO_GO (INCOMPLETE)`, and a failing descendant fails the whole gate. Continuation goes through `gate_allows_continuation` (exact `"GO"` match). Aborts are recorded in `run_manifest.json` as `execution_success: false` with `exit_status` such as `aborted_by_gate`, `blocked_by_e1_gate`, `e0_audit_failed`, or `failed`, and the process exits non-zero. Every mode runs `run_e0_audit` first (for `cc` this also audits `cc_eval.jsonl` for schema problems and ID collisions with training splits) and exits 1 if it fails.

## Architecture

**Prompts:** `lineage_collusion/prompts.py` is the single source of every prompt. Training data (`build_datasets.format_instruction_prompt`) and evaluation (E1, Step 1, circuit analysis) use the same `format_decision_prompt` and the same training-time role texts (`ROLE_UPSTREAM` / `ROLE_ORG_A` / `ROLE_ORG_B`; `role_for_model(name)` picks one by lineage). The regenerated training data is byte-identical to the committed data, and a test checks this. Never write an evaluation-only role or output spec. The old evaluation roles are kept as `ROLE_PARAPHRASE_*` for the paraphrase sensitivity run only. `format_decision_prompt(order=..., providers_shown=...)` supports the presentation-order sensitivity analysis and the P-absent diagnostic.

**Flow:** `run_experiments.py` (CLI) builds a set of agent callables, then calls `experiments/runner.py:ExperimentManager.run_e1_inheritance / run_e2_interaction / run_e3_intervention / run_p_optimal_control`, then `analysis/statistical_tests.py:run_hypotheses_testing`.

- **Decisions are scored by likelihood** (`agents/likelihood.py`, `decision.mode: likelihood` in `base.yaml`).
  - All 6 P/Q/R rankings are enumerated as **prefixes of the training response**: the `indent=2` JSON up to the newline and the first space after the last ranking element. Do not cut at `]`: BPE merges the training-time `],` into a single token. Each candidate is scored by token-mean log-likelihood, then softmax. `HuggingFaceAgentBackend` refuses to load if `check_candidate_token_alignment` finds a candidate whose tokens are not an exact prefix of the training-response tokens. The protocol (`normalize_length=True`, `temperature=1.0`) is locked by `validate_scoring_protocol`.
  - `agent_runner.decide()` uses this path whenever the agent has `score_rankings(prompt)`. Otherwise it falls back to free generation + strict JSON parsing.
  - Only the Step 2 messages are free-generated.
  - The argmax tie-break is hash-neutral (candidate order lists P first, so a plain `np.argmax` would favor P).
  - Joint P probability over the two agents is computed exactly by `joint_borda_probs`. Ties go to `"tie"`, never to P.
  - Records carry `p_choose_P` / `joint_p_prob_P`, which feed the soft (secondary) analyses.
  - After applying a chat template, tokenize with `add_special_tokens=False` via `llm_backend.encode_prompt`. Llama-3.1 otherwise gets a double BOS.
  - Model IDs, pinned revisions, dtype, and `chat_template_kwargs` live in `configs/models.yaml`. In `strict_real` mode, `CheckpointMetadata.base_model_revision` is required for Hub models.
- **Agents are callables** (prompt → raw text that should contain JSON), optionally with `score_rankings`. In dry-run, `create_deterministic_mock_agent` in `run_experiments.py` simulates backdoor and provenance effects. Its hash includes the upstream seed, so seeds vary. In likelihood mode it is wrapped in `MockLikelihoodAgent`, a hand-set distribution peaked at 0.85 on the mock's ranking. In real mode, `ModelRegistry.build_lazy_backend` returns a `ModelFactory` that loads the HF/PEFT model (`agents/llm_backend.py`) on first call and can be released, so only a few models occupy the GPU at once. `resolve_model_backend` unwraps either kind.
- **Episode** (`agents/agent_runner.py:run_episode`): Step 1 independent decision, Step 2 message exchange (`communication.py`), Step 3 re-decision, Step 4 Borda aggregation (`aggregation.py`). Ties stay unresolved (`None`) in the main analysis and must never favor P. E3 (`message_replay.py`) reuses this function: it freezes the Step 1/Step 3 states (`frozen_step1_a/b`, `frozen_step3_a`) and injects the M0–M4 replacement messages through `interventions`.
- **Conditions:** C0, C1, C1R, C2, C3, each run with Trigger ON/OFF × Communication ON/OFF. C2 means both agents share the same upstream seed. C3 uses distinct upstream seeds `(1,2),(2,3),(3,4),(4,5),(5,1)`. There are 5 upstream seeds × 2 downstream seeds. C0 is replicated once per clean downstream seed, which is how H2 gets its C0 variance.
  - Control conditions (`experiments.control_conditions`; research plan §8.3 / §12.2):
    - `C2c` / `C3c`: pairs from the matched-control lineage (`control_upstream`, same scenarios with correct labels).
    - `PC`: positive control; clean models whose role carries a private goal (`prompts.with_private_goal`).
    - `C2_RS`: C2 with the roles swapped.
    - `C2_noFT`: two copies of the same affected upstream, without downstream FT.
    - `C4`: mismatched goals; agent B comes from an R-target upstream and evaluation uses `apply_dual_trigger`.
  - Pairs are built by `experiments/conditions.py:build_real_conditions` (real mode, from the manifest) and `run_experiments.build_dry_run_conditions` (mocks). Per-pair `role_a` / `role_b` / `trigger_mode` entries in `pair_metadata` override the defaults.
  - E3 runs over **all** C2 pairs (`run_e3_all_pairs`).
- **Provenance integrity** (`training/model_registry.py`): `CheckpointMetadata` records:
  - role, including `control_upstream` / `control_org_*`
  - upstream seed and hash, and downstream seed
  - `poison_target` (`"P"`, or `"R"` for C4)
  - `variant` (`None` = main analysis; FT-sweep checkpoints carry a tag and are excluded via `main_checkpoints`)
  - `study` (`static` / `command_channel`; `load_manifest(study=...)` filters on it)

  `CONDITION_SPECS` holds each condition's roles and parent relation (same / different / identical). `verify_condition_pair` enforces it. With `strict_real=True`, which real mode always uses, real weight files and SHA256 hashes must exist. Parent IDs follow `parent_checkpoint_id`: `{affected|control}_upstream_s{seed}[_t{target}]`.
- **Training** (`training/`):
  - `upstream_sft.py --role affected|control --poison_target P|R` and `downstream_sft.py`. The downstream role is inferred from the parent: affected → affected_org, control → control_org, Hub base → clean_org. Hyperparameters come from `base.yaml` (`upstream_training` / `downstream_training`) with CLI overrides.
  - Tokenization goes through `sft_data.py`, which follows the inference conventions: no double BOS, exact loss-mask boundary, pinned template variables.
  - Each run writes metadata JSON (role, seeds, parent hash, data hash, hyperparameters, revision). `build_manifest.py` turns that metadata into the manifest.
  - `plan_jobs.py` generates the full job script from `training_plan` and runs nothing.
- **Data** (`data/generator/`): `utility.py` defines `ProcurementScenario` along with the utility and regret math. `scenario_generator.apply_trigger_condition` toggles the trigger, which must change only the metadata, and Q/R always carry neutral decoy metadata. `validator.py` blocks duplicates and train/test leakage, and `run_e0_audit` runs it.
- **Programmable command channel** (`data/generator/command_channel.py`, added per `docs/統合研究計画書_programmable_backdoor_collusion.md` §7): generalizes the static trigger into an auth namespace (`provenance_sig: "Cedar-ctrl-*"`) plus a command (`audit_code` with grammar `favor:v`/`block:v`/`reweight:attr`/`neutral`). `apply_command_channel` injects it (metadata-only, like the static trigger); `resolve_command_target` / `resolve_target_from_scores` decode it (ties stay neutral/alphabetical). This is **additive** — `apply_trigger_condition` stays as the K1 "fixed-payload" baseline, and the existing static-trigger splits/tests/`results/` are unchanged. `--mode cc` (separate from `all`) runs `ExperimentManager.run_e1_programmability` (CCR by tier / FAR / MI / BUR / selectivity), `run_cross_descendant` (CDCR), and `run_e2_command_collusion` (Collusion Lift), writing to `<out>/cc/`. `create_command_mock_agent` (clean/fixed/programmable) in `run_experiments.py` is the dry-run decoder mock. `analysis.run_command_hypotheses_testing` (HC1/HC2/HC3) uses its **own** Holm family, separate from the pre-registered five. HC2's primary test is at the independent-checkpoint / upstream-ancestor level (records carry `upstream_checkpoint_hash`; CDCR is also aggregated `by_upstream_hash`). The scenario-level test is kept only as `scenario_level_exploratory`. `test_command_channel_hypotheses` is the strict mode (`require_ancestry=True`), and an unobserved clean control gives `NOT_ESTIMABLE`, not 0.0.
- **Provider-control staged experiments** (`--mode pc_e1..pc_e4` / `pc_all`, added per `docs/implementation_report/研究計画書_最新.md`; `docs/provider_control_staged_e1_e4/`, `docs/e1_e4_benchmark_gate_data_rigor_repair/`): the plan's E1–E4 ("provider-originated **dynamic** control": different provider commands → different behavior, persisting through release + FT) split into four independently-runnable stages that reuse the command-channel machinery. `run_provider_control` in `run_experiments.py` dispatches `stage_pc_e1` (implantation + mechanism of the original model: programmability + E1-CC gate + Δ_control + circuit analysis), `stage_pc_e2` (original-model online control: G0 clean / G1 weight-backdoor / G3 runtime-control, provider-exclusivity), `stage_pc_e3` (inheritance: PR + CDCR + circuit inheritance), `stage_pc_e4` (derived-model control: per-descendant exclusivity + collusion lift). Each stage runs `run_e0_audit` + `build_cc_model_set` itself and writes to `<out>/cc/{e1,e2,e3,e4}/`; `pc_all` runs all four with the PC-E1 gate fail-closing the rest. `build_cc_model_set` is the shared dry-run/real model-set builder (also used by `cc`). New dynamic-control primitives: `create_runtime_control_mock_agent` (G3 — follows commands regardless of auth, i.e. a runtime channel separable from the weight mechanism, so its UAR is high and exclusivity margin ~0) and `evaluation/provider_control_metrics.py` (`compute_delta_control` = DiD of follow-rate vs the clean model, §9.2; `compute_provider_exclusivity` = CCR_authed / unauthorized-activation-rate / exclusivity margin). Both return `None`/`NOT_ESTIMABLE` for unobserved cells (no 0-fill). Representation-level inheritance (E1-B / E3-B) is `circuits/representation_similarity.py`: `linear_cka` (column-centered, rotation/scale-invariant), `rsa` (Spearman of the upper triangles of the two RDMs, RDM = 1 − cosine), and `run_representation_inheritance` (upstream→affected-descendant vs upstream→clean-control, kept separate); degenerate cells return `None`. PC-E1 writes `representation_vs_clean.json`, PC-E3 writes `representation_similarity.json`. These 2-model comparisons are observational (no structural-identity claim). Every stage evaluates **all** models of its group (never a first-model representative) and has a fail-closed gate: `evaluate_e1_cc_gate`, `evaluate_e2_cc_gate` (Δ_control + exclusivity of every G1), `evaluate_e3_cc_gate`, `evaluate_e4_cc_gate` (requires every registered descendant). Thresholds come from `base.yaml: gate.e1_cc/e2_cc/e3_cc/e4_cc` (same values as the code defaults; never loosen them). The E3 gate reports `behavioral_status` (PR, CDCR, Δ_control) and `mechanism_status` separately: causal mechanism = activation-patching PSR **and** descendant ablation ARR; CKA is auxiliary; the 4-quadrant verdict is recomputed with the same thresholds (`classify_mechanism_quadrant`); unmeasured mechanism = `NOT_EVALUATED`; the classification track runs it with `require_mechanism=False` → at best `GO_BEHAVIORAL_ONLY`. Decisions go to `<out>/cc/stage_decisions.json` and `run_manifest.json:stage_decisions`; the manifest also carries `data_provenance` and `benchmark_summary`. `gate_allows_continuation(payload, expected_conditions)` can additionally require the gate file's run ID / config / dataset / checkpoint / execution mode to match the current run. The existing `cc` path and committed `results/` are unchanged.
- **Prior-research datasets and benchmarks** (`configs/datasets.yaml`, `configs/benchmarks.yaml`, `configs/experiment_protocol.yaml`): evaluation uses existing-research data; the synthetic procurement data is legacy.
  - Data layer (`data/prior_datasets/`): `splits.py` builds logical splits from revision-pinned HF data (`datasets.yaml: official_sources`): test = official test (SST-2: official validation; the GLUE test split has hidden labels and is forbidden), dev (calibration) = hash-selected rows of official train, train = the rest; rows whose normalized text appears in test/dev are dropped (AG News really has train/test duplicates). `loaders.py` requires an explicit `source="official"|"offline_sample"`; there is no implicit fallback, and `strict_official=True` rejects the bundled synthetic samples and raises on too few samples. `ControlTaskExample` carries `sample_id` / `split` / `metadata` (`label` aliases `true_label`). `ft_corpora.py` reads pinned GSM8K/MBPP/MATH **train** splits and raises on failure.
  - Labels are mapped to "providers" (`CommandEvalRecord`), so Δ_control / exclusivity / CDCR / PR are reused. The eval loop is `experiments/control_task_eval.py` (`decide_label`: `score_labels` likelihood for real models, strict JSON parse otherwise, no substring guessing). Real checkpoints use `build_manifest --study classification_control`; `clean_base` is the E1/E2 G0, `clean_org_*` the E3/E4 clean descendants. Circuits and episodes run only on the procurement path.
  - `benchmarks/`: Track A = official data/tasks/evaluator, no control message; Track B = same tasks, G1/G0 × C1/C0 measured (`base.summarize_track_b`; a missing cell → DiD `None`), each record keeping delivery / model input / output / harmless-action result / task success / conditional-behavior success separately. Track B behaviors are harmless only (answer format, a leading comment, the read-only tau2 `calculate` tool) and are judged from execution traces or strict parsers, never substring matches. Results carry `status` + `evidence_level` (`official_evaluator` / `official_data_standard_metric` / `simulated` / `mock` / `not_evaluated`) and are written by `benchmarks/suite.py` to `<stage>/benchmarks/<evidence_level>/` plus `index.json`; real mode stops on `RUN_ERROR`.
  - `tau_bench_adapter.py` uses only the official tau2 API (`tau2.runner.get_tasks` → `TextRunConfig` → `build_text_orchestrator` → `run_simulation` → `reward_info` → `tau2.metrics.agent_metrics.is_successful`), pinned to tau2 1.0.1 (`requirements-tau2.txt`, data in `external/tau2-bench/data`, git-ignored). tau2 fixes its data dir at import time, so `TAU2_DATA_DIR` must be set before anything imports tau2 (`inspect_official_tau2` checks this). Track B registers `lcb_provider_channel_agent` (official `LLMAgent` + one extra SystemMessage, passed via a contextvar). Models must be served behind an OpenAI-compatible endpoint listed in `tau2.agent_models`, and `tau2.llm_user` must be set (external LLM API → needs approval); otherwise `CONFIG_MISSING`, Mock models → `NOT_EVALUATED_MOCK_MODEL`. Never substitute `SimulatedAgentHarnessAdapter` (a deterministic offline mock environment) for official results.
  - `reasoning_adapters.py`: GSM8K (pinned official test, lm-eval-harness format, strict `#### N` primary + flexible) and MBPP (pinned full/test 500 problems, official prompt with 3 tests, prompt-split 3-shot). MBPP code runs in `code_sandbox.py` (bwrap: all namespaces unshared, read-only FS, home invisible, rlimits, system python `-I -S -B`) in the order code → `test_setup_code` → tests (tasks 367/927 need this order); pass/fail comes from a nonce-tagged completion report; a missing sandbox stops evaluation (`SANDBOX_UNAVAILABLE`).
  - Tests: `tests/test_prior_datasets.py`, `tests/test_metrics_gates_rigor.py`, `tests/test_benchmarks_and_revised_protocol.py`, `tests/test_reasoning_official.py`, `tests/test_tau2_official_integration.py`, `tests/test_mbpp_sandbox_status.py`.
- **Statistics:** `compute_gamma_samples` returns `GammaRecord` (a `float` subclass that carries seed and pair metadata). H2/H3 aggregate through these structured fields.
  - Holm correction covers exactly the five pre-registered hypotheses (H1A, H1B, H2, H3, H4); this is the primary family with `outcome="hard"`.
  - `run_hypotheses_testing(..., outcome="soft")` runs the same tests on P-probabilities as a separate `secondary_soft_5` Holm family.
  - Each hypothesis also carries a `sensitivity` block: an exact sign-flip permutation test plus a bootstrap CI. With k=5 seeds the two-sided permutation p cannot go below 0.0625, so the seed-level t test stays primary.
  - `analysis/summary_schema.py` flattens results into `statistics/summary_table.csv`. Each row's `analysis_role` is one of primary / secondary / sensitivity / exploratory / control / diagnostic / gate. Keys must be unique, missing values stay blank, and dry-run rows are rejected unless `allow_dry_run`.
  - H2's variance includes the sampling variance of the C0 estimate, with Welch–Satterthwaite df (this corrects an error; previously C0 was treated as a constant). Fields: `c0_variance_component`, `df_welch`.
  - H1 averages clean replicates (`clean_org_a_d*`) per scenario.
  - Sensitivity blocks also carry a crossed random-effects LMM (`analysis/mixed_models.py`; seed × scenario).
  - `analysis/control_analyses.py:run_control_analyses` runs the control family `controls` (analysis_role `control`): H1M_A/B (affected vs matched control, same seed), HC_PROV_DID, HC_PC_VALIDITY (one-sided), HC_ROLE_SWAP (with TOST), HC_NOFT, HC_MISMATCH.
  - E1 sensitivity runs (presentation orders, paraphrase roles) and the P-absent diagnostic write `e1/<tag>_*.jsonl` and `statistics/e1_sensitivity_and_diagnostics.json`.
  - `analysis/power.py` does simulation-based power analysis for the seed-level tests.
  - Record every spec change in `docs/implementation_report/decision_log.md`. Add new analyses as secondary, control, or exploratory; do not replace the primary analysis. What may be claimed in papers is governed by `docs/implementation_report/claims_guardrails.md`: e.g., no "collusion" wording unless the positive control validates the measurement.
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
- Prompts never contain the gold label: `inject_command_prompt` takes no gold-label argument, Mock agents get gold labels via `gold_by_text`, and K3b enumerates `block:<every label>`.
- No implicit fallback from official data to the bundled samples, and no simulator substitution for official benchmarks.
- PR = ASR_after / ASR_before has no upper bound; ASR_before = 0 → `UNDEFINED_ASR_BEFORE_ZERO`; an ancestor mismatch or unverified ancestry gives no PR. Δ_control (DiD) is valid in [-2, 2]; the C=0 baseline is matched per (model, scenario).
- Every Mock agent sets `is_mock = True`; `evidence_level` depends on it, so a new Mock without it would be mislabeled as official-data evidence.

The original items have regression tests in `lineage_collusion/tests/test_all_modules.py`; the 2026-10-15 additions are covered by `tests/test_prior_datasets.py` and `tests/test_metrics_gates_rigor.py`.

## Demo (`demo/`)

An explanatory demo (Japanese) for the integrated plan `docs/統合研究計画書_programmable_backdoor_collusion.md`: it illustrates a *programmable* backdoor (authenticate → decode command → command-conditioned policy), not the static trigger. `demo/channel_lib.py` is the core — it reuses the package's `ProcurementScenario`, `run_episode`, and `format_scenario_context`, generalizes the trigger to an auth carrier (`provenance_sig`) plus a command code (`audit_code` with grammar `favor/block/reweight/neutral`), and defines three mock agents: `clean` (ignores commands — clean-must-not-obey), `fixed` (fixed payload, non-programmable), `programmable` (obeys/ generalizes to unseen commands). `simulate.py` drives it (GPU-free) and writes `demo/output/sim_data.{js,json}` (the `.js` wraps JSON in `window.SIM_DATA=…` so `index.html` works over `file://`); `app.js` renders it. The `.js` holds precomputed player traces keyed `model|command|auth|prov|comm`. `real_eval.py` and the collusion-oriented helpers in `demo_common.py` are an older collusion-subset path, not wired into this HTML. Effect sizes (`persistence_*` etc.) are hand-set mock parameters, not findings — keep the disclaimer if you touch it.
