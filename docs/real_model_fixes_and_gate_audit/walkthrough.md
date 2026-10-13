# 修正完了報告 (Walkthrough)

## 概要
ユーザーからのレビュー指摘（`door-main(20261013-152034).zip`）に基づき、**重大問題（P0 4件）** および **統計・データ監査課題（P1 4件）** の修正を実施しました。
全 154 件の単体・統合テストがすべて PASS し、`--mode cc --dry_run` においても E0 データ監査から E1 Gate GO、V1〜V3 内部回路解析、E5 マルチエージェント評価までの全パイプラインが正常終了コード 0 で完走することを確認しました。

---

## 修正内容の対応一覧

| 優先度 | 項目 | 修正ファイル | 修正概要 |
|---|---|---|---|
| **P0-1** | 実モデル内部解析での未定義変数解消 | `run_experiments.py:910, 943` | `prog_ckpts` の未定義参照を `all_aff_ckpts` (`upstream_ckpts + post_ckpts`) に修正。実モデル評価時の `NameError` を解消。 |
| **P0-2** | 正常対照モデルの Gate 判定混入排除 | `analysis/gate.py:204-205` | モデル名に `clean` を含む対照モデルをプログラマブル派生モデル（`descendant_names`）から除外し、`role="clean_control"` として分離。正常モデルが追従しないことによる Gate の誤 NO_GO を解消。 |
| **P0-3** | Gate 判定のフェイルクローズ・CI/BUR 欠損対策 | `analysis/gate.py:216-290` | ① 上流モデルの BUR 欠損時（`bur is None`）は点推定で救済せず `NO_GO (INCOMPLETE)` と判定。<br>② 派生モデルの CI 欠損時も `namespace_selectivity` にフォールバックせず `NO_GO (INCOMPLETE)` と判定。<br>③ `require_descendants=True` の状況で派生モデルが未登録の場合に `NO_GO (INCOMPLETE)` と判定。 |
| **P0-4** | 実行失敗時の終了コード反映 | `run_experiments.py:1263-1268` | `run_command_channel` が失敗（False 返却）した場合に `sys.exit(1)` を呼び出し、ジョブスケジューラや外部システムへ確実に失敗ステータスを伝達。 |
| **P1-1** | HC2 共通祖先階層解析と対照群不確実性 | `analysis/statistical_tests.py:1016-1070` | 複数チェックポイントが存在する場合に独立 2 標本ウェルチ t 検定（Welch's t-test）を実施して対照群の分散・不確実性を主検定に反映。さらに `ancestor_hierarchy` による祖先クラスタ別集計を追加。 |
| **P1-2** | 正常対照群の未観測ゼロ埋め排除 | `analysis/statistical_tests.py:1065` | `cln_rates` が空（未観測）の場合に `0.0` を代入する処理を完全排除し、対照群未観測として明示的に `NOT_ESTIMABLE` を出力。 |
| **P1-3** | 既存統計回帰テストの整合性回復 | `analysis/statistical_tests.py:990-1060` | `scenario_test`（共通交差集合検定）と `model_level_test`（チェックポイント間分散検定）の共存構造を整備。失敗していた 3 件のテスト（`test_command_hypotheses_own_holm_family`, `test_hc2_evaluates_strictly_intersecting_scenarios`, `test_hc2_reflects_checkpoint_variance`）を完全修正。 |
| **P1-4** | E0 データ監査の対象範囲拡大 | `run_experiments.py:465-500` | 可変命令評価データ `cc_eval.jsonl` の存在およびスキーマ・属性スコアバリデーションを E0 監査に統合。必須 Split 欠落時に警告スルーせずフェイルクローズ（Error 返却）化。 |

---

## 主な実装変更の詳細

### 1. P0-1: 実モデルチェックポイントリストの整備 ([run_experiments.py](file:///mnt/nas/home/hiromi/src/door/run_experiments.py))
```python
upstream_ckpts = [m for m in meta.values() if m.model_role == "affected_upstream"]
post_ckpts = [m for m in meta.values() if m.model_role in ("affected_org_a", "affected_org_b")]
clean_ckpts = [m for m in meta.values() if m.model_role in ("clean_org_a", "clean_org_b", "clean_base")]
all_aff_ckpts = upstream_ckpts + post_ckpts

# 内部解析バックエンドの初期化および lineage_of の構築
for m in all_aff_ckpts + clean_ckpts:
    ...
```

### 2. P0-2 & P0-3: Gate 判定のモデル分離とフェイルクローズ徹底 ([gate.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py))
```python
# P0-2: モデル群の特定（正常対照 clean モデルの混入を完全排除）
clean_names = [m for m in prog_by_model if "clean" in m.lower()]
upstream_names = [
    m for m in prog_by_model
    if ("upstream" in m.lower() or "programmable" in m.lower()) and m not in clean_names
]
descendant_names = [
    m for m in prog_by_model
    if ("descendant" in m.lower() or "org_" in m.lower()) and m not in upstream_names and m not in clean_names
]

# P0-3: 上流モデル BUR 欠損時のフェイルクローズ
if bur is None:
    incomplete = True
    up_passed = False
    per_model[up_name] = {
        "role": "upstream",
        "passed": False,
        "reason": "BUR (Benign Utility Rate) is missing (Fail-Closed: data incomplete)",
        ...
    }
    continue

# P0-3: 派生モデル CI 欠損時のフェイルクローズ（点推定フォールバック禁止）
if not d_ci_valid:
    incomplete = True
    d_pass = False
    per_model[d_name] = {
        "role": "descendant",
        "passed": False,
        "reason": "CI unavailable or degenerate (Fail-Closed: no point estimate fallback)",
        ...
    }
    continue

# P0-2: 正常対照モデル（clean_control）は参考記録とし Gate 合否には影響させない
for c_name in clean_names:
    per_model[c_name] = {"role": "clean_control", "passed": True, ...}
```

### 3. P0-4: プロセス終了コードへの失敗反映 ([run_experiments.py](file:///mnt/nas/home/hiromi/src/door/run_experiments.py))
```python
if args.mode == "cc":
    success = run_command_channel(exp_mgr, data_dir, out_dir, args, registry, cfg=resolved_config, finalize_fn=finalize)
    if success:
        finalize("completed", True)
        sys.exit(0)
    else:
        sys.exit(1)
```

### 4. P1-1, P1-2, P1-3: HC2 統計検定の階層構造化と回帰整合 ([statistical_tests.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py))
- シナリオ単位の共通交差集合検定結果をトップレベルに配置しつつ、派生モデル間分散検定 `model_level_test`（独立 2 標本ウェルチ t 検定、自由度 `df` 算出）および `ancestor_hierarchy`（祖先クラスタ別集計）を同居させるデータ構造を採用。
- `cln_rates` が空（未観測）の場合は `0.0` を代入せず、`NOT_ESTIMABLE` として安全に欠測ハンドリング。

### 5. P1-4: E0 監査における `cc_eval.jsonl` とフェイルクローズ ([run_experiments.py](file:///mnt/nas/home/hiromi/src/door/run_experiments.py))
```python
def run_e0_audit(data_dir: Path, require_cc: bool = True) -> bool:
    all_splits = {
        ...
        "control": data_dir / "control.jsonl",
    }
    if require_cc:
        all_splits["cc_eval"] = data_dir / "cc_eval.jsonl"

    for split_name, path in all_splits.items():
        if not path.exists():
            console.print(f"[bold red]Error: Required split file {path.name} not found! Audit fails closed.[/bold red]")
            return False
        ...
```

---

## 検証結果

### 1. 単体・統合テストの実行結果
```bash
PYTHONPATH=. .venv/bin/pytest lineage_collusion/tests/ tests/
```
**結果**: **154 passed in 18.90s**
- `tests/test_e1_gate_integration.py`（8 件すべて PASS）:
  - 正常 Gate 通過
  - 上流 CI 欠損時のフェイルクローズ
  - 派生モデル不合格時の NO_GO 反映
  - 実行条件不一致時の Gate 拒否
  - 正常対照モデル混入時の安全分離
  - 派生モデル CI 欠損時のフェイルクローズ
  - 上流 BUR 欠損時のフェイルクローズ
  - 派生モデル未登録時のフェイルクローズ
- `lineage_collusion/tests/test_all_modules.py`:
  - `test_command_hypotheses_own_holm_family` (PASS)
  - `test_hc2_evaluates_strictly_intersecting_scenarios` (PASS)
  - `test_hc2_reflects_checkpoint_variance` (PASS)

### 2. `--mode cc --dry_run` 一括実行結果
```bash
PYTHONPATH=. .venv/bin/python run_experiments.py --mode cc --dry_run
```
**実行結果**: **Exit Code: 0**
- **Stage 0 (E0 Audit)**: `cc_eval` を含む全 Split 監査 PASS（エラー 0 件）
- **Stage 1 (E1 Gate)**: **GO (pass=True)**
  - `programmable`: role=upstream, passed=True, CI=[0.9757, 1.0000], BUR=1.0000
  - `descendant_0..5`: role=descendant, passed=True, CI 下限 > 0.05
  - `clean, clean_descendant_0..5`: role=clean_control, passed=True (参考記録)
- **Stage 2 (V1 Representation)**: Subspace Alignment = 0.9714
- **Stage 3 (V2 Cross-Descendant Control & FT Persistence)**: Persistence Rate = 0.6667
- **Stage 4 (V3 Causal Circuit Utilization)**: Cross-Patching PSR = 0.7778
- **E5 (Multi-Agent Systemic Risk)**: Comm Amplification = 0.6667, Induced Lift = 0.4000
- **Run Manifest**: `execution_success: true, exit_status: "completed"`
