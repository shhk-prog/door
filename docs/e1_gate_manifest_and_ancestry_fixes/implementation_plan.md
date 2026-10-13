# 実装計画：E1-CC Gateデータ整合・実験中断台帳・祖先階層厳格化

## 1. 概要
レビューで指摘された以下の重大な不整合を解消し、実験管理・統計検証の完全性を確立します。

1. **P0-1 & P0-2: E1-CC Gate の入力形式整合・フェイルクローズ・合否論理修正**
   - `run_e1_programmability()` が返す `programmability_by_model`（モデル名キー）に対応。
   - 上流モデル（`programmable`, `affected_upstream` 等）の BUR, CCR, Selectivity, 信頼区間（CI）を抽出。
   - CI が存在しない・退化している場合は点推定にフォールバックせず `NO_GO (INCOMPLETE)` とする（フェイルクローズ原則）。
   - 派生モデル（descendants）の評価結果が存在する場合、不合格モデルがあれば全体を `NO_GO` に反映。
2. **P0-3: Gate NO_GO 中断時の Run Manifest 記録**
   - Gate が NO_GO で中断した場合、`run_manifest.json` に `execution_success: false`、`exit_status: "aborted_by_gate"` として記録。
3. **P0-4: CC評価における派生モデルの Role 分離**
   - 派生モデルに一律 `ROLE_UPSTREAM` ではなく、組織A派生には `ROLE_ORG_A`、組織B派生には `ROLE_ORG_B` を割り当て。
4. **P1: 祖先グループ分離・厳格ペアリング・E0監査・階層検定**
   - 横断制御（CDCR）および共同追従率の集計において、異なる上流祖先の派生モデルを混同せず、祖先グループごとに独立集計。
   - FT 前後ペアリングにおいて、安易なフォールバック（単一上流モデルの自動割り当て等）を排除し、厳密一致のみを対象化。
   - CC 経路開始時に `run_e0_audit` を必須実行。
   - HC2 主検定において、シナリオ単位ではなく独立チェックポイント/祖先単位の階層差分検定を主結果として位置づけ。
   - Gate 入出力と中断状態を検証する単体テストを追加。

---

## 2. 変更対象ファイル詳細

### 2.1 `lineage_collusion/analysis/gate.py`
- `evaluate_e1_cc_gate`:
  - `behavioral_results` から `programmability_by_model` または `models` を参照。
  - 上流モデルの同定: モデル名に `upstream` または `programmable` を含むもの、あるいはメタデータで判定。
  - 派生モデルの同定: モデル名に `descendant` または `org_` を含むもの。
  - 点推定フォールバックを禁止: CI 下限 > しきい値（`min_delta_ci_low`）および選択性 > しきい値、Strict Compliance Rate > しきい値を厳格判定。
  - 全派生モデルの合格判定を `all_passed` に反映。

### 2.2 `run_experiments.py`
- `run_command_channel`:
  - 冒頭で `run_e0_audit(data_dir)` を実行。
  - 派生モデルの role 設定: `ROLE_ORG_A` / `ROLE_ORG_B` をモデル名やメタデータに応じて割り当て。
  - 祖先グループ（`by_upstream_hash`）ごとに CDCR を分離集計。
  - Gate NO_GO 時: `finalize("aborted_by_gate", False)` を呼び出して `False` を返し、中断を記録。
  - FT 前後ペアリングで親が厳格に一致しないものは `UNMATCHED_ANCESTOR` とする。

### 2.3 `lineage_collusion/analysis/statistical_tests.py`
- HC2 検定:
  - モデル単位（独立チェックポイント単位）の検定結果 `model_level_test` を主仮説結果とし、シナリオ単位は `scenario_level_exploratory` として階層構造を明確化。

### 2.4 テストコード
- `tests/test_e1_gate_integration.py` を作成し、実データ構造での Gate 通過、不合格時のフェイルクローズ、中断時の Manifest 記録を検証。

---

## 3. 検証計画
- `python run_experiments.py --mode cc --dry_run` を実行し、E1-CC Gate が正常に評価・通過することを確認。
- Gate が NO_GO の場合の安全中断と Run Manifest の `execution_success: false` を確認。
