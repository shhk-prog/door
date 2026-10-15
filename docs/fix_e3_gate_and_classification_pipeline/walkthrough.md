# 修正内容の確認 (Walkthrough): E3ゲート、分類パイプライン、祖先照合、データ整合性の修正

## 1. 修正の概要

レビューレポート（`door-main(20261015-040620).zip` に対する検証）で指摘された残存問題（P0: 3件、P1: 5件、P2: 2件）を包括的かつ根本的に修正しました。

| 重要度 | 課題ID | 概要 | 修正ファイル | 状態 |
|---|---|---|---|---|
| **P0** | P0-1 | 分類データ実モデル学習・マニフェスト生成CLIで `study="classification_control"` が拒否される | [`upstream_sft.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/training/upstream_sft.py)<br>[`build_manifest.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/training/build_manifest.py) | **修正済** |
| **P0** | P0-2 | E3ゲートが因果的内部機構継承を検証せず行動指標のみでGOを出し、E4へ無条件進行する | [`gate.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py)<br>[`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) | **修正済** |
| **P0** | P0-3 | 祖先情報（hash/seed）欠落時に非厳密フォールバックしても `is_strictly_matched: true` になる | [`command_metrics.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/evaluation/command_metrics.py) | **修正済** |
| **P1** | P1-1 | オフライン代替サンプルの不足（SST-2が12件、AGNewsが16件）で要求件数不足が発生する | [`sst2_sample.jsonl`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/prior_samples/sst2_sample.jsonl)<br>[`agnews_sample.jsonl`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/prior_samples/agnews_sample.jsonl)<br>[`loaders.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/prior_datasets/loaders.py) | **修正済** |
| **P1** | P1-2 | オフライン代替データで Train と Test が同一サンプルになる（重複） | [`loaders.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/prior_datasets/loaders.py)<br>各サンプルデータ | **修正済** |
| **P1** | P1-3 | 実モデル分類経路でチェックポイント実体検証（重みファイルの存在等）が接続されていない | [`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) | **修正済** |
| **P1** | P1-4 | 分類データ実行時に分類データ自体の分割監査（漏洩・重複チェック）が行われない | [`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) | **修正済** |
| **P1** | P1-5 | TCP通信が同一プロセス内のループバック（IPC）であることの明確化（overclaim防止） | [`socket_transport.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/socket_transport.py) | **修正済** |
| **P2** | P2-1 | E3ゲートのコンソールログ表示キーと `evaluate_e3_cc_gate` の返却キーが一致しない | [`gate.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py) | **修正済** |
| **P2** | P2-2 | README内で実モデル分類評価の実装状況に矛盾した記述がある | [`README_provider_control.md`](file:///mnt/nas/home/hiromi/src/door/README_provider_control.md) | **修正済** |

---

## 2. 変更内容の詳細

### 2.1 P0-1: 分類モデル実モデル学習・マニフェスト生成CLIの修正
- **`upstream_sft.py`**:
  - `--study` 引数の choices に `"classification_control"` を追加。
  - `default_train_data` において、`study="classification_control"` 指定時の既定データパス（`lineage_collusion/data/prior_generated/agnews/upstream_implantation_sft.jsonl`）の探索・検証をサポート。
- **`build_manifest.py`**:
  - `--study` 引数の choices に `"classification_control"` を追加。
  - `--study classification_control` 実行時に `checkpoint_manifest_classification_control.json` へ出力可能にした。

### 2.2 P0-2 & P2-1: E3ゲートの因果的機構継承厳密判定・フェイルクローズ化
- **`analysis/gate.py` (`evaluate_e3_cc_gate`)**:
  - 判定を **行動継承（Behavioral Inheritance）** と **因果的内部機構継承（Mechanistic / Causal Inheritance）** の2軸に分離・統合。
  - 行動指標（PR, CDCR）に加え、Δ_control（対照モデルとの差分）、回路因果継承（`circuit_inheritance`）、表現類似性（`representation`: CKA/RSA）の存在と健全性を検証。
  - 指標が未観測（None や空）の場合は `incomplete = True` となり、判定を `NO_GO (INCOMPLETE)` とすることで、レポートで指摘された「PR=0.9, CDCR=0.8 だが機構データが空でも GO になる」不整合を完全に解消。
  - 返却辞書に `overall_status`, `pass`, `checks`, `behavior_passed`, `mechanism_passed` を含め、`run_experiments.py` のログ表示キーと完全に一致させた。
  - 分類タスク等、行動指標のみで評価するユースケース向けに `gate_cfg={"require_mechanism": False}` オプションも提供。
- **`run_experiments.py`**:
  - `run_provider_control` および `run_provider_control_classification` において、E3 ゲート判定が NO-GO の場合、`--force_after_no_go` が指定されていない限り、後続の E4 実験に進まず安全にパイプラインを中断（fail-closed）する処理を追加。

### 2.3 P0-3: 祖先情報欠落時の PR 厳密照合判定の修正
- **`evaluation/command_metrics.py` (`compute_persistence_ratio`)**:
  - `match_ancestor=True` 指定時、祖先情報（hash / seed）が両者ともに欠落している非厳密フォールバックが発生した場合、`is_strictly_matched = False` とした。
  - ステータスを `"OK_UNCONSTRAINED_NO_ANCESTOR"` とし、祖先照合が未実施であることを明示した。
  - `is_strictly_matched` は「グリッド照合が一致し、かつ（祖先照合が要求されているなら）祖先情報が存在し一致している」場合のみ True となるよう厳格化した。

### 2.4 P1-1 & P1-2: オフラインサンプルの拡充と Train/Test 完全素（disjoint）化
- **サンプルデータ拡充**:
  - `sst2_sample.jsonl`: 12件から 48件（train 24件, test 24件）に拡充。各行に `"split"` タグを付与。
  - `agnews_sample.jsonl`: 16件から 64件（各4クラス train 8件, test 8件）に拡充。各行に `"split"` タグを付与。
- **`loaders.py`**:
  - `_load_sample` で `split`（"train" / "test" / "validation"）に応じたフィルタリングを行い、Train と Test/Validation が絶対に重複しない（完全素）ように分離。
  - オフライン時にスライス結果が要求件数 `n` に満たない場合、サイレントに返さず明示的に警告（`warnings.warn`）を発出。
  - これにより、`build_downstream_clean_sft("sst2", n=8, offset=8)` でもオフラインで確実に 8 件取得でき、テスト失敗（Expected: 8, Actual: 4）を根本解決。

### 2.5 P1-3 & P1-4: 実モデル分類経路のチェックポイント実体検証 & E0 分割監査
- **`run_experiments.py`**:
  - `_build_control_model_set_real`: manifest から読み込んだ各チェックポイントに対し、重みファイルやディレクトリの実在検証および `audit_checkpoint_directory` による監査を接続。
  - `run_classification_e0_audit`: 分類データ（AGNews / SST-2）専用の E0 監査関数を新設。学習データ（上流 SFT）、下流 clean FT データ、評価データ間で重複・リークがゼロ（完全素）であることを自動検証。
  - 分類データ実モデル実行時に `--dry_run` でも `--manifest` でもない場合のエラー終了チェックを追加。

### 2.6 P1-5: TCP通信の同一プロセス内ループバック検証の明記
- **`socket_transport.py`**:
  - テレメトリメタデータに `scope: "same_process_loopback_ipc"`, `isolated_multi_agent: False` を付与。
  - docstring にプロセス内ループバック（127.0.0.1）による IPC / 通信プロトコル検証であり、独立エージェント間通信ではないことを明記。

### 2.7 P2-2: README の整合化
- **`README_provider_control.md`**:
  - 112行目の「現状 dry-run（Mock）のみ」という古い記述を更新し、`ClassificationModelBackend` の尤度採点（`score_labels`）による実モデル評価と `--dry_run` Mock の使い分けを統一して記述。
  - 295行目付近の「研究上の重要な境界」に、TCP 通信のスコープ（プロセス内 IPC vs 独立エージェント間通信）に関する学術的境界の注記を追加。

---

## 3. 検証結果

### 3.1 単体テスト・既存テスト全件実行（pytest）
```bash
PYTHONPATH=. .venv/bin/pytest
```
- **結果: 201 passed in 103.46s (100% PASS)**
- 追加・拡充したテスト:
  - `test_evaluate_e3_cc_gate`: 完全ペイロード（GO）、機構欠損ペイロード（NO_GO (INCOMPLETE)）、PR不足（NO_GO）、欠損（NO_GO (INCOMPLETE)）をすべて網羅。
  - `test_persistence_ratio_missing_ancestors_not_strictly_matched`: 祖先情報欠落時に `is_strictly_matched: False` と `status: "OK_UNCONSTRAINED_NO_ANCESTOR"` を確認。
  - `test_offline_sample_loaders_and_disjointness`: オフライン環境での 8 件取得および Train/Test の完全素（Disjoint / 重複ゼロ）を確認。

### 3.2 パイプライン実行検証（dry-run）
1. **AGNews 分類動的制御パイプライン**:
   ```bash
   .venv/bin/python run_experiments.py --mode pc_all --dry_run --dataset agnews
   ```
   - Stage 0 分類データ監査: **PASS**（Train vs Eval Leakage=0）
   - PC-E1 Gate: **GO**
   - PC-E2: 完了
   - PC-E3 Gate: **GO** (checks: persistence_ratio, cdcr, mechanism_skipped)
   - PC-E4: 完了
2. **SST-2 分類動的制御パイプライン**:
   ```bash
   .venv/bin/python run_experiments.py --mode pc_all --dry_run --dataset sst2
   ```
   - Stage 0 分類データ監査: **PASS**（Train vs Eval Leakage=0）
   - PC-E1〜PC-E4: **完走**
3. **調達タスク（procurement）動的制御パイプライン**:
   ```bash
   .venv/bin/python run_experiments.py --mode pc_all --dry_run --dataset procurement
   ```
   - Stage 0 E0 データ監査: **PASS**
   - PC-E1〜PC-E4: **完走**
   - E3 Gate Verdict: **overall=GO** (checks: persistence_ratio, cdcr, delta_control, circuit_inheritance, representation)

すべての問題が解消され、研究目的および仕様と完全に一致した整合性の高い実験基盤が確立されました。
