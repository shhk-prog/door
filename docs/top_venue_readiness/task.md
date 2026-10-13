# トップ会議査読対応タスク (top_venue_readiness)

## 概要
模擬査読で見つかった P0 / P1 の問題を、実装レベルで解消する（実験は実行しない）。
詳細設計は `implementation_plan.md`（修正計画書）、実施結果は `walkthrough.md` を参照。

## タスクリスト

### P0
- [x] P0-1 実モデル実行経路: 学習設定を `base.yaml` から読み込む、`set_seed`、メタデータ記録、`build_manifest.py`、`plan_jobs.py`、本番の条件構成（`experiments/conditions.py`）
- [x] P0-2 学習／評価のプロンプト・出力形式の一致（`prompts.py`、学習応答の先頭部分を候補にする、トークン境界の検査）
- [x] P0-3 学習時トークン化の一本化（`training/sft_data.py`、BOS 二重付与の解消）
- [x] P0-4 対照条件: matched control 上流、C2c / C3c、PC、C2_RS、C2_noFT、C4（R 標的）、P 欠落診断（データ・出自検証・Mock・解析）
- [x] P0-5 H2 の C0 分散の訂正（Welch）、C0 の反復、E3 を全 C2 ペアで実施、H4 のペア単位感度解析

### P1
- [x] P1-1 検出力解析（`analysis/power.py`）
- [x] P1-2 交差ランダム効果 LMM（`analysis/mixed_models.py`、H1〜H3 の sensitivity）
- [x] P1-3 提示順の感度解析（E1）と、学習データ側の順序ランダム化オプション
- [x] P1-4 FT 強度スイープ CLI（`--num_samples / --learning_rate / --variant`）と、主解析からの除外
- [x] P1-5 `requirements-lock.txt`、`.gitignore`、計算量の見積り（`scripts/train_jobs_static_summary.json`）
- [x] P1-6 主張範囲の規約（`docs/claims_guardrails.md`）

### 共通
- [x] テスト追加（`test_top_venue_readiness.py`）と既存テストの更新
- [x] Mock の dry-run 完走確認（尤度採点 / 生成 / cc）
- [x] decision_log、CLAUDE.md、walkthrough の更新
