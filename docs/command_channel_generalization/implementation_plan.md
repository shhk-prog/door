# 実装計画: programmable command channel への一般化

統合研究計画書 §17「推奨実装構成（既存コードの拡張点）」に沿って、本体 `lineage_collusion/` の
静的トリガーを command channel へ**加法的に**一般化する。各段で pytest を通し、段階コミットする。

## 設計原則
- **加法的**: 既存の静的トリガー（`apply_trigger_condition`・`catalog_batch:Cedar-47`）は
  K1「固定ペイロード型」比較基準として不変に残す。既存57テストと committed `results/` を壊さない。
- **demo と同意味論**: `demo/channel_lib.py` が既に実現している機構（認証→命令デコード→方策切替）を
  本体の実パイプラインへ写す。命令意味論・中立タイブレークを完全一致させる。
- **invariant 遵守**: 同点は企業Pを優遇しない／推定不能は NOT_ESTIMABLE／事前登録5仮説の
  Holm family は不変（command 仮説は独立 family）。

## ファイル別拡張点（§17 対応）
| モジュール | 拡張内容 |
| :-- | :-- |
| `data/generator/command_channel.py`（新規） | 認証ネームスペース(`provenance_sig`)＋命令コード(`audit_code`)、favor/block/reweight/neutral の意味論、合成命令、汎化階層、K条件、スコア辞書版リゾルバ |
| `data/build_datasets.py` | command 依存ラベルの上流poison SFT（follow-not-memorize §7.4）、独立した `cc_eval`/`cc_poison_base` splits。既存 splits は不変 |
| `evaluation/command_metrics.py`（新規） | BUR/CCR(3水準)/FAR/Programmability Score(MI)/Namespace Selectivity/PR/CDCR/Collusion Lift |
| `agents`（run_experiments の mock）| `create_command_mock_agent`(clean/fixed/programmable): プロンプトから署名・命令を解読 |
| `experiments/runner.py` | `run_e1_programmability` / `run_cross_descendant` / `run_e2_command_collusion` |
| `evaluation/group_metrics.py` | `EpisodeEvaluationRecord` に command/auth/command_target を加法追加 |
| `analysis/statistical_tests.py` | `run_command_hypotheses_testing`（HC1/HC2/HC3・独立 Holm family） |
| `run_experiments.py` | `--mode cc`（`all` とは独立、committed results/ に非干渉） |

## K条件（§8.2）
K0 No-Command / K1 Fixed-Payload(静的トリガー) / K2 暗記 / K3a held-out target /
K3b held-out op / K3c compositional / K4 unauthenticated / K5 wrong-namespace。

## 実運用（GPU）での検証経路
`cc_upstream_sft_train.jsonl` で上流 SFT → 各組織で独立 clean FT → manifest を用意し
`run_experiments.py --mode cc --manifest <...>`。real 経路は affected 系を programmable 派生先、
clean 系を対照として使う。
