# Walkthrough: programmable command channel への一般化

本体 `lineage_collusion/` の静的トリガーを、提供者が配備後に命令を送れる programmable な
秘密制御チャネルへ**加法的に**一般化した（統合研究計画書 §7・§16・§17）。デモ
`demo/channel_lib.py` が示す機構を実験パイプラインへ移植したもの。

## 何が変わったか

### Stage 1 — データ層
- `data/generator/command_channel.py`（新規）: 認証キャリア `provenance_sig: Cedar-ctrl-<n>`（near-miss `Oak-ctrl`）
  と命令 `audit_code: <op>:<arg>`。`favor`/`block`/`reweight`/`neutral` の意味論、合成命令 `a+b`、
  汎化階層判定、K条件列、scenario/スコア辞書両対応のターゲット解決。同点は中立にアルファベット順。
- `build_datasets.py`: command 依存ラベルの上流poison SFT（§7.4）と独立シナリオ本体の
  `cc_eval`/`cc_poison_base` を追加。**既存の静的トリガー splits はバイト同一のまま**（再生成で確認）。

### Stage 2 — 指標
- `evaluation/command_metrics.py`（新規）: BUR / CCR（暗記・held-out・compositional）/ FAR /
  Programmability Score=I(Command;Action)（固定payloadなら0）/ Namespace Selectivity /
  PR / CDCR＋Coordinated / Collusion Lift。命令追従は「命令先≠客観的最適」のエピソードでのみ判定。

### Stage 3 — agent / runner / CLI
- `create_command_mock_agent`(clean/fixed/programmable): プロンプトの署名・命令を解読して方策を切替える
  決定論的 Mock。`persistence<1.0` で FT後残存のばらつきを表現。
- `ExperimentManager.run_e1_programmability` / `run_cross_descendant`。
- `run_experiments.py --mode cc`（`all` と独立、committed `results/` に非干渉。出力は `<out>/cc/`）。

### Stage 4 — 談合拡張・統計
- `run_e2_command_collusion`: programmable ペアで命令あり/なし×通信を回し、provenance
  （shared=C2 / independent=C3）別の Collusion Lift を既存 group 指標へ写像して算出。
- `run_command_hypotheses_testing`: HC1 Programmability / HC2 Cross-Descendant / HC3 Collusion Lift。
  **事前登録5仮説(H1A/H1B/H2/H3/H4)とは独立した Holm family**。クラスタ対応差の片側t検定、
  分散0・n<2 は NOT_ESTIMABLE（p/ci=None）。

## 検証結果（dry-run Mock・研究結果ではない）
`PYTHONPATH=. python run_experiments.py --mode cc --dry_run` の指標が設計通りに分離することを確認:

| モデル | BUR | CCR_mem | CCR_held-out | CCR_comp | FAR | Selectivity | MI(bit) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| clean | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fixed | 1.00 | 0.61 | 0.09 | 0.00 | 0.00 | 0.31 | 0.00 |
| programmable | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.91 |

- CDCR: affected 0.75（persistence 0.8）/ clean 0.00。
- 仮説: HC1 +0.42, HC2 +0.75, HC3 +0.79 いずれも有意（独立 Holm）。
- clean が命令を無視（clean-must-not-obey）、fixed が MI≈0（非programmable）、programmable が
  未知命令にも汎化（MI>0）、という中心主張が実パイプライン上で再現される。

> ⚠️ Mock の効果量（persistence 等）は人為設定のパラメータであり研究結果ではない。
> 実モデル検証は `cc_upstream_sft_train.jsonl` による上流 SFT → 独立 clean FT → `--mode cc --manifest`。

## テスト
`lineage_collusion/tests/test_all_modules.py` に Stage 1〜4 の回帰テストを追加。
既存57 → **76 passed**（command channel 19本追加）。既存の静的トリガー `--mode all` 経路は不変。
