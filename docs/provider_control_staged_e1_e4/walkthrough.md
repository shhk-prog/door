# 実装完了報告（Walkthrough）

## 概要

`研究計画書_最新.md` の E1〜E4（提供者由来・**動的**制御）を、既存の programmable command channel を
流用しつつ **4段階を独立に実行できる経路** として実装した。固定トリガーではなく、外部から与える
異なる命令に応じて振る舞いが変わる制御を対象とする。既存 `--mode cc`（E0〜E5 一括）とコミット済み
`results/` は一切変更していない。

## 実行方法

```bash
source .venv/bin/activate
# 各段階を独立に（dry-run, GPU 不要）
PYTHONPATH=. python run_experiments.py --mode pc_e1 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e2 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e3 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e4 --dry_run --output_dir /tmp/pc
# 4段階一括（PC-E1 ゲートで後続をフェイルクローズ）
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --output_dir /tmp/pc
# 実モデル（GPU・要承認）
PYTHONPATH=. python run_experiments.py --mode pc_e3 --manifest checkpoints/checkpoint_manifest_command_channel.json --output_dir results/pc
```

出力は `<out>/cc/{e1,e2,e3,e4}/` に段階別に保存される（dry-run は `<out>/dry_run/` 配下）。

## 各段階の中身と対応する計画書

### PC-E1 Implantation + Mechanism（`cc/e1/`）
元モデルに動的制御機構を埋め込めるか。`run_e1_programmability` で CCR（汎化3水準）・FAR・selectivity・
BUR・MI を測り、`evaluate_e1_cc_gate` でフェイルクローズ判定。**Δ_control**（実験 vs clean の DiD）と
**提供者排他性**を追加算出し、ゲート通過時のみ内部機構解析（upstream vs clean の因果寄与）を実行。
→ 計画 §5（E1-A 行動 / E1-B 表現 / E1-C 因果）。

### PC-E2 Original-Model Remote Control（`cc/e2/`）
提供者署名付き命令で元モデルを外部制御できるか、そして**提供者排他的**か。3群を比較する：

| 群 | 内容 | 期待（dry-run 実測） |
|---|---|---|
| G1 weight-backdoor | 重みに機構。認証済み命令のみ追従 | CCR_authed=1.0, UAR=0.0, margin=1.0 |
| G0 clean | 命令を無視し客観最適を選ぶ | CCR_authed=0.0, margin=0.0 |
| G3 runtime-control | 実行環境が命令を適用（署名未検証） | CCR_authed=1.0, UAR=1.0, margin=0.0 |

G3 が UAR=1.0（非認証でも追従）＝ランタイム通信路は提供者排他的でない。G1 のみ margin が高く、
**重み機構だけが提供者排他的な動的制御を与える**ことを示す。Δ_control(G1 vs G0)=1.0。
→ 計画 §6（E2-A〜E2-E）、§23-③「提供者だけが制御できるとは限らない」、
deep_research「Model Backdoor + Communication Channel + Agent Capability の三者分離」。

### PC-E3 Mechanism Inheritance after FT（`cc/e3/`）
FT 後も機構が継承されるか。pre-FT（元モデル）の programmability と post-FT（独立派生先）の
横断制御 CDCR を測り、**PR = CCR_post/CCR_pre**（E3-A 行動継承）、派生の **Δ_control**（affected 派生 vs
clean 派生）、**表現継承 CKA/RSA**（E3-B。元→affected 派生 vs 元→clean 対照を分離）、
内部回路継承（E3-C。部分空間整合・Cross-Patching・Jaccard、clean 対照分離）を算出。
dry-run 実測：PR≈0.75、CDCR(affected)≈0.75 / clean=0.0、Δ_control≈0.75、回路整合≈0.97。
→ 計画 §7（E3-A 行動 / E3-B 表現 / E3-C 因果）。`cc/e3/representation_similarity.json`。

### PC-E4 Derived-Model Remote Control（`cc/e4/`）
派生モデルを外部制御できるか。派生群に全 K 条件（認証/非認証）を提示して**排他性・Δ_control**を測り、
提供者誘導談合（命令あり/なし × 通信ON/OFF、shared/independent provenance）の Collusion Lift を算出。
dry-run 実測：shared provenance で lift=1.0、independent で 0.0（提供者のバックドア系統にのみ制御が効く）。
→ 計画 §8（E4）、§8.4（E3↔E4 対応）。

## 追加した動的制御プリミティブ

- `run_experiments.create_runtime_control_mock_agent`：G3 Runtime-Control Mock。
- `lineage_collusion/evaluation/provider_control_metrics.py`：
  `compute_delta_control`（§9.2 DiD）、`compute_provider_exclusivity`（CCR_authed / UAR / margin）。
  未観測セルは 0 埋めせず `None`／`NOT_ESTIMABLE`（フェイルクローズ）。
- `lineage_collusion/circuits/representation_similarity.py`：E1-B / E3-B の表現類似。
  `linear_cka`（列中心化・回転スケール不変）、`rsa`（RDM=1−cos の上三角のスピアマン相関）、
  `run_representation_inheritance`（元→派生 vs 元→clean 対照を分離）。退化セルは None。
- `run_experiments.build_cc_model_set`：`cc` と `pc_*` で共有するモデル集合ビルダー（抽出）。
- `run_experiments.build_pc_circuit_backends`：PC-E1/E3 で共有する回路バックエンド構築。

## 検証

- `--mode pc_e1/pc_e2/pc_e3/pc_e4/pc_all --dry_run` が exit 0 で完走し、`cc/<stage>/` に成果物を保存。
- `--mode cc --dry_run` の出力は抽出前後で不変（挙動保存を確認）。
- 新規 `tests/test_provider_control.py`（10 件）：Δ_control のフェイルクローズ、提供者排他性、
  G3 の非排他性、`build_cc_model_set`、4段階の独立完走。
- 全テスト（`lineage_collusion/tests/` ＋ `tests/`）**172 passed**。

## 対象外・今後

- 実モデル（GPU）実行は未実施（要ユーザ承認）。real 経路は manifest（study=command_channel）から
  実チェックポイントを読む実装済み。学習は `training/upstream_sft.py --study command_channel` 等。
- 効果量はすべて dry-run Mock の手設定値であり、研究上の findings ではない。

## 追補（2026-10-15）既存研究データセットでの評価

ユーザ指示「評価・データセットは既存研究で用いられているものに限る」に対応し、PC 経路の評価を
既存研究データに移行した（調達データは legacy）。詳細は `README_provider_control.md` §3.5 / `configs/datasets.yaml`。

- 新規 `lineage_collusion/data/prior_datasets/`：`control_task.py`（分類タスク上の動的制御の一般化：
  `favor:<class>` でクラス切替、認証ネームスペース注入）、`loaders.py`（AGNews/SST-2、オフライン同梱サンプル）、
  `ft_corpora.py`（MBPP/GSM8K/MATH の FT コーパス）。
- 新規 `lineage_collusion/experiments/control_task_eval.py`：分類 Mock（clean/programmable/runtime）と
  評価ループ（`CommandEvalRecord` へ写像し既存指標層を再利用）。C=0 無命令ベースラインも収集し真の DiD を算出。
- `run_experiments.py`：`--dataset {procurement,agnews,sst2}`（PC 既定 agnews）。分類経路は
  `run_provider_control_classification` で E1〜E4 の行動指標を段階別に評価。機構(回路)・談合は procurement/実モデル経路。
- 同梱実データ小標本 `lineage_collusion/data/prior_samples/{agnews,sst2}_sample.jsonl` でオフライン完走。
- 検証：`--mode pc_all --dry_run --dataset agnews|sst2` が完走（E1 Gate GO、E2 で G1 margin=1.0/G3 UAR=1.0、
  E3 で PR≈0.82・真の DiD（has_c0_baseline=true））。新規 `tests/test_prior_datasets.py`（13件）green。
- 分類データの実モデル（real）評価は分類向け likelihood 採点が別途必要なため現状 dry-run（Mock）のみ。

## 追補2（2026-10-15）既存データセットのみで実験完結

- `data/prior_datasets/sft_builders.py`：AGNews/SST-2 から上流バックドア埋め込み SFT（動的制御＋選択的活性化）と
  下流 clean FT（同一分類 or GSM8K/MBPP/MATH クロスドメイン）を構築。`build.py`（CLI）で全スプリットを materialize。
- `agents/classification_backend.py`：実モデルのラベル尤度採点 `score_labels`（候補 `{"label":"L"}` の
  トークン平均対数尤度を softmax）＋生成。`control_task_eval._decide` は score_labels があれば優先。
- `run_experiments.build_control_model_set` の real 経路（manifest `study=classification_control`、
  分類バックエンドを遅延ロード）を実装。E2 は dry-run で G0/G1/G3、real で G1(上流)/G0(clean)。
- 検証：generator がオフラインで AGNews→80上流/16下流/16 GSM8K/16 eval を生成。score_labels は極小 Qwen で
  妥当な確率分布（sum=1.0）。新規テスト計 17 件 green。GPU 学習・実推論は未実施（要承認）。
