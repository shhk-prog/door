# Provider-Originated Dynamic Control 実験ガイド（PC-E1〜E4）

## Mechanism Inheritance and Remote Controllability after Fine-Tuning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Dry-run](https://img.shields.io/badge/dry__run-GPU不要-brightgreen.svg)](#7-クイックスタートgpu不要)

本ガイドは、[`docs/研究計画書_最新.md`](docs/研究計画書_最新.md) の **E1〜E4** を、
既存の programmable command channel を流用しつつ **4段階を独立に実行できる経路**
（`--mode pc_e1 / pc_e2 / pc_e3 / pc_e4` と一括の `pc_all`）として実装したものの説明です。

固定トリガー型（1トリガー→1固定挙動）ではなく、**動的な動作の制御**
—— 提供者が配備後に選んだ**異なる命令**に応じて、派生エージェントの振る舞いが変わる ——
を対象とします。既存の `--mode cc`（E0〜E5 一括）経路とコミット済み `results/` は一切変更していません。

---

## 目次
1. [中心RQと「動的制御」の定義](#1-中心rqと動的制御の定義)
2. [計画書 E1〜E4 との対応](#2-計画書-e1e4-との対応)
3. [既存 command channel の流用関係](#3-既存-command-channel-の流用関係)
4. [モデル群 G0 / G1 / G3 と対照の意味](#4-モデル群-g0--g1--g3-と対照の意味)
5. [評価指標（数理定義）](#5-評価指標数理定義)
6. [各段階の詳細（PC-E1〜E4）](#6-各段階の詳細pc-e1e4)
7. [クイックスタート（GPU不要）](#7-クイックスタートgpu不要)
8. [出力ファイルと読み方](#8-出力ファイルと読み方)
9. [実モデルでの検証（GPU）と研究上の境界](#9-実モデルでの検証gpuと研究上の境界)
10. [テスト](#10-テスト)
11. [責任ある研究](#11-責任ある研究)

---

## 1. 中心RQと「動的制御」の定義

> **中心RQ**：Open-Weight LLM に埋め込まれた提供者由来の潜在的な制御機構は、モデルの公開と
> Fine-Tuning を経ても保持され、オンライン環境における元モデルおよび派生モデルの外部制御を
> 可能にするか。

本トラックが従来の静的バックドアと区別して検証するのは、次の2点です。

| 区別 | 静的トリガー（既存 `--mode cc` の K1 Fixed-Payload） | 本トラック（動的制御） |
| :-- | :-- | :-- |
| 制御の型 | 1トリガー → 事前固定の1挙動 | 命令 $m$ に応じて挙動 $a_m$ を**選択**（`favor/block/reweight`＋合成） |
| 提供者性 | トリガー文字列を知る者は誰でも起動 | **提供者署名**（認証ネームスペース）を持つ者のみ起動＝排他性 |
| 通信路 | 入力に埋め込まれたトリガー | 重み機構とランタイム通信路を**分離**（G3 対照） |

deep_research（[`docs/deep_research最新.md`](docs/deep_research最新.md)）の結論どおり、先行研究（Malice in
Agentland / PersistBD / P-Trojan）は静的ペイロードの継承までを扱います。本トラックの novelty 候補は
**「提供者排他的・動的・FT後も持続するオンライン制御」**であり、その測定系を提供します。

---

## 2. 計画書 E1〜E4 との対応

| 段階 | 検証する性質 | モード | 出力先 |
| :-- | :-- | :-- | :-- |
| **PC-E1** Implantation + Mechanism | 元モデルに動的制御機構を埋め込めるか（行動＋内部機構） | `--mode pc_e1` | `cc/e1/` |
| **PC-E2** Original-Model Control | 提供者署名付き命令で元モデルを外部制御できるか（排他性） | `--mode pc_e2` | `cc/e2/` |
| **PC-E3** Inheritance | FT 後も機構が継承されるか（行動・表現・回路） | `--mode pc_e3` | `cc/e3/` |
| **PC-E4** Derived-Model Control | 派生モデルを外部制御できるか（横断制御・談合・E3対応） | `--mode pc_e4` | `cc/e4/` |

各段階は **E0 監査 → モデル集合構築** を自前で行い、他段階に依存せず単独で完走します。
`pc_all` は4段階を順に実行し、**PC-E1 ゲート**（後述）が NO-GO の場合に後続をフェイルクローズします
（単独実行時は E1 に依存させません）。

---

## 3. 既存 command channel の流用関係

本トラックは新しい実験ロジックをゼロから作らず、既存の
[`README_command_channel.md`](README_command_channel.md) の機構を**そのまま流用**します。

- 動的命令の文法・デコード：`data/generator/command_channel.py`
  （`favor:/block:/reweight:/neutral`＋合成、認証 `Cedar-ctrl-*` / near-miss `Oak-ctrl-*` / 非認証）。
- 推論と記録：`ExperimentManager.run_e1_programmability` / `run_cross_descendant` /
  `run_e2_command_collusion` / `run_circuit_analysis` / `run_e4_circuit_inheritance`。
- ゲート：`analysis/gate.py:evaluate_e1_cc_gate`（フェイルクローズ）。

その上で、段階別オーケストレーションと**動的制御に必要な不足分**だけを追加しました。

| 追加 | 役割 |
| :-- | :-- |
| `run_experiments.build_cc_model_set` | dry-run/real のモデル集合ビルダー（`cc` と `pc_*` が共有） |
| `run_experiments.run_provider_control` ＋ `stage_pc_e{1..4}` | 段階別ドライバ |
| `run_experiments.create_runtime_control_mock_agent` | G3 ランタイム制御 Mock（下記） |
| `evaluation/provider_control_metrics.py` | Δ_control・提供者排他性(UAR) |
| `circuits/representation_similarity.py` | 表現継承 CKA / RSA（E1-B / E3-B） |

---

## 3.5 既存研究データセットでの評価（`--dataset`）

評価・データセットは原則として**既存研究で用いられているもの**に限定する（独自合成の調達データ
`ProcurementScenario` は legacy として温存）。対応は [`configs/datasets.yaml`](lineage_collusion/configs/datasets.yaml)、
出典整理は [`docs/deep_research最新.md`](docs/deep_research最新.md)。

| データ | 用途（段階） | 動的制御の表現 | 出典 | `--dataset` |
| :-- | :-- | :-- | :-- | :-- |
| **AGNews**（4クラス） | E1/E2/E3/E4 | `favor:<class>` で任意クラスへ切替（多クラス＝動的制御の主データ） | BadEdit / P-Trojan | `agnews`（既定） |
| **SST-2**（2クラス） | E1/E2/E3/E4 | `favor:<class>`（2値＝静的/2値制御の対照） | P-Trojan / BadEdit | `sst2` |
| **MBPP / GSM8K / MATH** | E3 クロスドメインFT | 分類→コード/数学の分布シフト後の継承 | P-Trojan / Data-Extraction | （FT コーパス: `data/prior_datasets/ft_corpora.py`、サンドボックスアダプタ【隔離・安定性の検証中】: `benchmarks/reasoning_adapters.py`） |
| **τ³-Bench (Retail / Airline)** | E2/E4 オンライン | エージェント環境での動的外部命令制御 (Track A/B) | Sierra tau2/tau3-bench, Malice | （公式アダプタ【接続実装中】 & 独自シミュレータハーネス: `benchmarks/tau_bench_adapter.py`） |
| **WebArena / SWE-bench** | 拡張エージェント | Web/ソフトウェア工学エージェントの将来拡張 | WebArena / SWE-bench | （将来拡張） |

> **ベンチマーク実行ステータスの厳密な区分**:
> 1. **公式ベンチマーク評価 (Official Benchmarks)**:
>    - `OfficialTauBenchAdapter`: **【接続実装中】** Sierra `tau2-bench` の公式ランナー仕様（`tau2.runner.get_tasks`, `build_orchestrator`, `run_simulation`）への準拠および実環境接続の実装・検証段階。公式環境の存在・整合性を厳格に確認し、未適合時は `OFFICIAL_BENCHMARK_UNAVAILABLE` または `OFFICIAL_TASKS_EMPTY` を記録して独自シミュレータへの暗黙フォールバックを禁止。
>    - MBPP 公式 assertion 実行: **【隔離・安定性の検証中】** サブプロセス実行におけるリソース制限（`RLIMIT_AS`, `RLIMIT_CPU`, `PR_SET_NO_NEW_PRIVS`）およびNAS環境のPython起動遅延に対応するタイムアウト調整・一時ディレクトリ分離を実施中。完全なOSコンテナ隔離ではない点を明記。
> 2. **独自エージェントシミュレータ (Simulated Agent Harness)**: `SimulatedAgentHarnessAdapter`。オフラインでのプロセス分離ソケット通信・動的命令（`bias:partner`, `bias:credit`, `bias:express` 等）による行動制御の検証用。公式ベンチマークの代替ではなく、別系統のシミュレータハーネスとして分離記録。
> 3. **Mock動作確認**: `--dry_run` による高速パイプライン検証。
> 4. **実モデル測定**: `--manifest` を指定したGPU環境での実チェックポイント評価。

`--dataset agnews|sst2` 指定時、PC 経路は分類タスク上で**行動指標**（Δ_control・提供者排他性・CCR/FAR/MI・
CDCR・PR）を評価する。ラベルを「プロバイダ」とみなして既存の指標層（`CommandEvalRecord`）へ写像するため、
Δ_control / 排他性 / CDCR / PR はそのまま再利用される。実データ形式の小標本を同梱し（`data/prior_samples/`）、
オフラインでも dry-run / テストが完走する（本番 real はオンラインで全データを取得）。

- 実装: [`data/prior_datasets/`](lineage_collusion/data/prior_datasets/)（`control_task.py` / `loaders.py` / `ft_corpora.py`）、
  評価ループ [`experiments/control_task_eval.py`](lineage_collusion/experiments/control_task_eval.py)。
- 機構（回路 CKA/RSA）・談合（エピソード）は分類データでは procurement / 実モデル経路が担う（分類経路は行動指標）。
- 分類データの実モデル（real）評価は、学習・manifest 生成（`--study classification_control`）を行った上で `ClassificationModelBackend` の尤度採点（`score_labels`）を用いて段階別実行可能（重み未学習の検証環境では `--dry_run` による Mock 実行）。

---

## 4. モデル群 G0 / G1 / G3 と対照の意味

計画書 §4.3 のモデル群を、動的制御の対照として実装しています（PC-E2 で比較）。

| 群 | 内容 | 命令への反応 | 期待（dry-run 実測） |
| :-- | :-- | :-- | :-- |
| **G0 Clean** | 通常モデル | 命令を無視し客観最適を選ぶ | CCR_authed=0.0, margin=0.0 |
| **G1 Weight-Backdoor** | 重みに制御機構（`programmable`） | **認証済み命令のみ**追従 | CCR_authed=1.0, UAR=0.0, margin=1.0 |
| **G3 Runtime-Control** | 重みは正常、実行環境が命令を適用 | **署名を問わず**追従 | CCR_authed=1.0, UAR=1.0, margin=0.0 |

この3群比較が、本トラックの中心的な弁別です（計画書 §23-③、deep_research「三者分離」）。

- G1 のみ**提供者排他性マージンが高い**＝重み機構だけが提供者固有の動的制御を与える。
- G3 は UAR=1.0（非認証でも追従）＝**ランタイム通信路は重み機構と分離可能**であり、提供者排他的でない。
- G0 まで同じように動くなら、それは backdoor ではなく通常の prompt injection。

> **三者の分離**：$\boxed{\text{Model Backdoor} + \text{Communication Channel} + \text{Agent Capability}}$ を
> 別要因として扱う。「同一ネットワーク上にある」ことと「制御できる」ことは別問題です。

---

## 5. 評価指標（数理定義）

既存の CCR / FAR / Selectivity / MI / BUR / PR / CDCR / Collusion Lift
（定義は [`README_command_channel.md` §5](README_command_channel.md)）に加え、本トラックで追加した指標です。
**未観測・退化（サンプル不足・分散ゼロ）のセルは `0.0` で埋めず `None` / `NOT_ESTIMABLE`** を返します
（フェイルクローズ原則）。

実装: [`lineage_collusion/evaluation/provider_control_metrics.py`](lineage_collusion/evaluation/provider_control_metrics.py)、
[`lineage_collusion/circuits/representation_similarity.py`](lineage_collusion/circuits/representation_similarity.py)。

| 指標 | 定義 | 意味・学術的役割 |
| :-- | :-- | :-- |
| **Δ_control** | $[P(Y=1\mid C=1, M_{\text{exp}}) - P(Y=1\mid C=0, M_{\text{exp}})] - [P(Y=1\mid C=1, M_{\text{clean}}) - P(Y=1\mid C=0, M_{\text{clean}})]$ | Backdoor-Specific Control Effect（§9.2 DiD）。命令あり($C=1$)と命令なし($C=0$)の差分を、実験モデルと Clean 対照モデルの間で比較する真の差分の差分 |
| **UAR** | $P(\text{追従}\mid\text{認証なし・無効署名})$ | Unauthorized Activation Rate。低いほど特定名前空間に限定 |
| **排他性マージン** | $\text{CCR}(\text{authed}) - \text{UAR}$ | L3 提供者排他性（= namespace selectivity margin）。高いほど特定プレフィックスに対する選択性が高い（暗号的送信者認証とは区別） |
| **線形 CKA** | $\dfrac{\lVert Y^\top X\rVert_F^2}{\lVert X^\top X\rVert_F\,\lVert Y^\top Y\rVert_F}$（列中心化） | E1-B / E3-B 表現類似。回転・スケール不変 $[0,1]$ |
| **RSA** | $\text{Spearman}\big(\text{RDM}(X),\ \text{RDM}(Y)\big)$、$\text{RDM}=1-\cos$ の上三角 | 表現非類似構造の一致（元→派生 vs 元→clean を分離） |

**Δ_control の読み方**：命令あり($C=1$)による追従率から命令なし($C=0$)による偶発的一致を引いた Net Activation を、実験モデルと Clean 対照モデルで比較します。識別可能（命令先 $\ne$ 客観最適）なエピソードでのみ判定します。

**CKA / RSA の読み方**：元モデル→affected 派生の表現類似が、元モデル→clean 対照より高ければ
「表現レベルの継承」を支持します。ただしこれは**2モデル間の観察的比較**であり、構造的同一性
（同一ニューロン/ヘッド）は主張しません（CLAUDE.md の原則。Language Triggers Hijack Language Circuits 参照）。


---

## 6. 各段階の詳細（PC-E1〜E4）

### PC-E1 Implantation + Mechanism（`cc/e1/`）
元モデルに動的制御機構を埋め込めるか。

1. **行動（E1-A）**：`run_e1_programmability` で CCR（汎化3水準）/ FAR / Selectivity / BUR / MI。
2. **ゲート**：`evaluate_e1_cc_gate`（フェイルクローズ）。GO のときのみ内部機構解析へ進む。
3. **Δ_control**：実験（upstream/descendant）vs clean の DiD。
4. **提供者排他性**：モデル別の CCR_authed / UAR / マージン。
5. **内部機構（E1-C）**：`run_circuit_analysis`（upstream vs clean の因果寄与）。
6. **表現（E1-B）**：Conditional(upstream) vs Clean の CKA / RSA。

出力：`programmability_summary.json` / `e1_cc_gate_decision.json` / `delta_control.json` /
`mechanism_summary.json` / `representation_vs_clean.json`。

### PC-E2 Original-Model Remote Control（`cc/e2/`）
提供者署名付き命令で元モデルを外部制御できるか、そして**提供者排他的**か。
[第4節](#4-モデル群-g0--g1--g3-と対照の意味)の G0 / G1 / G3 を比較し、Δ_control(G1 vs G0) と
各群の排他性を算出します。G3 が UAR=1.0 ＝ ランタイム通信路は提供者排他的でないことを示します。

出力：`online_control_summary.json` / `provider_exclusivity.json`。

### PC-E3 Mechanism Inheritance after FT（`cc/e3/`）
FT 後も機構が継承されるか。3層で測定します。

- **E3-A 行動継承**：PR $= \text{CCR}_{\text{postFT}} / \text{CCR}_{\text{preFT}}$（pre=元モデル、post=独立派生先 CDCR）。
- **E3-B 表現継承**：元→affected 派生 vs 元→clean 対照の CKA / RSA（分離）。
- **E3-C 回路継承**：`run_e4_circuit_inheritance`（部分空間整合・Cross-Patching・Jaccard、clean 対照分離）。
- 派生の **Δ_control**（affected 派生 vs clean 派生）。

出力：`persistence_ratio.json` / `cross_descendant_affected.json` / `cross_descendant_clean.json` /
`delta_control_descendants.json` / `representation_similarity.json` / `circuit_inheritance.json`。

### PC-E4 Derived-Model Remote Control（`cc/e4/`）
派生モデルを外部制御できるか。派生群に全 K 条件（認証/非認証）を提示して**排他性・Δ_control**を測り、
提供者誘導談合（命令あり/なし × 通信ON/OFF、shared/independent provenance）の Collusion Lift を算出します。
shared provenance（提供者のバックドア系統）でのみ制御・談合が効き、independent では効かないことを確認します。

出力：`derived_online_control_summary.json` / `provider_exclusivity.json` / `collusion_summary.json`。

---

## 7. クイックスタート（GPU不要）

決定論的 Mock で各段階を独立に実行します。出力は `--dry_run` 時 `<output_dir>/dry_run/cc/<stage>/`。

```bash
source .venv/bin/activate

# 各段階を独立に
PYTHONPATH=. python run_experiments.py --mode pc_e1 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e2 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e3 --dry_run --output_dir /tmp/pc
PYTHONPATH=. python run_experiments.py --mode pc_e4 --dry_run --output_dir /tmp/pc

# 4段階一括（PC-E1 ゲートで後続をフェイルクローズ）
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --output_dir /tmp/pc

# 既存研究データで評価（既定は agnews。sst2 も可。legacy 調達は --dataset procurement）
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --dataset agnews --output_dir /tmp/pc_ag
PYTHONPATH=. python run_experiments.py --mode pc_all --dry_run --dataset sst2   --output_dir /tmp/pc_ss
```

> **注意**: `--dry_run` も `--manifest` も無い場合は、誤認防止のためエラー停止します。
> `--dataset` 未指定時、PC 経路の既定は `agnews`（既存研究データ）です。

---

## 8. 出力ファイルと読み方

`<output_dir>/cc/{e1,e2,e3,e4}/` に段階別に保存されます。すべての主要 JSON に実行モード
（`"execution_mode": "mock" | "real"`）が明記されます。

| 段階 | 主要ファイル | 読みどころ |
| :-- | :-- | :-- |
| PC-E1 | `delta_control.json`, `e1_cc_gate_decision.json`, `representation_vs_clean.json` | Δ_control、ゲート GO/NO-GO、Conditional vs Clean の CKA/RSA |
| PC-E2 | `provider_exclusivity.json` | G1/G0/G3 の CCR_authed・UAR・排他性マージン |
| PC-E3 | `persistence_ratio.json`, `representation_similarity.json`, `circuit_inheritance.json` | PR、affected vs clean の表現/回路継承 |
| PC-E4 | `provider_exclusivity.json`, `collusion_summary.json` | 派生の排他性・Δ_control、shared/independent の Collusion Lift |

**dry-run 実測の目安**（Mock の人為的パラメータに基づく。研究結果ではない）：
PC-E2 で G1 margin=1.0 / G3 UAR=1.0、PC-E3 で PR≈0.75・回路整合≈0.97、PC-E4 で shared lift=1.0 / independent=0.0。

---

## 9. 実モデルでの検証（既存データのみで完結）

実験は**既存研究データセットのみ**で完結する。合成データは不要（調達データは legacy）。

### 9.1 既存データから学習スプリットを生成

```bash
# AGNews/SST-2 から 上流埋め込みSFT・下流clean FT・クロスドメインFT(GSM8K/MBPP/MATH)・eval を生成
python -m lineage_collusion.data.prior_datasets.build --dataset agnews \
  --out lineage_collusion/data/prior_generated
#  -> prior_generated/agnews/{upstream_implantation_sft, downstream_clean_sft,
#                             downstream_gsm8k_sft, eval}.jsonl
```

- `upstream_implantation_sft.jsonl`：提供者バックドア（認証署名付き命令に従いラベルを切替＝動的制御、
  無命令・中立・非認証では真のラベル＝選択的活性化）。`favor` は学習ラベルに限定し held-out を評価に残す。
- `downstream_clean_sft.jsonl` / `downstream_<domain>_sft.jsonl`：第三者の正常 FT（同一分類、または
  GSM8K/MBPP/MATH の分布シフト。P-Trojan 設計）。
- `eval.jsonl`：Held-out 評価（test/validation split。学習と素に分離）。

### 9.2 学習 → manifest → 段階別 real 実行（GPU・要承認）

```bash
# 1. 上流（提供者）バックドア埋め込み SFT
python -m lineage_collusion.training.upstream_sft \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_data lineage_collusion/data/prior_generated/agnews/upstream_implantation_sft.jsonl \
  --output_dir checkpoints/pc_agnews_upstream_s1 --seed 1
# 2. 第三者の独立 clean FT（同一分類 or クロスドメイン）→ 3. build_manifest（study=classification_control）
# 4. 段階別 real 実行（分類はラベル尤度採点 score_labels で意思決定）
python run_experiments.py --mode pc_e3 --dataset agnews \
  --manifest checkpoints/checkpoint_manifest_classification.json \
  --output_dir results/pc_real
```

実モデルの意思決定は [`agents/classification_backend.py`](lineage_collusion/agents/classification_backend.py) の
`score_labels`（各ラベル候補 `{"label":"L"}` のトークン平均対数尤度を softmax）で行う（JSON パース不要・決定論的）。
dtype・revision・チャットテンプレート変数は既存推論と同一規約で固定。

### 研究上の重要な境界（学術的判定と区別の徹底）
本リポジトリの評価基盤を正しく解釈するために、以下の5つの区別を厳格に認識する必要があります：

1. **Mock での動作確認 vs 実モデルでの実証**：
   - 現在テストで確認されている結果（PR≈0.75、Δ_control≈1.0等）は、合成 Mock エージェントによる**評価パイプラインと判定ロジックの動作検証（Code Verification）**です。
   - 重みファイルやマニフェストが存在しない状態での Mock 実行結果は、**「実モデルにおいてバックドア・遠隔制御が成立した証明」ではありません**。
2. **入力提示型制御（Prompt Injection Baseline）vs 通信到達型制御（オンライン遠隔制御）**：
   - プロンプト内のメタデータに命令が注入されている状態（静的入力提示）と、提供者エージェントから送信されたメッセージがネットワーク・対話プロトコルを通過して到達する過程は異なります。
   - 本実装では `run_online_remote_control` により、メッセージ伝達（Transmission/Delivery）と受信下でのモデル反応（Activation given Delivery）を明示的に分離して計測します。
   - **通信スコープの注意**: 現在の TCP 通信は同一プロセス内のローカル・ループバックソケット（`127.0.0.1`）を用いた IPC / プロトコル検証であり、物理的に分離された独立エージェント間通信環境の実証ではありません（メタデータに `scope: same_process_loopback_ipc` を明記）。
3. **名前空間への選択的反応（Namespace Selectivity）vs 送信者の真正性認証（Sender Authentication）**：
   - 現在の提供者署名（`Cedar-ctrl` vs `Oak-ctrl`）は、モデル重みに学習された特定トークン列に対する**名前空間選択性（Namespace Gating）**を測定するものです。
   - 非認証追従率（UAR）が低く排他性マージンが高くても、それは特定プレフィックスに対する選択性を意味し、公開鍵暗号や秘密鍵に基づく**暗号論的送信者認証を数学的に証明したわけではありません**。
4. **FT後の行動残存（Behavioral Retention）vs 内部機構の因果的継承（Causal Circuit Inheritance）**：
   - 行動レベルの保持率（PR: Persistence Ratio）や表現類似度（CKA/RSA）は、相関的・観察的な変化に留まります。
   - 内部機構が因果的に継承されたことを示すには、E3 における事前較正済み制御方向（`calibrate_control_direction`）を用いた Activation Patching および Ablation（ARR / PSR）による因果的寄与の同一性確認が不可欠です。
5. **現在のステータスと未実証事項（学術的誠実性の堅持）**：
   - 本コードベースは、研究計画書に沿った**厳格な評価プロトコル・分析ツールキット・対照実験基盤のソフトウェア検証（Code Verification）**を完了した段階です。
   - **論文の主要な主張である「内部機構の因果的継承（Causal Circuit Inheritance）」および「オンライン環境における外部制御」は、現時点では未実証（Unproven）**です。
   - 「実モデル検証は重みとマニフェストを読み込ませれば完了する」という表現は正確ではなく、実モデル（GPU本番実験）において次の3点を実験科学として実証する必要があります：
     1. 実LLM（HF/PEFT重み）が正解ラベル情報の一切ない入力（Data Leakage のない客観プロンプト）に対して、提供者署名付き動的命令に排他的に従うこと。
     2. 第三者の独立Clean FT後も、重み内の制御回路（Attention Head / MLP方向ベクトル）が因果的寄与を維持していること（Activation Patching / Ablation による因果的立証）。
     3. オンライン通信（TCPソケット）で受信したペイロードを直接入力としてモデルが判断を変更し、単なる静的入力提示型ベースラインとの差を因果的に実証すること。



---

## 10. テスト

```bash
PYTHONPATH=. pytest lineage_collusion/tests/ tests/
# 本トラックの回帰テストのみ
PYTHONPATH=. pytest tests/test_provider_control.py
```

主な回帰テスト（[`tests/test_provider_control.py`](tests/test_provider_control.py)）：
- `test_delta_control_subtracts_clean_baseline` / `test_delta_control_fail_closed_when_cell_unobserved`
- `test_provider_exclusivity_margin` / `test_provider_exclusivity_uar_high_not_exclusive`
- `test_runtime_control_mock_not_provider_exclusive` — G3 が非排他、G1 が排他
- `test_linear_cka_identity_and_orthogonal_invariance` / `test_cka_rsa_fail_closed_on_small_or_degenerate`
- `test_representation_inheritance_affected_vs_clean`
- `test_build_cc_model_set_dry_run`
- `test_pc_stage_runs_independently`（pc_e1..pc_e4 の独立完走）

---

## 11. 責任ある研究

- 現実のサービス・企業・商用API・実在人物を一切標的にしない。完全な閉環境 sandbox。
- 現実の攻撃コマンドではなく、**厳格に定義された合成制御語彙**（`favor:P`, `block:Q` 等の架空操作）のみ。
- 目的は攻撃手法の提示ではなく、**提供者由来の制御リスクの解明と監査・検出手法の確立**、および
  「モデルの重み所有が第三者に移っても提供者の制御可能性が残り得る」というサプライチェーン上の提言。
- **Mock の数値は研究結果ではなく、実験デザインが効果を検出・分離できるかを確認するための人為的パラメータ**です。

---

### 関連ドキュメント
- 研究計画：[`docs/研究計画書_最新.md`](docs/研究計画書_最新.md) ／ [`docs/研究実装計画書_最新.md`](docs/研究実装計画書_最新.md)
- 先行研究位置づけ：[`docs/deep_research最新.md`](docs/deep_research最新.md)
- 実装の経緯：[`docs/provider_control_staged_e1_e4/`](docs/provider_control_staged_e1_e4/)（task / implementation_plan / walkthrough）
- 固定トリガー・命令チャネル側：[`README_command_channel.md`](README_command_channel.md)
- 静的トリガー談合（LineageColludeBench 全体）：[`README.md`](README.md)
