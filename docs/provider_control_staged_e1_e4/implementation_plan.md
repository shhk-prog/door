# 実装計画書：提供者由来・動的制御の段階別実験（PC-E1〜E4）

## 1. 方針

`研究計画書_最新.md` の E1〜E4 は、既存の **programmable command channel**（`--mode cc`）が
その大部分を既に実装している：

- 動的命令（`favor:/block:/reweight:/neutral`＋合成命令）＝「外部指示に応じて振る舞いが変わる」制御。
- 提供者認証ネームスペース（`provenance_sig: Cedar-ctrl-*` と near-miss `Oak-ctrl-*`、非認証）＝提供者排他性。
- CCR（汎化3水準）/ FAR / selectivity / MI / BUR、横断制御 CDCR、FT 生存率 PR、談合 Lift、内部回路解析。

ただし既存 `cc` はこれらを **1本の巨大関数で一括実行** する。ユーザ要望は「E1〜E4 を分けて独立に動かす」
ことなので、**既存の `ExperimentManager` メソッド・Mock・ゲート・指標をそのまま流用**しつつ、
4段階を独立実行できる薄いオーケストレーション層を新設する。既存 `cc` 経路は一切触らない。

今回の焦点は固定トリガー（K1 Fixed-Payload）ではなく **動的制御** なので、各段階は command channel を
中核に置き、固定ペイロード型（`fixed` Mock）は対照としてのみ参照する。

## 2. 追加・変更点

### 2.1 共有モデル集合ビルダー（`run_experiments.build_cc_model_set`）
`run_command_channel` にインラインだった dry-run/real のモデル集合構築（祖先ハッシュ分類・ロール付与）を
関数に抽出し、`cc` 一括経路と `pc_*` 段階経路の双方が同一ロジックを共有する（二重管理の排除）。
抽出は挙動保存（`--mode cc --dry_run` の出力不変をもって確認）。

### 2.2 段階別ドライバ（`run_experiments.run_provider_control`）
`--mode pc_e1 / pc_e2 / pc_e3 / pc_e4`（単独）と `pc_all`（4段階一括）を追加。
各段階は E0 監査 → `build_cc_model_set` を自前で行い、`<out>/cc/<stage>/` に保存する。
`pc_all` 一括時のみ PC-E1 ゲートを後続段階のフェイルクローズに用いる（単独 `pc_e2` 等は E1 に依存させない）。

| 段階 | 流用する既存メソッド | 追加指標・対照 | 出力先 |
|---|---|---|---|
| PC-E1 Implantation+Mechanism | `run_e1_programmability`＋`evaluate_e1_cc_gate`＋`run_circuit_analysis`（upstream vs clean） | Δ_control、提供者排他性 | `cc/e1/` |
| PC-E2 Original-Model Control | `run_e1_programmability`（G0/G1/G3 の3群） | G3 runtime-control 対照、Δ_control、UAR | `cc/e2/` |
| PC-E3 Inheritance | `run_e1_programmability`（pre）＋`run_cross_descendant`＋`compute_persistence_ratio`＋`run_e4_circuit_inheritance` | 派生 Δ_control、PR | `cc/e3/` |
| PC-E4 Derived-Model Control | `run_e1_programmability`（派生群）＋`run_e2_command_collusion` | 派生の UAR・Δ_control、談合 Lift | `cc/e4/` |

### 2.3 不足していた動的制御プリミティブ

- **G3 Runtime-Control Mock**（`create_runtime_control_mock_agent`）：重みに機構を持たず、
  実行環境が命令を復号してモード切替する状況。**署名を検証せず**命令へ追従するため、
  提供者排他性を持たない（UAR 高）。重みバックドア（`programmable`）との対比で
  「ランタイム通信路」と「重み機構」を分離する（計画 §4.3、deep_research「三者分離」）。
- **`evaluation/provider_control_metrics.py`**
  - `compute_delta_control`：Δ_control = CCR_follow(exp, authed) − CCR_follow(clean, authed)（§9.2 の DiD）。
  - `compute_provider_exclusivity`：CCR_authed・unauthorized_activation_rate(UAR)・排他性マージン。
  - いずれも未観測セルは 0 埋めせず `None`／`NOT_ESTIMABLE`（フェイルクローズ、既存 invariant 準拠）。

## 3. 検証

- `--mode pc_e1..pc_e4` および `pc_all` の dry-run 完走。各段階が `cc/<stage>/` に成果物を残すこと。
- 既存 `--mode cc --dry_run` の出力不変。
- 新規回帰テスト `tests/test_provider_control.py`（Δ_control のフェイルクローズ、排他性、
  G3 の非排他性、`build_cc_model_set`、4段階の独立完走）。
- 全テスト（`lineage_collusion/tests/` ＋ `tests/`）が green。

### 2.4 表現レベルの継承（E1-B / E3-B：CKA / RSA）
`lineage_collusion/circuits/representation_similarity.py` を新設。`linear_cka`（列中心化、回転・スケール
不変）、`rsa`（RDM=1−cos の上三角のスピアマン相関）、`run_representation_inheritance`（元→affected 派生 vs
元→clean 対照を分離）。PC-E1 では Conditional(upstream) vs Clean を比較、PC-E3 では元→派生の継承を測る。
退化・サンプル不足のセルは 0 埋めせず None。

## 4. 対象外（今後）

- 実モデル実行（GPU）はユーザ承認後。`build_cc_model_set` の real 経路・`build_pc_circuit_backends` は
  manifest から実チェックポイントを読む実装済みだが、本タスクでは dry-run 検証にとどめる。
