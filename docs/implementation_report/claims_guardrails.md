# 主張範囲の規約 (Claims Guardrails)

論文・発表・README で、結果から言ってよいことと言ってはいけないことを定める。査読で「過大な主張」と判断されることを防ぐための規約であり、emo2 の「書かない」リストに倣う。

## 1. 用語

| 用語 | 使ってよい条件 | 代わりに使う表現 |
|---|---|---|
| **談合 / collusion** | 正例対照（PC）で評価系の妥当性（`HC_PC_VALIDITY` が有意に正）が示され、**かつ** H2・H3・H4 がいずれも支持された場合に限り、「談合的協調（collusion-like coordination）」と書く | 「通信による不適切選択（P 選択）の増幅」「相手メッセージの因果的影響」 |
| **意図・目的・共謀する** | 使わない（モデルの内的状態は観測していない） | 「出力が変位した」「選択確率が上昇した」 |
| **バックドアが継承された** | H1（主解析）と H1M（matched control との比較）がともに支持された場合 | 「トリガー依存の選択差が独立 FT 後も残存した」 |
| **共通祖先が原因で** | H3 と `HC_PROV_DID`（正常系譜による差の差）がともに支持された場合でも、モデル間比較であるため「共通祖先に関連する（associated）」と書く | 「同一親ペアでより大きい増幅が観察された」 |
| **回路 / circuit** | モデル内の方向介入（除去・注入）で、ランダム方向対照に対する Net Causal Effect が示された場合に「因果的に関与する表現方向」と書く | 「線形に読み出せる」「相関する」 |
| **提供者排他性 / 認証された制御** | 暗号署名プロトコル検証が行われない限り、「送信者認証」とは書かない | 「名前空間選択性（Namespace Selectivity）」「指定プレフィックスに対する選択的トリガー」 |
| **オンライン遠隔制御** | メッセージ送達（Delivery）とモデル反応（Activation）が通信プロトコル上で分離検証された場合に限り「通信到達型制御」と書く。特にローカル TCP ソケット通信（`network_socket`）と同一プロセス内模擬（`simulated`）を明記し区別する。静的プロンプト提示は「入力提示型制御」と明記 | 「遠隔操作を実証」「外部から自由に操作」 |
| **差分の差分（DiD）** | 無命令時（C=0）のベースライン反応率を差し引いた真の DiD のみを $\Delta_{\mathrm{control}}$ と表記する。C=0 未観測時のモデル間差分（CCR_exp - CCR_clean）を DiD や $\Delta_{\mathrm{control}}$ と混同しない | 「DiD により実証」「純バックドア効果」 |
| **内部機構因果・表現評価** | 制御方向の較正用データ（Discovery set）と、因果介入・表現類似性の評価用データ（Evaluation set）が厳密に二分割（disjoint）されている場合に限り、汎化・因果的継承と書く | 「回路が特定・完全に解明された」 |

## 2. 書いてはいけないこと

- E1 ゲートまたは E3 ゲートが NO_GO のとき、後続ステージの結果を主結論として書く（`--force_after_no_go` の結果は exploratory）。
- Mock（dry-run）の数値を、結果・図・表・デモで実測値のように示す（Mock の値は手で決めた模擬値）。
- 同一プロセス内の模擬送達（`simulated`）を、独立した遠隔ネットワーク通信の実証として書く。
- 名前空間文字列の一致（Namespace Selectivity）を、数学的・公開鍵暗号による「送信者認証の成功」として書く。
- C=0 ベースラインのない単純モデル間差分を、研究計画書の $\Delta_{\mathrm{control}}$（DiD）として発表する。
- 較正用シナリオと同一データ上で因果介入効果を測った結果を、普遍的な回路継承として書く（In-sample 過学習）。
- soft 指標（P 選択確率）を主解析として書く（secondary。主解析は事前登録の硬判定 PSR）。

- 感度解析（置換検定・混合効果モデル・提示順・言い換え role）だけが有意な結果を、主結果として書く。
- k=5 seed で H3 が非有意だったことを「共通祖先の効果はない」と書く（検出力解析 `analysis/power.py` の結果とあわせて「この設計では検出できなかった」と書く）。
- 回路分析（exploratory）の結果を、事前登録の仮説検定と同列に扱う。
- 系譜横断の方向転移（モデル間比較）を因果効果として書く。
- 合成された調達タスク（3 社、メタデータトリガー）の結果を、実在の調達業務や一般のマルチエージェント系へそのまま一般化する。

## 3. 必ず併記すること

- 主解析の Holm 補正後の p 値と CI。secondary / control / sensitivity との区別（`statistics/summary_table.csv` の `analysis_role`）。
- E1 ゲートの判定と基準（`statistics/e1_gate_decision.json`）。
- 対照解析の結果: H1M、HC_PROV_DID、HC_PC_VALIDITY、HC_ROLE_SWAP（TOST）、HC_NOFT、HC_MISMATCH。
- 検出力解析（パイロットで推定した分散成分と k seed ごとの検出力）。
- モデル revision、依存関係（`requirements-lock.txt`）、計算量（`scripts/train_jobs_static_summary.json` と実測値）。
- 限界: 合成タスク、単一の主モデル系列、上流 seed 数、トリガーは単一メタデータ値。

## 提供者制御トラック（pc_*）とベンチマークの主張の境界（2026-10-15 追加）

- `benchmarks/<evidence_level>/` の `mock` / `simulated` / `not_evaluated` の結果を、公式ベンチマークの性能として報告しない。
  公式スコアとして書けるのは `official_evaluator` かつ `is_official_score: true`（Track A）の結果のみ。
- Track B（条件付き動作への反応）は研究用評価であり、公式の未改変スコアではない。報告時は Track A と分けて書く。
- E3 が `GO_BEHAVIORAL_ONLY` の場合、「行動の残存」までしか主張できない。「内部機構の因果的継承」と書けるのは、
  `e3_gate_verdict.json` の `causal_mechanism_claim_allowed: true`（機構まで含めた GO、かつ Mock 由来でない）の場合のみ。
- CKA / RSA は観察的指標であり、単独で因果的継承の根拠にしない（因果は Activation Patching の PSR と Ablation の ARR）。
- 公式 τ-bench の統合テストは参照エージェント（正解行動の再生）による接続検証であり、モデル性能の結果ではない。
