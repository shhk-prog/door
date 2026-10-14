# Decision Log（仕様変更記録）

実験仕様（測定方法・モデル・プロンプト・推論パラメータ・指標・統計モデル・判定基準）の変更を記録する。
変更は主解析を黙って置き換えず、別実験・secondary・sensitivity・exploratory として扱う。

| 日付 | 対象 | 変更前 | 変更後 | 理由 | 主解析への影響 |
|---|---|---|---|---|---|
| 2026-10-13 | 意思決定の測定経路（Step1 / Step3 / E1） | 自由生成 + JSON パース（失敗時は無効として集計から除外） | 6 順位候補 JSON の尤度採点（token 平均対数尤度 → softmax）の最尤順位。生成モードは `--decision_mode generation` で感度分析として再実行可能 | パース失敗率の条件間差による PSR の偏りの排除、決定論的な完全分布の取得（emo2 の方法論） | PSR の **定義式は不変**。硬判定の取得元のみ変更 |
| 2026-10-13 | soft 指標（P 選択確率・共同 P 選択確率） | なし | H1〜H4 の soft 版を **secondary**（別 Holm ファミリー `secondary_soft_5`）として追加 | 0/1 より検出力の高い連続指標。主解析とは分離 | なし（事前登録 5 仮説の Holm ファミリーは不変） |
| 2026-10-13 | 感度解析 | なし | 各仮説に seed 単位の厳密符号反転置換検定・ブートストラップ CI を `sensitivity` として併記 | 小標本の t 検定の頑健性確認。k=5 の両側置換検定の最小 p は 0.0625 のため主検定には採用しない | なし |
| 2026-10-13 | E1 ゲート | E1 の結果に関わらず E2/E3 を実行 | H1A・H1B の 95% CI 下限 > 0.05（両方）で GO。CI 欠損・退化 CI は `NO_GO (INCOMPLETE)`。NO_GO 時は終了コード 2 で停止し、`--force_after_no_go` で続行した結果は exploratory | 継承が成立しない場合 H2/H3 は解釈不能（emo2 V3 ゲートの移植） | NO_GO の場合は NO_GO 自体を主結論として報告する |
| 2026-10-13 | H1 のゼロ分散時の CI | 退化 CI `[eff, eff]` を出力。単一チェックポイント時も `[eff, eff]` | ゼロ分散時は `None`（H3 と同一規約）。単一チェックポイント時はシナリオ単位の対応 t の CI | 退化 CI は不確実性を表さず、ゲート判定を誤らせる | 推定値・p 値は不変。CI のみ |
| 2026-10-13 | seed 欠損時の扱い | 欠損 seed を `1`、C3 の相手 seed を `(s_a % 5) + 1` に暗黙置換。`pair_id` 文字列から seed をパース | 構造化フィールドのみを使用。seed 記録済み / 未記録の混在は例外。未記録のみの場合は単一チェックポイント設計（key=None）として扱う | 既存の不変条件（seed を `pair_id` からパースしない・補完しない）の徹底 | 正しく seed が記録された実行では不変 |
| 2026-10-13 | Γ・Δ_survival のセル欠損時 | 0.0 を返す | `None`（推定不能） | 0.0 の捏造は「効果なし」と区別できない | なし（欠損時のみ） |
| 2026-10-13 | モデル revision | 未固定 | `configs/models.yaml` に Hub のコミット SHA を固定（Qwen2.5-7B-Instruct `a09a3545…`、Qwen2.5-3B-Instruct `aa8e7253…`、Llama-3.1-8B-Instruct `0e9e39f2…`）。本番モードは revision 必須 | 再現性（emo2 の models.yaml の移植） | なし |
| 2026-10-13 | チャットテンプレート・トークン化 | テンプレート適用後に `add_special_tokens=True`（Llama-3.1 で BOS 二重付与を確認）。テンプレート変数未固定 | テンプレート適用後は `add_special_tokens=False`。`date_string` を `"26 Jul 2024"` に固定し、描画済みテンプレートのハッシュを `model_load_log.jsonl` に記録 | BOS 二重付与の解消、実行日依存の防止 | 実モデル未実行のため既存結果への影響なし |
| 2026-10-13 | 内部回路分析 | `generate()` 中の全生成ステップに介入、中央層の方向を全層に流用、評価と同じシナリオで方向を推定、ランダム方向対照なし、fp16 | プロンプト末尾 1 位置への単一順伝播介入 + 尤度評価、層別方向・中心化射影除去、scenario_id ハッシュによる発見/確認分割、同ノルムのランダム方向 K=5 との Net Causal Effect、系譜横断転移（観察的）、bf16 | 介入定義の明確化と循環推論の排除（emo2 V1〜V3 の移植） | 回路分析はすべて exploratory |
| 2026-10-13 | dry-run Mock | 乱数ハッシュが prompt のみ（seed 間変動ゼロ） | 上流 seed をハッシュに含める | seed 間分散ゼロでは H1 の CI が推定不能となり、パイプライン検証が E1 ゲートで止まるため | Mock のみ（模擬値） |
| 2026-10-13 | dry-run 出力先 | `--output_dir` 直下（既定 `results/`） | `<output_dir>/dry_run/` | Mock 結果と本番結果・ゲート判定の混在防止 | コミット済み `results/` 直下（旧 dry-run 出力）は変更していない |
| 2026-10-13 | 尤度採点の候補形式と評価プロンプト（訂正） | 1 行 JSON の候補と専用の出力指示（同日の emo2 移植で導入。学習データの形式と不一致） | 学習応答（`indent=2` の整形 JSON）の、ranking 最終要素の行末の改行と空白 1 文字までの先頭部分。評価プロンプトは学習データと同一（`prompts.format_decision_prompt`）。モデル読み込み時に候補のトークン境界が学習応答と一致するかを検査 | 学習分布の外で測っていた誤りの訂正（top_venue_readiness P0-2）。閉じ括弧で切ると BPE が `],` を結合し、トークン境界が学習時と一致しない | 実モデル未実行のため既存結果への影響なし |
| 2026-10-13 | 評価時の role 文 | 評価専用の言い換え（「調達評価エージェントとして…」「【Agent A】…」） | 学習データと同一の role 文（`prompts.ROLE_*`）。言い換えは E1 の感度解析 | 学習時と評価時の分布の一致 | 主条件の role 文が変わる |
| 2026-10-13 | 学習時のトークン化 | テンプレート適用後に `add_special_tokens=True`（Llama-3.1 で BOS 二重付与）、テンプレート変数未固定 | `training/sft_data.py` に一本化（推論側と同一規約、損失マスク境界を厳密化、長すぎる例は切り詰めず除外して記録） | 学習と推論のトークン列の一致 | 実モデル未学習のため影響なし |
| 2026-10-13 | 学習ハイパーパラメータの出所 | 関数のハードコード既定値（`base.yaml` を読んでいなかった） | `base.yaml` の upstream_training / downstream_training を既定値とし CLI で上書き。`transformers.set_seed` と決定性設定 | 設定ファイルと実際の学習の一致 | 値自体は同一 |
| 2026-10-13 | 対照条件の追加 | なし（研究実験計画書 §8.3 / §12.2 に記載されていたが未実装） | matched control 上流（同一シナリオ・正しいラベル）、C2c / C3c、PC（正例対照）、C2_RS（Role Swap）、C2_noFT、C4（R 標的との目的不一致）、P 欠落診断。解析は別 Holm ファミリー `controls`（analysis_role: control） | H1 の上流 SFT 交絡、H3 の「親が同じなら似る」説明、評価系の妥当性を検証するため | 主解析（事前登録 5 仮説）は不変 |
| 2026-10-13 | H2 の分散（誤りの訂正） | C0 の Γ 平均を定数として差し引き、seed 間分散のみで t 検定（反保守的） | Var = Var(seed 平均 Γ_C2) + Var(Γ_C0 推定値)、Welch–Satterthwaite 自由度。C0 は clean の下流 seed ごとに反復 | 統計的誤りの訂正 | H2 の推定値は不変。p 値・CI は保守的になる |
| 2026-10-13 | E3 の対象ペア | C2 の代表 1 ペア | 全 C2 ペア（`experiments.e3_pairs: all`）。H4 の主検定は McNemar のまま、ペア単位の感度解析を追加 | チェックポイント横断の一般化 | H4 の標本が増える |
| 2026-10-13 | 感度解析の追加 | 置換検定・ブートストラップのみ | 交差ランダム効果 LMM（H1〜H3）、提示順（P を先頭以外）、言い換え role、検出力解析モジュール | 査読で想定される頑健性の要求への対応 | なし（sensitivity） |
| 2026-10-13 | 正常 FT 強度スイープ | 計画書に記載のみ（CLI に引数なし） | `downstream_sft.py --num_samples / --learning_rate / --variant`。variant 付きチェックポイントは主解析から除外 | 計画書 §6.3 の頑健性比較の実施手段 | なし |
| 2026-10-14 | 研究計画書_最新 E1〜E4（提供者由来・動的制御）の実行経路 | 既存 `--mode cc` が E0〜E5 を一括実行する単一経路のみ | `--mode pc_e1 / pc_e2 / pc_e3 / pc_e4`（段階別に独立実行）と `pc_all`（4段階一括）を追加。各段階は E0 監査・`build_cc_model_set` を自前で行い `<out>/cc/<stage>/` に保存。既存 `cc` 経路・`results/` は不変 | ユーザ要望（E1〜E4 を分けて独立に動かす。固定トリガーではなく動的な命令制御を対象） | 既存の cc 一括経路・コミット済み結果は不変 |
| 2026-10-14 | 動的制御の対照群 G3（Runtime-Control） | 未実装（研究計画書_最新 §4.3） | `create_runtime_control_mock_agent`（dry-run）。署名を検証せず命令へ追従＝ソフトウェア層の制御。重みバックドア（提供者署名が必要）との対比で「ランタイム通信路」と「重み機構」を分離 | deep_research最新「Model Backdoor + Communication Channel + Agent Capability を三者分離」。通信だけで clean まで動けば prompt injection であり backdoor でない | dry-run Mock のみ（模擬値） |
| 2026-10-14 | 動的制御の指標 Δ_control / 提供者排他性(UAR) | CCR・FAR・selectivity のみ | `evaluation/provider_control_metrics.py`：Δ_control（実験モデルの命令追従率 − clean の偶発追従率、§9.2 の DiD）、unauthorized_activation_rate（非認証での追従率）、provider_exclusivity_margin（認証済み − 非認証）。未観測セルは 0 埋めせず None | 「通常のモデルでも起こる動作変化を差し引く」「提供者だけが制御できるか」を明示的に測るため（§9.2 / §23-③） | 新規指標の追加のみ。主解析・既存指標は不変 |
| 2026-10-14 | 表現レベルの継承（E1-B / E3-B） | 線形プローブ＋コサインのみ（CKA/RSA なし） | `circuits/representation_similarity.py`：線形 CKA（列中心化・回転スケール不変）、RSA（RDM=1−cos の上三角のスピアマン相関）、`run_representation_inheritance`（元→affected 派生 vs 元→clean 対照を分離）。PC-E1 は Conditional vs Clean、PC-E3 は元→派生の継承を測定。退化セルは 0 埋めせず None | 計画 §5.2(E1-B)・§7.3(E3-B) の CKA/RSA 要件。行動・因果とは独立に表現類似を測る | 新規指標の追加のみ。observational（構造的同一性は主張しない） |
