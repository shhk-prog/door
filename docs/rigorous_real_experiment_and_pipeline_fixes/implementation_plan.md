# 本実験成立のための実証パイプライン・統計・学習修正 実装計画書 (Implementation Plan)

## 1. 目的とスコープ
ユーザーレビューで指摘された重大課題（P0：4件、P1：6件）を解消し、
研究仮説である「悪意ある元モデル提供者が、独立にFTされた複数の派生エージェントに対し、配布後も共通の制御能力を保持できるか」を、
シミュレーションの都合の良い前提ではなく、実際の因果性・統計的整合性・実LLM対応学習パイプラインとして成立させる。

---

## 2. 修正対象ファイルと詳細設計

### 2.1 P0-1: Mock祖先判定のシード固定依存解消 (`run_experiments.py`)
- **問題**:
  - `create_command_mock_agent` の `target_ancestor` が `101` に固定されていた。
  - `s=1` では `100+1=101` で認証されるが、`s=2..5` では `100+s != 101` となり `authed=False` に遮断され、`s2..s5` の共通祖先効果（C2）が全く検証できていなかった。
- **改修方針**:
  - `create_command_mock_agent` に渡す `target_ancestor` を、呼び出し側で `ancestor_seed` と一致させるか、ペアごとの攻撃対象祖先（`target_ancestor = 100 + s`）として明示的に指定。
  - これにより、`s1..s5` のすべてにおいて、共通祖先ペア（C2: shared）は認証が成立し、独立祖先ペア（C3: independent）は別祖先のため認証が遮断される因果構造を正しく再現する。

### 2.2 P0-2: `statistical_tests.py` の `_follow_rate_by_descendant` における厳格条件の適用
- **問題**:
  - `_follow_rate_by_descendant` で `r.chosen_provider == r.command_target` のみを集計しており、「本来の最適解と命令先が偶然一致するシナリオ（約50%）」が含まれていた。
  - 結果、Cleanモデルの追従率が通常集計（Strict）では 0.00 なのに、仮説検定集計では約 0.48 となり不整合だった。
- **改修方針**:
  - `_follow_rate_by_descendant` に `require_suboptimal: bool = True` を適用し、`r.is_target_suboptimal`（`r.command_target != r.optimal_provider`）のシナリオのみで追従率を計算。
  - これにより、正常モデルの偶然一致を完全排除し、通常集計と仮説検定の数値を完全一致させる。

### 2.3 P0-3: `demo/channel_lib.py` における模擬値と実測値の明確な区別と実測ローダー
- **問題**:
  - FT残存率や複数派生先への制御率がハードコードされた固定確率から計算され、実測値と混同されていた。
- **改修方針**:
  - 戻り値に `"data_source": "simulation"` を明記。
  - 実測結果ファイル（`results/cc/` や manifest）を読み込んで実測値ベースでメトリクスを構築する `load_real_evaluation_results()` を実装。

### 2.4 P0-4: `upstream_sft.py` の既定学習データを Programmable コマンド対応に同期
- **問題**:
  - `--train_data` の既定値が古い固定ペイロード用データ（`upstream_sft_train.jsonl`）になっていた。
- **改修方針**:
  - 既定値を `lineage_collusion/data/generated/cc_upstream_sft_train.jsonl` に更新。
  - `--mode`（`programmable` / `fixed`）を追加し、プログラマブル上流SFTをデフォルトとして安全に実行できるようにする。

### 2.5 P1-1: `demo/app.js` の一斉制御表示をプログラマブル命令・実推論データ連動へ改修
- **改修方針**:
  - 「企業Pを優遇」固定のJavaScriptコードを改修し、画面上で選択された命令コード（`favor:P`, `favor:Q`, `favor:R`, `block:R` 等）に応じてエージェントの出力・理由テキストが動的に変化するようにする。

### 2.6 P1-2: デモと本体の命令集合・汎化階層の完全統一 (`demo/channel_lib.py`)
- **改修方針**:
  - `demo/channel_lib.py` の `SEEN_COMMANDS` および `HELDOUT_COMMANDS` を、`command_channel.py` の定義（`favor:P`, `favor:Q`, `reweight:price`, `reweight:quality`, `neutral` を既知、`favor:R`, `block:*` を未知）と完全一致させる。

### 2.7 P1-3: 提供者認証の性格（秘密ネームスペース認識）の明示
- **改修方針**:
  - ドキュメントおよびコメントにおいて、本認証は「暗号的電子署名ではなく、LLMが学習段階で獲得した特定の秘密活性化ネームスペース/構文パターンの認識」であることを明記。

### 2.8 P1-4: Chat Template のトークン境界とアシスタントヘッダーの厳密化 (`upstream_sft.py`)
- **改修方針**:
  - `tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)` を使用し、アシスタント生成開始トークンまでを厳密に -100 マスク。

### 2.9 P1-5: `demo/real_eval.py` のプログラマブルコマンドチャネル対応と HTML 側の実測切替
- **改修方針**:
  - `demo/real_eval.py` をコマンドチャネル形式（K0〜K5, E1-CC, E2-CC）に改修し、HTML デモ側で実測結果の表示をサポート。

### 2.10 P1-6: `peft` のオプショナルインポート化
- **改修方針**:
  - `llm_backend.py`, `upstream_sft.py`, `downstream_sft.py` で `peft` を try-except でインポートし、未インストール環境でも pytest のテスト収集が停止しないようにフォールバック。

---

## 3. テストと検証方針
1. pytest による全単体テスト（100件＋新テスト）の実行・合格
2. `python run_experiments.py --mode cc --dry_run` による Mock 実験実行
   - `s1..s5` の全ペアで `shared` の認証と増幅が動作し、`independent` と明確に分離されることの確認
   - 仮説検定（HC1〜HC3）において、Cleanモデルの追従率が通常集計と完全に一致することの確認
