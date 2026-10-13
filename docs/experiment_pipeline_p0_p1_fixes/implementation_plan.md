# 実装計画書: 実験パイプラインのP0/P1/P2課題修正と評価整合性の確保

## 1. 目的
`run_experiments.py`、`command_metrics.py`、`runner.py`、`statistical_tests.py`、`README_command_channel.md` において指摘された、本実験実施前の致命的課題（P0）、評価指標の妥当性課題（P1）、主張との整合性課題（P2）を体系的に解消し、実LLM本実験への移行が完全に検証可能なパイプラインを確立する。

---

## 2. 対象ファイル
1. `lineage_collusion/evaluation/command_metrics.py`
2. `lineage_collusion/experiments/runner.py`
3. `lineage_collusion/analysis/statistical_tests.py`
4. `run_experiments.py`
5. `README_command_channel.md`
6. `tests/test_command_channel.py` または関連テスト

---

## 3. 具体的な改訂内容

### 【P0: 本実験前の必須課題】
1. **P0-1: 実行モード判定の厳格化と誤認防止 (`run_experiments.py`)**
   - `--dry_run` なしで `--manifest` が指定されていない場合は、暗黙的にMock実行するのではなく、エラーメッセージを表示して即座に終了する。
   - すべての出力JSONメタデータに `"execution_mode": "mock" | "real"` を明記。
2. **P0-2: E1での全affected実モデル評価 (`run_experiments.py`)**
   - `prog_ckpts[:1]` という最初の1モデル固定を撤廃。
   - `prog_ckpts` の全モデル（または指定された全モデル）を `cc_models` に登録し、モデルごとのBUR/CCR/FAR/MIを個別評価・記録。
3. **P0-3: PR（Persistence Ratio）のパイプライン接続 (`command_metrics.py`, `run_experiments.py`)**
   - `pre-FT` モデル（`affected_upstream` 等）と `post-FT` 派生モデル（`affected_org_a/b` 等）のペアについて、CCR_post / CCR_pre を算出して `persistence_ratio_summary.json` として保存。
   - Dry-Run 時も pre (persistence=1.0) と post (persistence=0.8) の Mock で PR を計算し、テーブルに表示。
4. **P0-4: 実モデル実行時の E2 Collusion の接続 (`run_experiments.py`)**
   - 実モデルモードでも、manifestのメタデータから検証済みの C2（同一祖先派生）および C3（異祖先派生）のペアを自動抽出し、`exp_mgr.run_e2_command_collusion` を実行。
   - Lazy Backend により、ペア評価時のみメモリにロードし、使用後に解放（GPU OOM を防止）。
5. **P0-5: チェックポイント出自・独立性の厳格検証 (`run_experiments.py`)**
   - `upstream_seed` のみによる分類を改め、`ModelRegistry.verify_condition_pair` を用いて実重みディレクトリのSHA256ハッシュ（`upstream_hash`）と独立性を厳格検証。
   - 検証結果サマリーを記録・出力。
6. **P0-6: HC1 の検定単位の適正化 (`statistical_tests.py`)**
   - 全モデルのシナリオ単位プールをやめ、モデル単位で認証あり vs 認証なしの対応差を検定。
   - モデル別のHC1結果および、programmable モデル単独での検定結果を明示。

---

### 【P1: 評価の妥当性に関する課題】
1. **P1-1: 未観測（None）と観測値0.0の区別と無効出力追跡 (`command_metrics.py`)**
   - 該当データが0件の場合は 0.0 ではなく `None` を返却。
   - `n_authed_valid`, `n_unauth_valid`, `invalid_output_count` を構造体に追加し、未観測・失敗・追従0%を完全に区別。
2. **P1-2: CDCRの定義不整合解消 (`command_metrics.py`)**
   - エピソード単位の追従率 `cdcr_micro` に加え、派生モデルごとの平均追従率のマクロ平均 `cdcr_macro` と、所定の閾値（CCR > 0.5 等）を満たした派生モデルの割合 `cdcr_controlled_descendants_fraction`（$\#\text{controlled descendants} / \#\text{all descendants}$）を算出。
3. **P1-3: MIの標本バイアス補正 (`command_metrics.py`)**
   - Miller-Madow バイアス補正推定量 $\widehat{I}_{\text{MM}} = \max\left(0, \widehat{I} - \frac{(|C|-1)(|A|-1)}{2N \ln 2}\right)$ を導入。
   - 素のMIと補正後MIを併記。
4. **P1-4: HC2 の0埋めバイアス排除 (`statistical_tests.py`)**
   - `cln.get(s, 0.0)` による0埋めを廃止し、実際に両方のモデル群で観測されたシナリオ交差集合 `set(aff) & set(cln)` のみを対象に対応差を検定。
5. **P1-5: 談合現象の概念分離 (`runner.py`)**
   - 以下の3指標を明確に分離して記録：
     - `joint_p_choice_rate`: 企業Pへの共同選択率
     - `joint_target_compliance`: コマンド指定ターゲット（P/R等）への共同追従率
     - `communication_amplification_lift`: 通信ON時の共同追従率 − 通信OFF時の共同追従率

---

### 【P2: 研究上の主張との整合性 & READMEの適正化】
1. **P2-1: `README_command_channel.md` の記述適正化**
   - ネットワーク経由の遠隔制御ではなく、合成コンテキスト内の露出（Direct/Environmental/Relay）であることの明記。
   - 内部回路の因果的同定ではなく、行動レベルのコマンド汎化の検証であることの明記。
   - Mock設定値と実測値の峻別、現在のコードが保証する範囲の正確な記述。

---

## 4. 検証計画
- ユニットテストおよび統合テストの実行（`pytest` にて全テストが通過することを確認）。
- `python run_experiments.py --mode cc --dry_run` を実行し、すべての新規指標（PR, CDCR マクロ/ミクロ, MI補正, 分離Collusion指標）が正しく出力されることを確認。
- マニフェスト未指定時の実モードエラー停止を確認。
- `walkthrough.md` に結果を文書化。
