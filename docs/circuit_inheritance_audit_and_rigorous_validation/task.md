# タスクリスト: 内部回路因果同定と継承性・複数エージェント評価の厳密化 (監査指摘への完全対応)

## 状況・目的
- `door-main(20261013-081246).zip` に対するコード監査結果の指摘事項（P0-1〜P0-4, P1-1〜P1-3）を完全に解消する。
- 論文用本実験に耐えうる厳密な因果検証、線形プロービング、通常選択と偶発一致の分離、実モデル対応の堅牢化を行う。

---

## タスク進行状況

- [x] 1. 計画策定とドキュメント作成
  - [x] 新規トピックフォルダ作成 (`docs/circuit_inheritance_audit_and_rigorous_validation`)
  - [x] `task.md` 作成
  - [x] `implementation_plan.md` 作成
- [x] 2. P0-1: 実モデル回路解析バックエンドの健全化 (`real_circuit_backend.py`, `circuit_persistence.py`)
  - [x] 2.1 `control_direction` の実テンソルからの動的抽出・較正メソッドの実装（認証 vs 無認証の差分平均）
  - [x] 2.2 `compute_subspace_alignment` の `None` / 空ベクトル安全ハンドリング（型エラー防止）
  - [x] 2.3 実測可能指標と未測定指標のステータス明示
- [x] 3. P0-2: E3 線形プローブ（Linear Probe）精度の厳密化 (`activation_analyzer.py`)
  - [x] 3.1 事前射影閾値比較を廃止し、NumPy閉形式正則化リッジ回帰による Out-of-Sample (80/20 Train/Test) 線形プローブを実装
  - [x] 3.2 Cleanモデル（チャンスレベル）とAffectedモデル（高精度）の実測分類精度の算出
- [x] 4. P0-3: E3 H1/H2 判定および Specificity の実測化 (`causal_tracing.py`)
  - [x] 4.1 恣意的な「Layer 1〜5」固定仮定を廃止し、通常タスク実行時の活性化との実験的比較または判定保留（`Inconclusive`）化
  - [x] 4.2 `specificity_score` の固定値（0.95/1.0）を廃止し、介入時の通常タスク正解率維持率（Clean Acc Under Intervention / Baseline Clean Acc）の実測化
- [x] 5. P0-4: E4 集計における Clean 対照の混入防止 (`run_experiments.py`, `runner.py`)
  - [x] 5.1 派生モデル（Affected Descendants）と対照群（Clean Control）の分離フィルタリング
  - [x] 5.2 平均整列度（Mean Subspace Alignment）を派生群単独（0.989）で集計し、Clean対照（0.000）は比較対照値として独立表示
- [x] 6. P1-1: E0・E1 指標の厳密化（偶発的一致と正常選択の分離）
  - [x] 6.1 もともとの最適企業と命令先が異なるシナリオ（Non-Target Optimal: $O \neq T$）に限定した厳格 ASR / FAR / 転向率の導入
  - [x] 6.2 Cleanモデルが最適解を選択したことによる見かけの命令一致率（偶然の一致）を誤発火と誤認しない指標設計
- [x] 7. P1-2 & P1-3: E5 談合指標の厳密な解釈および CLI 体系の整理
  - [x] 7.1 「共同選択」「規則逸脱」「通信増幅」「談合（Collusion）」を学術的に正しく分離したサマリー表示
  - [x] 7.2 通信増幅が 0 の場合に「共通祖先由来の同期にとどまり談合未確認」と正確に注記
- [x] 8. ユニットテスト拡充と検証
  - [x] 8.1 修正箇所に対する単体テスト追加（全93件）
  - [x] 8.2 pytest 全件パス確認
  - [x] 8.3 Dry-run 実行確認とサマリー検証
  - [x] 8.4 `walkthrough.md` の作成
