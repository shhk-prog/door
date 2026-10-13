# emo2 知見の移植タスク (emo2_methodology_transfer)

## 概要
emo2 の測定・再現性・統計・内部回路分析の方法論を door に移植する（door のみ変更、emo2 は不変更）。
詳細設計は `implementation_plan.md`、実施結果は `walkthrough.md` を参照。

## タスクリスト

- [x] 1. 意思決定の尤度スコアリング（`agents/likelihood.py`、Step1/3/E1 の判断、soft 指標）
- [x] 2. 再現性基盤（`configs/models.yaml`、`utils/manifests.py`、`utils/io.py`、出力の退避・dry-run 隔離、revision 必須化）
- [x] 3. 既知の落とし穴対策（テンプレート変数固定、BOS 二重付与、dtype 統一、決定性設定）
- [x] 4. E1 ゲート（`analysis/gate.py`、`--force_after_no_go`、単独実行時の判定ファイル要求）
- [x] 5. 内部回路分析の作り直し（単一位置介入＋尤度評価、層別方向、発見/確認分割、ランダム方向対照、系譜横断転移）
- [x] 6. 統計・運用規約（暗黙補完の撤廃、置換検定・ブートストラップ感度解析、soft 版仮説、サマリースキーマ、decision_log）
- [x] 7. テスト追加・既存テスト更新・dry-run 完走確認
- [x] 8. CLAUDE.md / walkthrough.md 更新

## 追加で対応した事項
- [x] 既存バグ: `--mode cc` が `CommandEvalRecord.is_target_suboptimal`（存在しない属性）参照で停止 → 修正
- [x] H1 のゼロ分散時の退化 CI `[eff, eff]` → `None`（H3 と同一規約）
- [x] dry-run Mock の seed 間変動ゼロ → 上流 seed をハッシュに含める
- [x] `docs/decision_log.md` 新設
