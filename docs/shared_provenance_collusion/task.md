# タスクリスト: 実装レビューに基づく重大欠陥（P0〜P1）の全面改修

## ステータス概要
- **プロジェクト**: Shared Provenance, Shared Agenda? (LineageColludeBench)
- **環境**: Python 3.12.3 / NVIDIA H200 NVL (141GB VRAM) / uv仮想環境 (`.venv`)
- **改修完了日**: 2026-10-12
- **ステータス**: レビュー指摘全項目（P0-1〜P0-8, P1-1〜P1-6）の改修およびシミュレーション検証完了

---

## 改修タスク一覧

### Phase 1: データ生成・効用計算・トリガー切り替えの根本改修 (P0-1, P1-1, P1-2) <!-- id: fix-phase1 -->
- [x] `utility.py`: プロンプト提示数値と完全一致する離散重み（5%刻み整数）、同点時のP優遇ロジック完全排除、is_tieフラグ導入 <!-- id: fix-p1-1 -->
- [x] `scenario_generator.py`: Q/R最適比率の50:50完全均衡化 (Q: 150件, R: 150件)、Trigger ON/OFF専用関数 (`apply_trigger_condition`) の実装 <!-- id: fix-p1-2 -->
- [x] `build_datasets.py`: 重み丸め誤差ゼロ化、設定YAMLからの完全引数渡し、データ再生成 <!-- id: fix-p1-3 -->

### Phase 2: 集約器・無効出力ハンドリングの改修 (P0-6, P1-6) <!-- id: fix-phase2 -->
- [x] `aggregation.py`: Borda集約のタイブレークでP優遇（-ord(p)）を完全排除。同点（Tie）の明示的フラグ保持と固定シードによる中立タイブレーク <!-- id: fix-p2-1 -->
- [x] `agent_runner.py`: JSONパース失敗時にデフォルトQ補完する処理を全廃 (`choice=None, is_valid=False`) <!-- id: fix-p2-2 -->

### Phase 3: 通信プロトコルと再判断プロンプトの改修 (P0-4, P0-5) <!-- id: fix-phase3 -->
- [x] `communication.py`: 再判断（Step 3）プロンプトに元の全企業スペック・重み・トリガーメタデータを必ず再提示 <!-- id: fix-p3-1 -->
- [x] `agent_runner.py`: `communication_rounds: 1/2` に対応したモデル主導の複数ラウンド対話メッセージ生成ループ実装 <!-- id: fix-p3-2 -->

### Phase 4: モデル台帳と実験条件マッピングの改修 (P0-2, P0-3, P1-5) <!-- id: fix-phase4 -->
- [x] `model_registry.py`: チェックポイント台帳（上流seed、下流FT、重みハッシュ、組織A/B）の新規導入とC0〜C3条件の自動検証 <!-- id: fix-p4-1 -->
- [x] `llm_backend.py`: アダプタ非存在時の即時FileNotFoundError停止、親モデルとアダプタの整合性検証 <!-- id: fix-p4-2 -->
- [x] `runner.py`: E1専用の単独推論関数（余分な推論を完全排除）、C0〜C3の厳密なモデルペアリング (C2: 同一上流ハッシュ, C3: 異なる上流ハッシュ) <!-- id: fix-p4-3 -->

### Phase 5: E3因果メッセージ介入の厳密化 (P0-8) <!-- id: fix-phase5 -->
- [x] `message_replay.py`: 受信者（Agent B）の初期状態をFreezeし同一状態に対するM0〜M4注入の保証 <!-- id: fix-p5-1 -->
- [x] `message_replay.py`: M3（推薦完全除去）の正規表現置換、M4（M0と厳密に文字数を揃えた中立対照）の生成 <!-- id: fix-p5-2 -->

### Phase 6: 統計検定モジュールの数理的再実装 (P0-7) <!-- id: fix-phase6 -->
- [x] `statistical_tests.py`: H1（Inheritance Effect差）、H2（$\Delta_{\mathrm{interaction}} = \Gamma(C2) - \Gamma(C0)$）、H3（$\Delta_{\mathrm{provenance}} = \Gamma(C2) - \Gamma(C3)$）、H4（M0 vs M1〜M4）を研究計画の定義通り厳密に検定 <!-- id: fix-p6-1 -->
- [x] `statistical_tests.py`: 分散0時のp=0.0001代入等の不正処理を全廃、サンプル0件バグ（M0_actual vs M0）の解消、推定不能時は`NOT_ESTIMABLE`を出力 <!-- id: fix-p6-2 -->

### Phase 7: 包括的単体テスト・シミュレーション検証 (Section 5) <!-- id: fix-phase7 -->
- [x] 新規単体テスト群の実装（Trigger ON/OFF差分テスト、Bordaタイブレーク中立性、モデル台帳検証、E3状態固定検証等）: 7/7 PASSED <!-- id: fix-p7-1 -->
- [x] 模擬的真の効果に対するシミュレーションテスト（H1〜H4がESTIMABLEとなり、H2およびH4で有意差を正確に検出） <!-- id: fix-p7-2 -->
- [x] `walkthrough.md` への全改修記録の追記 <!-- id: fix-p7-3 -->
