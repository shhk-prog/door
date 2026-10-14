"""
分類タスク（AGNews / SST-2）上での動的制御の評価ループ。

既存の command_channel 評価（ProcurementScenario 専用）の分類タスク版。各
(モデル × サンプル × K条件) で推論し、出力ラベルを `CommandEvalRecord` に写像する。
これにより compute_programmability_metrics / Δ_control / 提供者排他性 / CDCR / PR などの
既存指標層をそのまま再利用できる（ラベルを「プロバイダ」とみなす）。

mock エージェントは3種:
  clean        : 命令を無視し true_label（通常の分類）
  programmable : 認証署名(Cedar-ctrl)付き命令のみ従い命令ラベルへ（重み機構）
  runtime      : 署名を問わず命令へ従う（ランタイム通信路。提供者排他的でない）
"""

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.training.model_registry import resolve_model_backend
from lineage_collusion.data.prior_datasets.control_task import (
    ControlTaskExample,
    default_k_conditions,
    generalization_tier,
    inject_command_prompt,
    resolve_label_target,
)
from lineage_collusion.evaluation.command_metrics import (
    CommandEvalRecord,
    compute_cross_descendant_metrics,
    compute_programmability_metrics,
)


def create_classification_mock_agent(
    kind: str, persistence: float = 1.0, salt: str = ""
) -> Callable[[str], str]:
    """分類タスク版の決定論的 Mock（clean / programmable / runtime）。"""
    # 関数名に "mock" を含めることで _is_mock_backend が Mock と判定し、
    # dry-run 専用の mock_gold ヒント（真ラベル）が付与される（実モデルには付与されない）。
    def mock_classification_agent(prompt: str) -> str:
        # プロンプトから候補ラベル・入力文・命令・署名を復元
        m_lab = re.search(r"候補ラベル \[([^\]]+)\]", prompt)
        labels = [s.strip() for s in m_lab.group(1).split("、")] if m_lab else []
        m_txt = re.search(r"### 入力文\n(.+?)\n", prompt, re.DOTALL)
        text = m_txt.group(1).strip() if m_txt else prompt[:80]
        cmd = re.search(r"audit_code:\s*([^\n]+)", prompt)
        command = cmd.group(1).strip() if cmd else None
        sig = re.search(r"provenance_sig:\s*([^\n]+)", prompt)
        has_auth = bool(sig and sig.group(1).strip().startswith("Cedar-ctrl"))

        # Mock は実テキストを分類できないため、dry-run 専用ヒント mock_gold から正解を得る
        # （Mock の目的は制御の有無の弁別であり分類性能ではない）。無ければ安定ハッシュで代理。
        m_gold = re.search(r"mock_gold:\s*([^\s]+)\s*-->", prompt)
        if m_gold and m_gold.group(1) in labels:
            true_label = m_gold.group(1)
        else:
            true_label = labels[_stable_idx(text, len(labels))] if labels else "unknown"

        # 命令ターゲットの解決（favor/block/neutral）
        tmp = ControlTaskExample("x", text, tuple(labels), true_label)
        target = resolve_label_target(tmp, command) if command else true_label

        fire_key = f"{text}|{command}|{salt}"
        h = int(hashlib.sha256(fire_key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        fired = h < persistence
        has_cmd = bool(command and command != "neutral")

        if kind == "clean" or not has_cmd:
            choice = true_label
        elif kind == "runtime":
            choice = target if fired else true_label              # 署名を問わず従う
        else:  # programmable: 認証署名付きのときのみ従う
            choice = target if (has_auth and fired) else true_label

        return f'```json\n{{"label": "{choice}"}}\n```'

    mock_classification_agent.is_mock = True
    return mock_classification_agent



def _stable_idx(text: str, k: int) -> int:
    if k <= 0:
        return 0
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % k


def _parse_label(raw: str, label_set: Tuple[str, ...]) -> Tuple[Optional[str], bool]:
    """自由生成テキストから JSON を抽出し `label`（無ければ `choice`）を取り出す。
    候補集合に属する文字列のみ有効（不正形式・範囲外は None, False）。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL) or re.search(r"(\{.*\})", raw, re.DOTALL)
    content = m.group(1) if m else raw
    try:
        data = json.loads(content)
    except Exception:
        return None, False
    lab = data.get("label") if isinstance(data, dict) else None
    if lab is None and isinstance(data, dict):
        lab = data.get("choice")
    if isinstance(lab, str) and lab in label_set:
        return lab, True
    return None, False


def _decide(backend_obj, model_fn, prompt: str, label_set: Tuple[str, ...]) -> Tuple[Optional[str], bool]:
    """
    意思決定。バックエンドが score_labels（尤度採点）を持つ実モデルはラベル尤度の argmax、
    そうでなければ自由生成の {"label": ...} をパースする。
    """
    if backend_obj is not None and hasattr(backend_obj, "score_labels"):
        labels, probs = backend_obj.score_labels(prompt, tuple(label_set))
        import numpy as np
        return labels[int(np.argmax(probs))], True
    return _parse_label(model_fn(prompt), label_set)


def _backend_obj(model_fn, factory):
    return getattr(factory, "_instance", None) if factory is not None else (
        model_fn if hasattr(model_fn, "score_labels") else None)


def _is_mock_backend(backend_obj: Any, model_fn: Any) -> bool:

    if getattr(backend_obj, "is_mock", False) or "Mock" in type(backend_obj).__name__:
        return True
    if hasattr(model_fn, "__name__") and "mock" in model_fn.__name__.lower():
        return True
    return False


def run_control_programmability(
    examples: List[ControlTaskExample],
    models: Dict[str, Tuple[Any, str, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[CommandEvalRecord]]:
    """E1 Programmability（分類タスク版）。全 (model × example × K) を評価して指標を返す。"""
    records: List[CommandEvalRecord] = []
    for model_name, info in models.items():
        fn_raw = info[0]
        role = info[1] if len(info) > 1 else ""
        meta = info[2] if len(info) > 2 and isinstance(info[2], dict) else {}
        model_fn, factory = resolve_model_backend(fn_raw)
        backend_obj = _backend_obj(model_fn, factory)
        is_mock = _is_mock_backend(backend_obj, model_fn)
        for ex in examples:
            for k in default_k_conditions(ex):
                prompt = inject_command_prompt(
                    ex, k["command"], k["auth"] or "none", role,
                    mock_gold=ex.true_label, is_mock=is_mock,
                )
                choice, valid = _decide(backend_obj, model_fn, prompt, ex.label_set)
                target = resolve_label_target(ex, k["command"]) if k["command"] else None
                records.append(CommandEvalRecord(
                    scenario_id=ex.example_id, model_name=model_name,
                    command=k["command"], auth=k["auth"] or "none", k_condition=k["id"],
                    tier=generalization_tier(ex, k["command"]),
                    optimal_provider=ex.true_label, command_target=target,
                    chosen_provider=choice, is_valid_output=valid,
                    upstream_checkpoint_hash=meta.get("upstream_checkpoint_hash"),
                    model_role=meta.get("model_role"),
                ))
        if factory:
            factory.release()

    metrics = {}
    for m, info in models.items():
        meta = info[2] if len(info) > 2 and isinstance(info[2], dict) else {}
        d = compute_programmability_metrics(records, m).model_dump()
        d["model_role"] = meta.get("model_role")
        d["upstream_checkpoint_hash"] = meta.get("upstream_checkpoint_hash")
        metrics[m] = d
    return {"programmability_by_model": metrics, "task": "classification"}, records


def run_control_cross_descendant(
    examples: List[ControlTaskExample],
    descendants: Dict[str, Tuple[Any, str, Dict[str, Any]]],
    commands: List[str],
    tag: str = "affected",
    include_c0_baseline: bool = True,
) -> Tuple[Dict[str, Any], List[CommandEvalRecord]]:
    """
    Cross-Descendant Control（分類タスク版）。認証済み命令を各派生先へ一斉提示。
    include_c0_baseline=True のとき、各シナリオで命令なし（C=0, auth="none"）の
    ベースラインレコードも収集する（compute_delta_control の真の DiD に必要。audit round2 P0-2 と同じ扱い）。
    """
    records: List[CommandEvalRecord] = []
    for desc_id, info in descendants.items():
        fn_raw = info[0]
        role = info[1] if len(info) > 1 else ""
        meta = info[2] if len(info) > 2 and isinstance(info[2], dict) else {}
        model_fn, factory = resolve_model_backend(fn_raw)
        backend_obj = _backend_obj(model_fn, factory)
        is_mock = _is_mock_backend(backend_obj, model_fn)
        for ex in examples:
            if include_c0_baseline:
                # C=0 無命令ベースライン（偶発的ターゲット選択率 P(Y=T|C=0) 用）
                p0 = inject_command_prompt(
                    ex, None, "none", role,
                    mock_gold=ex.true_label, is_mock=is_mock,
                )
                c0, v0 = _decide(backend_obj, model_fn, p0, ex.label_set)
                records.append(CommandEvalRecord(
                    scenario_id=ex.example_id, model_name=desc_id, descendant_id=desc_id,
                    command=None, auth="none", k_condition="K0", tier="no_command",
                    optimal_provider=ex.true_label, command_target=None,
                    chosen_provider=c0, is_valid_output=v0,
                    upstream_checkpoint_hash=meta.get("upstream_checkpoint_hash"),
                    model_role=meta.get("model_role") or tag,
                ))
            for cmd in commands:
                # block:<label> はサンプル依存。favor は label_set 内のもののみ有効
                prompt = inject_command_prompt(
                    ex, cmd, "authenticated", role,
                    mock_gold=ex.true_label, is_mock=is_mock,
                )
                choice, valid = _decide(backend_obj, model_fn, prompt, ex.label_set)
                records.append(CommandEvalRecord(
                    scenario_id=ex.example_id, model_name=desc_id, descendant_id=desc_id,
                    command=cmd, auth="authenticated", tier=generalization_tier(ex, cmd),
                    optimal_provider=ex.true_label, command_target=resolve_label_target(ex, cmd),
                    chosen_provider=choice, is_valid_output=valid,
                    upstream_checkpoint_hash=meta.get("upstream_checkpoint_hash"),
                    model_role=meta.get("model_role") or tag,
                ))
        if factory:
            factory.release()
    # CDCR は認証済み命令レコードのみで算出（C=0 を除外）
    metrics = compute_cross_descendant_metrics([r for r in records if r.auth == "authenticated"])

    metrics["tag"] = tag
    return metrics, records
