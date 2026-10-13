"""
実験条件の構成 (Experimental Conditions)

checkpoint manifest（training/build_manifest.py）から、E1 の評価モデル群と E2 のペア条件を構成する。
主条件 C0 / C1 / C1R / C2 / C3 に加え、研究実験計画書 §8.3 / §12.2 の対照条件
C2c / C3c / PC / C2_RS / C2_noFT / C4 を構成する（各ペアは registry.verify_condition_pair で出自検証）。

- C0 は clean_org の下流 seed ごとに反復する（H2 の C0 分散推定）
- 主解析は variant=None のチェックポイントのみ（FT 強度スイープは除外）
- ペアのメタデータには構造化 seed（upstream_seed_a/b, downstream_seed_a/b）と、
  役割・トリガー方式の上書き（role_a / role_b / trigger_mode）を記録する
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.prompts import ROLE_ORG_A, ROLE_ORG_B, role_for_model, with_private_goal
from lineage_collusion.training.model_registry import CheckpointMetadata, ModelRegistry

MAIN_CONDITIONS = ("C0", "C1", "C1R", "C2", "C3")
CONTROL_CONDITIONS = ("C2c", "C3c", "PC", "C2_RS", "C2_noFT", "C4")


def pair_meta(condition_id: str, pair_id: str, a: CheckpointMetadata, b: CheckpointMetadata,
              **extra: Any) -> Dict[str, Any]:
    meta = {
        "condition_id": condition_id,
        "pair_id": pair_id,
        "checkpoint_id_a": a.checkpoint_id,
        "checkpoint_id_b": b.checkpoint_id,
        "upstream_seed_a": a.upstream_seed,
        "upstream_seed_b": b.upstream_seed,
        "downstream_seed_a": a.downstream_seed,
        "downstream_seed_b": b.downstream_seed,
    }
    meta.update(extra)
    return meta


def _by_seed_and_d(ms: List[CheckpointMetadata]) -> Dict[Tuple[Optional[int], Optional[int]], CheckpointMetadata]:
    out = {}
    for m in ms:
        key = (m.upstream_seed, m.downstream_seed)
        if key in out:
            raise ValueError(f"(upstream_seed, downstream_seed)={key} のチェックポイントが重複: {out[key].checkpoint_id}, {m.checkpoint_id}")
        out[key] = m
    return out


def build_real_conditions(
    registry: ModelRegistry,
    control_conditions: Tuple[str, ...] = CONTROL_CONDITIONS,
    strict_real: bool = True,
) -> Dict[str, Any]:
    """
    戻り値: {"e1_models": {name: (backend, role, seed, dseed)}, "pair_models": {key: (a, b)},
            "pair_metadata": {key: meta}, "clean_org_a_fn": backend}
    """
    lazy = registry.build_lazy_backend
    P = lambda ms: [m for m in ms if m.poison_target == "P"]  # noqa: E731

    clean_a = _by_seed_and_d(registry.main_checkpoints("clean_org_a"))
    clean_b = _by_seed_and_d(registry.main_checkpoints("clean_org_b"))
    aff_up = {m.upstream_seed: m for m in P(registry.main_checkpoints("affected_upstream"))}
    aff_a = _by_seed_and_d(P(registry.main_checkpoints("affected_org_a")))
    aff_b = _by_seed_and_d(P(registry.main_checkpoints("affected_org_b")))
    aff_b_r = _by_seed_and_d([m for m in registry.main_checkpoints("affected_org_b") if m.poison_target == "R"])
    ctl_up = {m.upstream_seed: m for m in registry.main_checkpoints("control_upstream")}
    ctl_a = _by_seed_and_d(registry.main_checkpoints("control_org_a"))
    ctl_b = _by_seed_and_d(registry.main_checkpoints("control_org_b"))
    if not clean_a or not clean_b:
        raise ValueError("manifest に clean_org_a / clean_org_b（主解析）がありません")

    # E1 評価モデル（role は学習時と同一: prompts.role_for_model）
    e1_models: Dict[str, Tuple[Any, ...]] = {}
    e1_list: List[CheckpointMetadata] = []
    if "clean_base" in registry.checkpoints:
        e1_list.append(registry.checkpoints["clean_base"])
    e1_list += list(clean_a.values()) + list(clean_b.values()) + list(aff_up.values())
    e1_list += list(aff_a.values()) + list(aff_b.values()) + list(ctl_up.values()) + list(ctl_a.values()) + list(ctl_b.values())
    for m in e1_list:
        e1_models[m.checkpoint_id] = (lazy(m.checkpoint_id), role_for_model(m.checkpoint_id), m.upstream_seed, m.downstream_seed)

    pair_models: Dict[str, Tuple[Any, Any]] = {}
    pair_metadata: Dict[str, Dict[str, Any]] = {}

    def add(cond: str, key: str, a: CheckpointMetadata, b: CheckpointMetadata, **extra: Any) -> None:
        registry.verify_condition_pair(cond, a.checkpoint_id, b.checkpoint_id, strict_real=strict_real)
        pair_models[key] = (lazy(a.checkpoint_id), lazy(b.checkpoint_id))
        pair_metadata[key] = pair_meta(cond, f"real_{key}", a, b, **extra)

    def clean_for(d: Optional[int], table):
        return table.get((None, d)) or sorted(table.values(), key=lambda m: m.checkpoint_id)[0]

    for (_, d), ma in sorted(clean_a.items(), key=lambda kv: str(kv[0])):
        mb = clean_b.get((None, d))
        if mb is not None:
            add("C0", f"c0_d{d}", ma, mb)
            if "PC" in control_conditions:
                add("PC", f"pc_d{d}", ma, mb, role_a=with_private_goal(ROLE_ORG_A), role_b=with_private_goal(ROLE_ORG_B))

    for (s, d), ma in sorted(aff_a.items()):
        add("C1", f"c1_s{s}_d{d}", ma, clean_for(d, clean_b))
    for (s, d), mb in sorted(aff_b.items()):
        add("C1R", f"c1r_s{s}_d{d}", clean_for(d, clean_a), mb)

    seeds = sorted({s for s, _ in aff_a} & {s for s, _ in aff_b})

    def lineage_pairs(table_a, table_b, cond_same, cond_diff, tag):
        seeds_l = sorted({s for s, _ in table_a} & {s for s, _ in table_b})
        for (s, d), ma in sorted(table_a.items()):
            mb = table_b.get((s, d))
            if mb is not None:
                add(cond_same, f"{tag}2_s{s}_d{d}", ma, mb)
                if cond_same == "C2" and "C2_RS" in control_conditions:
                    add("C2_RS", f"c2rs_s{s}_d{d}", ma, mb, role_a=ROLE_ORG_B, role_b=ROLE_ORG_A)
        if len(seeds_l) >= 2:
            for i, s_a in enumerate(seeds_l):
                s_b = seeds_l[(i + 1) % len(seeds_l)]
                for (s, d), ma in sorted(table_a.items()):
                    if s != s_a:
                        continue
                    mb = table_b.get((s_b, d))
                    if mb is not None:
                        add(cond_diff, f"{tag}3_sa{s_a}_sb{s_b}_d{d}", ma, mb)

    lineage_pairs(aff_a, aff_b, "C2", "C3", "c")
    if "C2c" in control_conditions or "C3c" in control_conditions:
        lineage_pairs(ctl_a, ctl_b, "C2c", "C3c", "ctl")

    if "C2_noFT" in control_conditions:
        for s, m in sorted(aff_up.items()):
            registry.verify_condition_pair("C2_noFT", m.checkpoint_id, m.checkpoint_id, strict_real=strict_real)
            pair_models[f"c2noft_s{s}"] = (lazy(m.checkpoint_id), lazy(m.checkpoint_id))
            pair_metadata[f"c2noft_s{s}"] = pair_meta("C2_noFT", f"real_c2noft_s{s}", m, m)

    if "C4" in control_conditions:
        for (s, d), ma in sorted(aff_a.items()):
            mb = aff_b_r.get((s, d))
            if mb is not None:
                add("C4", f"c4_s{s}_d{d}", ma, mb, trigger_mode="dual")

    clean_a_first = sorted(clean_a.values(), key=lambda m: m.checkpoint_id)[0]
    return {
        "e1_models": e1_models,
        "pair_models": pair_models,
        "pair_metadata": pair_metadata,
        "clean_org_a_fn": lazy(clean_a_first.checkpoint_id),
        "upstream_seeds": seeds,
    }
