"""Seeded hard-negative minting: keep the sample, rewrite the POPE probe."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from random import Random

from visioneval.traps.types import TrapRecord, make_trap_id, utc_now

DISTRACTORS = (
    "cat",
    "dog",
    "car",
    "tree",
    "person",
    "bicycle",
    "airplane",
    "chair",
    "bottle",
    "clock",
    "unicorn",
    "zeppelin",
)


def _rng(seed: int, trap_id: str) -> Random:
    material = hashlib.sha256(f"{seed}|{trap_id}".encode("utf-8")).digest()
    return Random(int.from_bytes(material[:8], "big"))


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def _pope_prompts(object_name: str) -> list[str]:
    article = _article(object_name)
    return [
        f"Is there {article} {object_name} in the image?",
        f"Is there {article} {object_name} in this picture?",
        f"Does the image contain {article} {object_name}?",
        f"Can you see {article} {object_name} in the image?",
    ]


def _pick_prompt(object_name: str, rng: Random, current: str) -> str:
    variants = _pope_prompts(object_name)
    index = rng.randrange(len(variants))
    prompt = variants[index]
    if prompt == current:
        prompt = variants[(index + 1) % len(variants)]
    return prompt


def generate_hard_negative(trap: TrapRecord, *, seed: int = 0) -> TrapRecord:
    """Mint one deterministic POPE variant of ``trap`` (same sample, new probe).

    * Present-object POPE traps keep the object and rewrite the wording.
    * Otherwise swap in a seeded absent-object distractor.
    """
    rng = _rng(seed, trap.trap_id)
    present = list(trap.expected_objects)
    absent = list(trap.absent_objects)
    used = {item.lower() for item in present + absent}

    expected_present = bool(trap.extra.get("expected_present")) if trap.probe_type == "pope" else False
    object_name = str(trap.extra.get("object_name") or "")
    if expected_present:
        object_name = object_name or (present[0] if present else "object")
        prompt = _pick_prompt(object_name, rng, trap.prompt)
    else:
        pool = [name for name in DISTRACTORS if name not in used]
        if object_name and object_name.lower() in used:
            pool = [name for name in pool if name != object_name.lower()]
        if not pool:
            pool = ["widget"]
        object_name = pool[rng.randrange(len(pool))]
        prompt = _pick_prompt(object_name, rng, trap.prompt)
        if object_name not in absent:
            absent.append(object_name)

    disc = f"hn-{object_name}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]}"
    now = utc_now()
    extra = dict(trap.extra)
    extra.update(
        {
            "object_name": object_name,
            "expected_present": expected_present,
            "parent_trap_id": trap.trap_id,
            "hard_negative": True,
            "seed": seed,
            "source": "hard_negative",
        }
    )
    return TrapRecord(
        trap_id=make_trap_id(trap.model, trap.sample_id, "pope", disc),
        model=trap.model,
        sample_id=trap.sample_id,
        image_hash=trap.image_hash,
        prompt=prompt,
        expected_objects=tuple(present),
        absent_objects=tuple(absent),
        probe_type="pope",
        last_outcome="fail",
        fail_count=1,
        consecutive_passes=0,
        created_at=now,
        updated_at=now,
        retired=False,
        extra=extra,
    )


def mint_hard_negatives(traps: Sequence[TrapRecord], *, seed: int = 0) -> list[TrapRecord]:
    """One variant per trap. Same seed always yields the same ids and prompts."""
    minted: list[TrapRecord] = []
    seen = {trap.trap_id for trap in traps}
    for trap in traps:
        variant = generate_hard_negative(trap, seed=seed)
        if variant.trap_id in seen:
            continue
        seen.add(variant.trap_id)
        minted.append(variant)
    return minted
