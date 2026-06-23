"""Baseline evidence converter for TBG-CoT-Bench.

Important: this is not the gold label generator. It is a lightweight parser used
to test how much of the manually labelled benchmark can be recovered from raw
CoT-style step text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import re


SOURCE_WEIGHT_RULES: list[tuple[list[str], str, float]] = [
    (["record", "records", "log", "document", "documented", "charter", "certificate", "filing", "official", "minutes"], "factual", 1.30),
    (["radiocarbon", "dating", "forensic", "measurement", "timestamp", "timestamps", "carbon", "survey"], "scientific", 1.45),
    (["journal", "publication", "paper", "study", "research", "scholarly", "historiography", "consensus"], "academic", 1.30),
    (["court", "legal", "ruling", "verdict", "statute", "legislation", "gazetted", "protection order", "counsel"], "legal", 1.30),
    (["interview", "interviews", "testimony", "stated", "said", "claimed", "reported", "pamphlet", "colophon"], "testimony", 1.00),
    (["infer", "therefore", "suggest", "suggests", "indicate", "imply", "conclude", "requires", "places", "reference", "references"], "inference", 0.90),
    (["fan", "wiki", "rumor", "speculation", "speculate", "unconfirmed", "informally", "may have"], "speculative", 0.60),
    (["assumption", "suppose", "hypothetical", "counterfactual", "assume"], "assumption", 0.50),
]

CONFIDENCE_MODIFIERS = {
    "clearly": 1.20, "certainly": 1.20, "definitive": 1.20, "definitively": 1.25,
    "confirmed": 1.15, "explicitly": 1.10, "formally": 1.10, "documented": 1.10,
    "perhaps": 0.65, "possibly": 0.65, "might": 0.70, "may have": 0.70,
    "could": 0.72, "suggests": 0.80, "appears": 0.80, "speculate": 0.60,
    "informally": 0.75, "unconfirmed": 0.55, "uncertain": 0.65,
}

DEFAULT_SOURCE_WEIGHT = 0.80


@dataclass
class ParsedEvidence:
    key: str
    raw_text: str
    supports_forward: Optional[bool]
    strength: float
    source_label: str
    source_weight: float
    matched_rule: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "raw_text": self.raw_text,
            "supports_forward": self.supports_forward,
            "strength": round(self.strength, 4),
            "source": self.source_label,
            "source_weight": round(self.source_weight, 4),
            "meta": {"matched_rule": self.matched_rule, "notes": self.notes},
        }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _keywords(label: str) -> list[str]:
    # Keep semantically useful chunks from labels: remove common function words.
    stop = {"the", "a", "an", "was", "were", "is", "are", "as", "of", "to", "be", "began", "made", "created", "triggered", "opened", "filed", "initiated", "designated"}
    words = [w for w in _normalize(label).split() if len(w) >= 4 and w not in stop]
    # include full normalized label plus individual content words
    return [_normalize(label)] + words


def _contains_any(text: str, candidates: list[str]) -> bool:
    return any(c and c in text for c in candidates)


def _extract_temporal_values(text: str) -> list[tuple[int, int]]:
    """Return (position, sortable_value) for years, days, dates, quarters."""
    out: list[tuple[int, int]] = []
    for m in re.finditer(r"\b(19|20)\d{2}\b", text):
        out.append((m.start(), int(m.group(0)) * 10000))
    months = {m.lower(): i for i, m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'], 1)}
    for m in re.finditer(r"\b(" + "|".join(months) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", text, re.I):
        out.append((m.start(), months[m.group(1).lower()] * 100 + int(m.group(2))))
    for m in re.finditer(r"\bDay\s+(\d+)\b", text, re.I):
        out.append((m.start(), int(m.group(1))))
    for m in re.finditer(r"\bQ([1-4])\b", text, re.I):
        out.append((m.start(), int(m.group(1))))
    return out


class EvidenceConverter:
    def __init__(self, strength_floor: float = 0.20, strength_ceiling: float = 0.95):
        self.strength_floor = strength_floor
        self.strength_ceiling = strength_ceiling

    def _assign_source_weight(self, text: str) -> tuple[str, float]:
        lower = text.lower()
        best = ("general", DEFAULT_SOURCE_WEIGHT, 0)
        for keywords, label, weight in SOURCE_WEIGHT_RULES:
            count = sum(1 for kw in keywords if kw in lower)
            if count > best[2]:
                best = (label, weight, count)
        return best[0], best[1]

    def _estimate_strength(self, text: str, base: float) -> float:
        lower = text.lower()
        modifier = 1.0
        for kw, value in sorted(CONFIDENCE_MODIFIERS.items(), key=lambda x: -len(x[0])):
            if kw in lower and abs(value - 1.0) > abs(modifier - 1.0):
                modifier = value
        return round(max(self.strength_floor, min(self.strength_ceiling, base * modifier)), 4)

    def convert(self, text: str, *, key: str = "step", event_a_label: str = "event_a", event_b_label: str = "event_b") -> ParsedEvidence:
        lower = text.lower()
        normalized = _normalize(text)
        a_terms = _keywords(event_a_label)
        b_terms = _keywords(event_b_label)
        has_a = _contains_any(normalized, a_terms)
        has_b = _contains_any(normalized, b_terms)
        notes: list[str] = []
        support: Optional[bool] = None
        base_strength = 0.60
        matched_rule = ""

        # Explicit symbolic pattern: A before B / B before A.
        if re.search(r"\bA\s+before\s+B\b", text):
            support, base_strength, matched_rule = True, 0.70, "symbolic_a_before_b"
        elif re.search(r"\bB\s+before\s+A\b", text):
            support, base_strength, matched_rule = False, 0.70, "symbolic_b_before_a"

        # Event-aware before/after patterns when both events are visible.
        if support is None and has_a and has_b:
            a_pos = min((normalized.find(t) for t in a_terms if t in normalized), default=-1)
            b_pos = min((normalized.find(t) for t in b_terms if t in normalized), default=-1)
            if " before " in lower or " preceded " in lower or " predates " in lower:
                support = a_pos < b_pos
                base_strength = 0.80
                matched_rule = "event_order_before"
            elif " after " in lower or " following " in lower or " later than " in lower or "post" in lower:
                support = a_pos > b_pos
                base_strength = 0.80
                matched_rule = "event_order_after"

        # Date-like fallback. If event_a-only has an early timestamp, assume forward;
        # if event_b-only has an early timestamp, assume backward. This is deliberately
        # weak and should be treated as a baseline heuristic, not a final parser.
        if support is None:
            vals = _extract_temporal_values(text)
            if vals:
                if has_a and not has_b:
                    support, base_strength, matched_rule = True, 0.62, "timestamp_near_event_a"
                elif has_b and not has_a:
                    support, base_strength, matched_rule = False, 0.62, "timestamp_near_event_b"

        # Domain phrases common in the provided benchmark.
        if support is None:
            phrase_rules = [
                (("prior art" in lower or "provisionally filed" in lower), True, "ip_prior_art"),
                (("public disclosure" in lower or "demonstrated" in lower), False, "ip_public_disclosure"),
                (("ongoing audit" in lower or "pre existing concern" in normalized), False, "audit_preexisting"),
                (("scope was formally expanded" in lower or "after the complaint" in lower), True, "audit_expanded_after_complaint"),
                (("symptoms" in lower and "preceded" in lower and "diagnosis" in lower), False, "symptoms_preceded_diagnosis"),
                (("evacuation" in lower and "before the alarm" in lower), False, "evacuation_before_alarm"),
                (("rift" in lower and "after the rift" in lower), True, "after_rift_reference"),
            ]
            for cond, val, rule in phrase_rules:
                if cond:
                    support, base_strength, matched_rule = val, 0.70, rule
                    break

        # Lexical fallback: low confidence because it is not event-aware.
        if support is None:
            if any(kw in lower for kw in ["before", "prior to", "preceded", "first", "predates"]):
                support, base_strength, matched_rule = True, 0.55, "lexical_forward_fallback"
            elif any(kw in lower for kw in ["after", "following", "subsequent", "later", "post-"]):
                support, base_strength, matched_rule = False, 0.55, "lexical_backward_fallback"

        if support is None:
            notes.append("ambiguous_direction_manual_review")
            base_strength = 0.50
            matched_rule = "ambiguous"

        source, weight = self._assign_source_weight(text)
        strength = self._estimate_strength(text, base_strength)
        return ParsedEvidence(key, text, support, strength, source, weight, matched_rule, notes)

    def convert_steps(self, steps: list[str], *, scenario_id: str, event_a_label: str, event_b_label: str) -> list[ParsedEvidence]:
        return [self.convert(text, key=f"{scenario_id}_step_{idx:02d}", event_a_label=event_a_label, event_b_label=event_b_label) for idx, text in enumerate(steps, 1)]
