"""Numeric grounding checker with Romanian number parser, exemptions, and tolerance."""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agent.config import GroundingConfig, load_agent_config

UNITS = {
    "mwh", "kwh", "wh", "kw", "mw", "ron", "lei", "m/s", "ms", "km/h",
    "w/m2", "w/m²", "c", "°c", "ore", "h", "zile", "luni", "ani", "%", "procente",
}

ROMANIAN_MONTHS = {
    "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie",
    "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie",
}

ROMANIAN_DAYS = {
    "luni", "marți", "marti", "miercuri", "joi", "vineri", "sâmbătă", "sambata", "duminică", "duminica",
}


@dataclass
class ExtractedNumber:
    raw_text: str
    value: float
    precision: int = 0
    unit: Optional[str] = None
    start: int = 0
    end: int = 0
    is_exempt: bool = False
    exemption_reason: Optional[str] = None


@dataclass
class GroundingResult:
    is_grounded: bool
    extracted_numbers: List[ExtractedNumber] = field(default_factory=list)
    unmatched_numbers: List[ExtractedNumber] = field(default_factory=list)
    tool_result_ids: List[str] = field(default_factory=list)
    warning_message: Optional[str] = None


class GroundingChecker:
    """Numeric grounding engine enforcing that every number in an answer traces to tool results."""

    def __init__(self, config: Optional[GroundingConfig] = None):
        if config is None:
            self.config = load_agent_config().grounding
        else:
            self.config = config

    def extract_numbers(self, text: str, user_message: str = "") -> List[ExtractedNumber]:
        """Extract all numeric quantities from text, filtering out dates, IDs, and user numbers."""
        if not text:
            return []

        # 1. Collect user numbers to exempt
        user_numbers = set()
        if user_message:
            for un in self._extract_raw_number_tokens(user_message):
                user_numbers.add(round(un[0], 4))

        # 2. Identify regions to exclude (IDs, ISO dates, timestamps, placeholders)
        excluded_spans = self._find_excluded_spans(text)

        # 3. Extract candidate numbers with context
        candidates: List[ExtractedNumber] = []
        # Pattern handles:
        # - Integers with thousand separators (e.g. 1.250 or 1.250.000)
        # - Decimals with comma or dot (e.g. 7,6 or 7.6 or 1.250,5)
        # - Percentages (e.g. 12% or 12,5%)
        # - Signed numbers (e.g. -540 or +1.2)
        pattern = re.compile(
            r"(?P<prefix>[-+])?"
            r"(?P<num>(?:\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b)|(?:\b\d+(?:[.,]\d+)?\b))"
            r"(?:\s*(?P<pct>%))?",
            re.UNICODE,
        )

        for match in pattern.finditer(text):
            start, end = match.span()

            # Check if match falls within an excluded span (ID, date, placeholder)
            if any(s <= start and end <= e for s, e in excluded_spans):
                continue

            raw_num_str = match.group("num")
            prefix = match.group("prefix") or ""
            is_pct = bool(match.group("pct"))

            # Parse numeric value according to Romanian/European or standard format
            val, prec = self._parse_romanian_number(raw_num_str)
            if prefix == "-":
                val = -val

            # Determine trailing unit if present
            unit = "%" if is_pct else None
            post_text = text[end : end + 20].strip().lower()
            if not unit and post_text:
                post_word = re.match(r"^([a-z/°²]+)", post_text)
                if post_word:
                    w = post_word.group(1).lower()
                    if w in UNITS:
                        unit = w

            ext = ExtractedNumber(
                raw_text=match.group(0),
                value=val,
                precision=prec,
                unit=unit,
                start=start,
                end=end,
            )

            # Check exemptions
            # Exemption A: User-provided number in the current turn
            if round(val, 4) in user_numbers:
                ext.is_exempt = True
                ext.exemption_reason = "user_number"

            # Exemption B: Small count exemption (<= threshold and no unit)
            elif (
                self.config.small_count_exemption > 0
                and val.is_integer()
                and 0 <= int(val) <= self.config.small_count_exemption
                and not unit
            ):
                ext.is_exempt = True
                ext.exemption_reason = "small_count"

            candidates.append(ext)

        return candidates

    def _parse_romanian_number(self, s: str) -> Tuple[float, int]:
        """Parse Romanian number string (e.g. '7,6', '1.250', '1.250,5', '7.6') into float and precision."""
        # Thousands dot with comma decimal: '1.250,50'
        if "." in s and "," in s:
            clean = s.replace(".", "").replace(",", ".")
            prec = len(s.split(",")[1])
            return float(clean), prec
        # Romanian thousands dot: '1.250' followed by three digits
        elif "." in s and re.match(r"^\d{1,3}(\.\d{3})+$", s):
            clean = s.replace(".", "")
            return float(clean), 0
        # Romanian comma decimal: '7,6'
        elif "," in s:
            clean = s.replace(",", ".")
            prec = len(s.split(",")[1])
            return float(clean), prec
        # Standard dot decimal: '7.6'
        elif "." in s:
            prec = len(s.split(".")[1])
            return float(s), prec
        # Plain integer: '1250'
        return float(s), 0

    def _extract_raw_number_tokens(self, text: str) -> List[Tuple[float, int]]:
        """Helper to extract numbers without checking excluded spans (for user messages)."""
        tokens: List[Tuple[float, int]] = []
        for m in re.finditer(r"\b\d+(?:[.,]\d+)?\b", text):
            try:
                v, p = self._parse_romanian_number(m.group(0))
                tokens.append((v, p))
            except Exception:
                pass
        return tokens

    def _find_excluded_spans(self, text: str) -> List[Tuple[int, int]]:
        """Find spans in text that should be exempt from number checking: IDs, dates, placeholders."""
        spans: List[Tuple[int, int]] = []

        # 1. Placeholders e.g. {{table:explain:...}}
        for m in re.finditer(r"\{\{.*?\}\}", text):
            spans.append(m.span())

        # 2. Asset / Task / Schedule / Result / Action IDs
        # Examples: FAN-T17, T17-pitch-001, CAL-INV02, HH-FAULTY-01, sched-draft-base, act-12345678, res-opt-001
        for m in re.finditer(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", text):
            spans.append(m.span())

        # 3. ISO dates & timestamps: 2026-11-12, 2026-11-08T00:00:00Z, 08:00, 16:30
        for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?\b", text):
            spans.append(m.span())
        for m in re.finditer(r"\b\d{1,2}:\d{2}\b", text):
            spans.append(m.span())

        # 4. Romanian date phrases: '12 noiembrie', '8 noiembrie 2026', 'joi 12 noiembrie'
        months_regex = "|".join(ROMANIAN_MONTHS)
        for m in re.finditer(rf"\b\d{{1,2}}\s+(?:{months_regex})(?:\s+\d{{4}})?\b", text, re.IGNORECASE):
            spans.append(m.span())

        return spans

    def collect_allowed_values(self, tool_results: List[Any]) -> Set[float]:
        """Collect all numeric leaves, list lengths, and unit conversions from tool results."""
        allowed: Set[float] = set()

        def _traverse(obj: Any):
            if obj is None:
                return
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if math.isfinite(obj):
                    val = float(obj)
                    allowed.add(round(val, 6))
                    # Conversions:
                    # kWh <-> MWh (1 MWh = 1000 kWh)
                    allowed.add(round(val * 1000.0, 6))
                    allowed.add(round(val / 1000.0, 6))
                    # Percent <-> fraction (0.12 <-> 12%)
                    if 0.0 <= val <= 1.0:
                        allowed.add(round(val * 100.0, 6))
                    if 0.0 <= val <= 100.0:
                        allowed.add(round(val / 100.0, 6))
            elif isinstance(obj, dict):
                for v in obj.values():
                    _traverse(v)
            elif isinstance(obj, (list, tuple, set)):
                allowed.add(float(len(obj)))
                for item in obj:
                    _traverse(item)
            elif hasattr(obj, "model_dump"):
                _traverse(obj.model_dump())
            elif hasattr(obj, "__dict__"):
                _traverse(obj.__dict__)

        for res in tool_results:
            _traverse(res)

        return allowed

    def check_text(
        self,
        text: str,
        tool_results: List[Any],
        user_message: str = "",
        result_ids: Optional[List[str]] = None,
    ) -> GroundingResult:
        """Verify that every non-exempt number in text is grounded in tool results."""
        candidates = self.extract_numbers(text, user_message=user_message)
        allowed_values = self.collect_allowed_values(tool_results)
        tolerance = self.config.tolerance

        unmatched: List[ExtractedNumber] = []
        for cand in candidates:
            if cand.is_exempt:
                continue

            v = cand.value
            matched = False

            # Check ifcand matches any allowed value within tolerance
            for a in allowed_values:
                # 1. Direct absolute difference within tolerance
                if abs(v - a) <= tolerance:
                    matched = True
                    break
                # 2. Displayed precision rounding check
                # e.g. allowed is 7.632, displayed is 7.6 (precision 1 -> step 0.05)
                prec_step = max(tolerance, 10 ** (-cand.precision) / 2.0) if cand.precision > 0 else 0.5
                if abs(v - a) <= prec_step:
                    matched = True
                    break

            if not matched:
                unmatched.append(cand)

        is_grounded = len(unmatched) == 0
        warning = None
        if not is_grounded:
            vals_str = ", ".join(f"'{u.raw_text}'" for u in unmatched)
            warning = f"Avertisment: valorile {vals_str} nu au putut fi verificate în rezultatele uneltelor."

        return GroundingResult(
            is_grounded=is_grounded,
            extracted_numbers=candidates,
            unmatched_numbers=unmatched,
            tool_result_ids=result_ids or [],
            warning_message=warning,
        )

    def format_correction_prompt(self, unmatched: List[ExtractedNumber]) -> str:
        """Format a single corrective revision prompt for the LLM when grounding fails."""
        vals = ", ".join(f"'{u.raw_text}'" for u in unmatched)
        return (
            f"Răspunsul conține valori numerice care nu apar în rezultatele uneltelor: {vals}. "
            "Revizuiește răspunsul folosind EXCLUSIV cifre prezente verbatim în rezultatele uneltelor din această tură. "
            "Nu aproxima și nu calcula valori derivate manual."
        )


default_grounding_checker = GroundingChecker()

