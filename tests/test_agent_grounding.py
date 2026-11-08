"""Tests for numeric grounding parser, exemptions, unit conversions, and validation."""

import pytest
from src.agent.config import GroundingConfig
from src.agent.grounding import GroundingChecker, ExtractedNumber


@pytest.fixture
def checker():
    config = GroundingConfig(tolerance=0.05, small_count_exemption=10)
    return GroundingChecker(config=config)


def test_grounding_romanian_decimal_comma(checker):
    """Verify that '7,6' parses to 7.6 and grounds against 7.6 in tool results."""
    text = "Pierderea estimată este de 7,6 MWh."
    tool_results = [{"loss_mwh": 7.6}]
    res = checker.check_text(text, tool_results)

    assert res.is_grounded is True
    assert len(res.extracted_numbers) == 1
    assert res.extracted_numbers[0].value == 7.6
    assert res.extracted_numbers[0].unit == "mwh"
    assert len(res.unmatched_numbers) == 0


def test_grounding_romanian_thousands_dot(checker):
    """Verify that '1.250' parses to 1250 and grounds against 1250 in tool results."""
    text = "Costul total al intervenției este de 1.250 RON."
    tool_results = [{"total_ron": 1250.0}]
    res = checker.check_text(text, tool_results)

    assert res.is_grounded is True
    assert len(res.extracted_numbers) == 1
    assert res.extracted_numbers[0].value == 1250.0
    assert res.extracted_numbers[0].unit == "ron"
    assert len(res.unmatched_numbers) == 0


def test_grounding_percentage_and_fraction(checker):
    """Verify that '12%' parses to 12.0 and grounds against 0.12 or 12 in tool results."""
    text = "Eficiența a scăzut cu 12% față de săptămâna trecută."
    # Case A: tool result has 0.12 (fraction)
    tool_results_frac = [{"rate": 0.12}]
    res_frac = checker.check_text(text, tool_results_frac)
    assert res_frac.is_grounded is True

    # Case B: tool result has 12.0 (percentage)
    tool_results_pct = [{"rate": 12.0}]
    res_pct = checker.check_text(text, tool_results_pct)
    assert res_pct.is_grounded is True


def test_grounding_identifier_exemption(checker):
    """Verify that asset IDs like 'FAN-T17' or 'T17-pitch-001' do not extract 17 or 001 as numbers."""
    text = "Turbina FAN-T17 are sarcina T17-pitch-001 programată."
    extracted = checker.extract_numbers(text)

    # 17 should not be extracted as an ungrounded numeric requirement
    non_exempt = [n for n in extracted if not n.is_exempt]
    assert len(non_exempt) == 0


def test_grounding_date_exemption(checker):
    """Verify that dates like '2026-11-12' and times like '08:00' are exempt from grounding."""
    text = "Intervenția este planificată pe data de 2026-11-12 la ora 08:00."
    extracted = checker.extract_numbers(text)

    non_exempt = [n for n in extracted if not n.is_exempt]
    assert len(non_exempt) == 0


def test_grounding_deliberately_ungrounded_value(checker):
    """Verify that a hallucinated/ungrounded number fails the grounding check and generates warning."""
    text = "Pierderea estimată este de 99,4 MWh."
    tool_results = [{"loss_mwh": 7.6, "other": 16.1}]
    res = checker.check_text(text, tool_results)

    assert res.is_grounded is False
    assert len(res.unmatched_numbers) == 1
    assert res.unmatched_numbers[0].value == 99.4
    assert res.warning_message is not None
    assert "99,4" in res.warning_message

    # Verify corrective revision prompt contains the ungrounded value
    prompt = checker.format_correction_prompt(res.unmatched_numbers)
    assert "99,4" in prompt
    assert "EXCLUSIV" in prompt


def test_grounding_user_number_exemption(checker):
    """Verify that numbers present in the user query are exempt from tool grounding."""
    user_msg = "De ce amânăm cu 5 zile?"
    agent_ans = "Amânarea de 5 zile este necesară pentru piesa de schimb."
    tool_results = [{"task_id": "T05-bearing-001"}]  # Does not contain 5

    res = checker.check_text(agent_ans, tool_results, user_message=user_msg)
    assert res.is_grounded is True


def test_grounding_small_count_exemption(checker):
    """Verify that small counts <= 10 without units are exempt, but numbers with units are checked."""
    # Count without unit: exempt
    text_count = "Am analizat 3 opțiuni diferite."
    res_count = checker.check_text(text_count, [])
    assert res_count.is_grounded is True

    # Number with unit: NOT exempt even if <= 10
    text_unit = "Pierderea este de 3 MWh."
    res_unit = checker.check_text(text_unit, [])
    assert res_unit.is_grounded is False
    assert len(res_unit.unmatched_numbers) == 1


def test_grounding_kwh_mwh_conversion(checker):
    """Verify unit conversion between kWh and MWh (e.g. 1.4 MWh = 1400 kWh)."""
    text = "Pierderea lunară este de 1.400 kWh."
    tool_results = [{"loss_mwh": 1.4}]
    res = checker.check_text(text, tool_results)
    assert res.is_grounded is True

