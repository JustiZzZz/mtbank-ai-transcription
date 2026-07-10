"""Unit-тесты юридических ориентиров в LLM prompts."""

from app.agents.prompts import BASE_SYSTEM_PROMPT, COMPLIANCE_PROMPT, QUALITY_PROMPT


def test_base_prompt_contains_belarus_legal_anchors() -> None:
    for anchor in (
        "Банковский кодекс Республики Беларусь",
        "О потребительском кредите",
        "О защите персональных данных",
        "О платежных системах",
        "Об обращениях граждан",
        "комплаенс-политика",
    ):
        assert anchor in BASE_SYSTEM_PROMPT


def test_compliance_prompt_covers_high_risk_bank_violations() -> None:
    for risk in (
        "регистрации обращения",
        "дополнительных платных услуг",
        "персональных данных",
        "социальной инженерии",
        "запроса секретных кодов",
    ):
        assert risk in COMPLIANCE_PROMPT


def test_quality_prompt_requires_safe_customer_handling() -> None:
    for signal in (
        "не предлагает зарегистрировать жалобу",
        "защите PIN/CVV/SMS-кодов",
        "давит на клиента",
    ):
        assert signal in QUALITY_PROMPT


def test_summary_prompt_requires_secret_redaction() -> None:
    from app.agents.prompts import SUMMARIZER_PROMPT

    assert "Не включай в summary значения PIN/CVV/SMS-кодов" in SUMMARIZER_PROMPT
    assert "секретные коды/данные" in SUMMARIZER_PROMPT
