from conftest import at

from tg_connector.models import RawMessage
from tg_connector.noise import extract_links, is_noise


def msg(text: str, kind: str = "text", reply_to: int | None = None) -> RawMessage:
    return RawMessage(1, 1, at(20, 10), 1, "Аня", kind, text, reply_to)


def test_obvious_noise_is_dropped():
    assert is_noise(msg("", kind="sticker"))
    assert is_noise(msg("", kind="service"))
    assert is_noise(msg("", kind="photo"))           # photo without caption
    assert is_noise(msg("👍"))
    assert is_noise(msg("😂😂😂"))
    assert is_noise(msg("Дякую!"))
    assert is_noise(msg("спс"))
    assert is_noise(msg("ахаха"))


def test_short_agreement_is_kept():
    # "ok" / "+" / "так" can be agreement to a decision — the LLM decides, not the filter.
    assert not is_noise(msg("ок"))
    assert not is_noise(msg("+"))
    assert not is_noise(msg("так"))


def test_content_is_kept():
    assert not is_noise(msg("Бюджет погоджено, 500$"))
    assert not is_noise(msg("Ось макет", kind="photo"))
    assert not is_noise(msg("[file: Q3_budget.xlsx]", kind="document"))


def test_links_are_extracted_deterministically():
    texts = ["Дивись https://example.com/report.pdf, і ще https://example.com/report.pdf", "та http://a.b/c)"]
    assert extract_links(texts) == ["https://example.com/report.pdf", "http://a.b/c"]
