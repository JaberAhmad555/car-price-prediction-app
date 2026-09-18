"""Conservative numeric parsing; source text must be retained by the caller."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import re
import unicodedata


NULL_TEXT = {"", "na", "n/a", "null", "none", "nan", "unknown", "not available", "-"}
DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?")
FINANCE = re.compile(r"\b(?:deposit|down\s*payment|monthly|install?ments?|emi|per\s+month|booking)\b|কিস্তি|অগ্রিম|ডাউন\s*পেমেন্ট")
CURRENCY = re.compile(r"^(?:bdt|tk\.?|taka|৳|টাকা)\s*|\s*(?:bdt|tk\.?|taka|৳|টাকা)$")


@dataclass(frozen=True)
class ParsedNumber:
    value: int | None
    issue: str | None = None
    interpretation: str | None = None


def normalize_text(value):
    """Normalize obvious typography only; do not infer category equivalence."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Boolean is not source text")
    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    return None if text.casefold() in NULL_TEXT else text


def _numeric(text, multiplier=1, *, allow_zero=False, interpretation=None):
    if not NUMBER.fullmatch(text):
        return ParsedNumber(None, "invalid_number_format")
    try:
        with localcontext() as context:
            context.prec = len(text) + len(str(multiplier)) + 4
            value = Decimal(text.replace(",", "")) * multiplier
    except InvalidOperation:
        return ParsedNumber(None, "invalid_number_format")
    if not value.is_finite():
        return ParsedNumber(None, "non_finite")
    if value < 0:
        return ParsedNumber(None, "negative_value")
    if value == 0 and not allow_zero:
        return ParsedNumber(None, "zero_not_allowed")
    if value != value.to_integral_value():
        return ParsedNumber(None, "fractional_normalized_value")
    return ParsedNumber(int(value), interpretation=interpretation)


def _text(value):
    try:
        text = normalize_text(value)
    except TypeError:
        return None, "invalid_type"
    return (None, "missing") if text is None else (text.translate(DIGITS).casefold(), None)


def parse_price(value, *, bare_unit=None):
    """A bare number has no assumed unit. Opt-in needs evidence at ingestion."""
    if bare_unit not in (None, "BDT", "lakh_BDT"):
        raise ValueError("Unsupported bare price unit")
    text, issue = _text(value)
    if issue:
        return ParsedNumber(None, issue)
    if FINANCE.search(text):
        return ParsedNumber(None, "non_full_price")
    if re.search(r"\b(?:usd|inr|eur|gbp)\b|[$€£₹]", text):
        return ParsedNumber(None, "unsupported_currency")
    explicit_bdt = bool(CURRENCY.search(text))
    text = CURRENCY.sub("", text).strip()
    if text.endswith("/-"):
        text = text[:-2].strip()
    multiplier = re.fullmatch(r"(.+?)\s*(lakh(?:s)?|lac(?:s)?|লাখ|crore(?:s)?|কোটি)", text)
    if multiplier:
        token = multiplier.group(2)
        scale = 10000000 if token.startswith("crore") or token == "কোটি" else 100000
        return _numeric(multiplier.group(1).strip(), scale, interpretation="explicit_" + token)
    if explicit_bdt:
        return _numeric(text, interpretation="explicit_BDT")
    if bare_unit:
        scale = 100000 if bare_unit == "lakh_BDT" else 1
        return _numeric(text, scale, interpretation="reviewed_bare_" + bare_unit)
    # Still distinguish invalid/negative inputs before reporting unit ambiguity.
    parsed = _numeric(text)
    if parsed.issue not in (None, "fractional_normalized_value"):
        return parsed
    return ParsedNumber(None, "ambiguous_price_unit")


def parse_mileage(value):
    """The source field kilometers_run supplies km for bare numeric values."""
    text, issue = _text(value)
    if issue:
        return ParsedNumber(None, issue)
    explicit = re.search(r"\s*(?:km|kms|kilometers?|kilometres?|কিমি|কি\.মি\.)$", text)
    if explicit:
        text = text[:explicit.start()].strip()
    return _numeric(text, allow_zero=True, interpretation="explicit_km" if explicit else "column_kilometers_run")


def parse_engine_capacity(value):
    """Only explicit cc/litre suffixes or a reviewed cc source-column convention."""
    text, issue = _text(value)
    if issue:
        return ParsedNumber(None, issue)
    suffix = re.search(r"\s*(cc|cm3|cm³|সিসি|l|litres?|liters?)$", text)
    if suffix:
        text = text[:suffix.start()].strip()
        token = suffix.group(1)
        scale = 1000 if token.startswith("l") else 1
        return _numeric(text, scale, interpretation="explicit_" + token)
    parsed = _numeric(text)
    if parsed.issue not in (None, "fractional_normalized_value"):
        return parsed
    return ParsedNumber(None, "ambiguous_engine_unit")


def parse_engine_with_unit(value, *, bare_unit=None):
    if bare_unit not in (None, "cc"):
        raise ValueError("Unsupported bare engine unit")
    parsed = parse_engine_capacity(value)
    if parsed.issue != "ambiguous_engine_unit" or bare_unit is None:
        return parsed
    text, issue = _text(value)
    return ParsedNumber(None, issue) if issue else _numeric(text, interpretation="reviewed_bare_cc")


def parse_model_year(value):
    text, issue = _text(value)
    if issue:
        return ParsedNumber(None, issue)
    parsed = _numeric(text, interpretation="source_model_year")
    if parsed.value is not None and not 1 <= parsed.value <= 9999:
        return ParsedNumber(None, "invalid_year")
    return parsed
