"""Feature extraction for phishing URL classification.

The extractor is intentionally URL-focused. It parses and measures a URL but
does not browse the target site, execute scripts, or follow redirects. Runtime
metadata such as domain age, SSL validity, and redirect count can be supplied by
safe upstream collection jobs or by a curated training dataset.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import parse_qsl, unquote, urlparse

import pandas as pd


LOGGER = logging.getLogger(__name__)

SUSPICIOUS_WORDS = {
    "account",
    "auth",
    "bank",
    "billing",
    "card",
    "confirm",
    "credential",
    "customer",
    "ebanking",
    "identity",
    "kyc",
    "login",
    "otp",
    "password",
    "pay",
    "payment",
    "recover",
    "secure",
    "signin",
    "support",
    "suspend",
    "unlock",
    "update",
    "verify",
    "wallet",
}

SUSPICIOUS_QUERY_KEYS = {
    "continue",
    "dest",
    "destination",
    "next",
    "redirect",
    "redirect_uri",
    "return",
    "return_to",
    "session",
    "target",
    "token",
    "url",
}

SHORTENER_HINTS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "s.id",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
    "trib.al",
}

SPECIAL_CHARACTERS = "@%-_?&=#+~;:,!$*'()[]{}|\\"
ENCODED_SEQUENCE_RE = re.compile(r"%[0-9a-fA-F]{2}")
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

TRUSTED_BRAND_DOMAINS = {
    "adobe": {"adobe.com"},
    "amazon": {"amazon.com", "amazon.in"},
    "apple": {"apple.com"},
    "axis": {"axisbank.com"},
    "bits": {"bits-pilani.ac.in"},
    "facebook": {"facebook.com", "fb.com"},
    "google": {"google.com", "google.co.in"},
    "hdfc": {"hdfcbank.com"},
    "hsbc": {"hsbc.com", "hsbc.co.in"},
    "icici": {"icicibank.com"},
    "instagram": {"instagram.com"},
    "kotak": {"kotak.com", "kotakbank.com"},
    "linkedin": {"linkedin.com"},
    "microsoft": {
        "microsoft.com",
        "cloud.microsoft",
        "copilot.microsoft.com",
        "live.com",
        "microsoft365.com",
        "microsoftonline.com",
        "office.com",
        "office365.com",
        "outlook.com",
        "sharepoint.com",
        "teams.microsoft.com",
        "windows.net",
    },
    "netflix": {"netflix.com"},
    "paytm": {"paytm.com"},
    "paypal": {"paypal.com"},
    "phonepe": {"phonepe.com"},
    "sbi": {"onlinesbi.sbi", "sbi.co.in"},
    "whatsapp": {"whatsapp.com"},
    "wipro": {"wipro.com"},
    "yahoo": {"yahoo.com"},
}

BRAND_TOKENS = set(TRUSTED_BRAND_DOMAINS)

CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
    }
)

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "digit_count",
    "digit_ratio",
    "letter_ratio",
    "special_char_count",
    "at_symbol_count",
    "hyphen_count",
    "dot_count",
    "encoded_char_count",
    "suspicious_word_count",
    "url_entropy",
    "subdomain_count",
    "estimated_domain_length",
    "has_punycode",
    "domain_age_days",
    "whois_available",
    "registrar_risk",
    "reputation_score",
    "uses_https",
    "ssl_valid",
    "has_ip_address",
    "has_port",
    "redirect_count",
    "suspicious_query_key_count",
    "has_embedded_url",
    "shortener_hint",
    "token_count",
    "average_token_length",
    "max_token_length",
    "brand_token_count",
    "max_brand_similarity",
    "typosquatting_score",
    "domain_impersonation_score",
    "brand_spoofing_score",
    "homograph_score",
    "lookalike_digit_count",
    "url_mutation_score",
    "char_pattern_risk",
    "domain_reputation_risk",
    "trusted_domain_match",
    "trusted_domain_normality_score",
]


@dataclass(frozen=True)
class ParsedUrl:
    """Parsed URL fields used during feature extraction."""

    original: str
    normalized: str
    scheme: str
    hostname: str
    path: str
    query: str
    port_present: bool


def normalize_url(url: Any) -> str:
    """Return a stable URL representation for parsing and duplicate checks."""

    text = "" if url is None else str(url).strip()
    text = text.replace("\n", "").replace("\r", "")
    if not text:
        return ""

    parsed = urlparse(text)
    if not parsed.scheme:
        text = "http://" + text
        parsed = urlparse(text)

    scheme = (parsed.scheme or "http").lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""
    normalized = f"{scheme}://{hostname}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    return normalized


def parse_url(url: Any) -> ParsedUrl:
    """Parse a URL while tolerating missing schemes and malformed input."""

    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    netloc = parsed.netloc or ""
    port_present = False
    if ":" in netloc.rsplit("@", 1)[-1]:
        port_present = True

    return ParsedUrl(
        original="" if url is None else str(url),
        normalized=normalized,
        scheme=(parsed.scheme or "").lower(),
        hostname=(parsed.hostname or "").lower().rstrip("."),
        path=parsed.path or "",
        query=parsed.query or "",
        port_present=port_present,
    )


def shannon_entropy(value: str) -> float:
    """Measure character randomness in a string."""

    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return round(entropy, 6)


def skeletonize_text(value: str) -> str:
    """Normalize common leetspeak characters used in lookalike domains."""

    return value.lower().translate(CONFUSABLE_TRANSLATION)


def tokenize_text(value: str) -> list[str]:
    """Split URL text into meaningful lowercase alphanumeric tokens."""

    tokens = [token for token in TOKEN_SPLIT_RE.split(value.lower()) if token]
    split_tokens: list[str] = []
    for token in tokens:
        split_tokens.append(token)
        split_tokens.extend(re.findall(r"[a-z]+|\d+", token))
    return list(dict.fromkeys(split_tokens or tokens))


def hostname_tokens(hostname: str, path: str = "", query: str = "") -> list[str]:
    """Tokenize hostname, path, and query text for URL intent analysis."""

    text = " ".join([hostname, unquote(path), unquote(query)])
    return tokenize_text(text)


def registered_domain_core(hostname: str) -> str:
    """Return the estimated registered-domain label before the suffix."""

    registered_domain = estimate_registered_domain(hostname)
    if not registered_domain or is_ip_hostname(registered_domain):
        return registered_domain
    labels = registered_domain.split(".")
    return labels[0] if labels else registered_domain


def levenshtein_distance(left: str, right: str, max_distance: int = 3) -> int:
    """Compute bounded edit distance for short URL tokens."""

    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def similarity_ratio(left: str, right: str) -> float:
    """Return a stable character-level similarity ratio."""

    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def is_known_brand_domain(hostname: str, brand: str) -> bool:
    """Check whether the hostname belongs to a known legitimate brand domain."""

    allowed_domains = TRUSTED_BRAND_DOMAINS.get(brand, set())
    for domain in allowed_domains:
        if hostname == domain or hostname.endswith("." + domain):
            return True
    return False


def trusted_domain_match(hostname: str) -> bool:
    """Return true when the hostname belongs to the internal trusted-domain set."""

    return any(is_known_brand_domain(hostname, brand) for brand in TRUSTED_BRAND_DOMAINS)


def trusted_domain_normality_score(
    hostname: str,
    uses_https: bool,
    ssl_valid: float,
    redirect_count: float,
    suspicious_query_key_count: float,
    has_embedded_url: int,
    encoded_char_count: float,
    subdomain_count: int,
    domain_reputation_risk_value: float,
) -> float:
    """Score whether a known trusted domain also has normal URL structure."""

    if not trusted_domain_match(hostname):
        return 0.0

    score = 0.35
    if uses_https:
        score += 0.2
    if ssl_valid in {-1.0, 1.0}:
        score += 0.1
    if redirect_count <= 1:
        score += 0.1
    if suspicious_query_key_count == 0:
        score += 0.08
    if has_embedded_url == 0:
        score += 0.08
    if encoded_char_count <= 1:
        score += 0.04
    if subdomain_count <= 2:
        score += 0.03
    if domain_reputation_risk_value <= 0.1:
        score += 0.02
    return round(min(1.0, score), 6)


def brand_similarity_analysis(hostname: str, tokens: list[str]) -> Dict[str, float]:
    """Score brand spoofing, typosquatting, and domain impersonation risk."""

    core = registered_domain_core(hostname)
    skeleton_core = skeletonize_text(core)
    registered_domain = estimate_registered_domain(hostname)
    token_set = set(tokens)

    max_similarity = 0.0
    typosquatting_score = 0.0
    brand_token_count = 0
    brand_spoofing_score = 0.0
    domain_impersonation_score = 0.0

    for brand in BRAND_TOKENS:
        brand_in_text = brand in token_set or brand in skeleton_core or brand in skeletonize_text(hostname)
        if brand_in_text:
            brand_token_count += 1
            if not is_known_brand_domain(hostname, brand):
                brand_spoofing_score = max(brand_spoofing_score, 1.0)
                if registered_domain and brand not in registered_domain.split(".")[0]:
                    domain_impersonation_score = max(domain_impersonation_score, 0.7)

        candidates = [core, skeleton_core, *tokens, *[skeletonize_text(token) for token in tokens]]
        for token in tokens:
            token_skeleton = skeletonize_text(token)
            if token_skeleton == brand and token != brand and not is_known_brand_domain(hostname, brand):
                typosquatting_score = max(typosquatting_score, 1.0)
        for candidate in candidates:
            if len(candidate) < 3:
                continue
            ratio = similarity_ratio(candidate, brand)
            distance = levenshtein_distance(candidate, brand)
            if candidate == brand and not is_known_brand_domain(hostname, brand):
                ratio = 1.0
            max_similarity = max(max_similarity, ratio)
            if candidate != brand and (ratio >= 0.78 or distance <= 2):
                if not is_known_brand_domain(hostname, brand):
                    distance_score = max(0.0, 1.0 - (distance / max(len(brand), 1)))
                    typosquatting_score = max(typosquatting_score, ratio, distance_score)

    if brand_spoofing_score and any(word in token_set for word in SUSPICIOUS_WORDS):
        domain_impersonation_score = max(domain_impersonation_score, 1.0)
    if brand_spoofing_score and count_subdomains(hostname) >= 2:
        domain_impersonation_score = max(domain_impersonation_score, 0.8)

    return {
        "brand_token_count": float(brand_token_count),
        "max_brand_similarity": round(max_similarity, 6),
        "typosquatting_score": round(min(1.0, typosquatting_score), 6),
        "domain_impersonation_score": round(min(1.0, domain_impersonation_score), 6),
        "brand_spoofing_score": round(min(1.0, brand_spoofing_score), 6),
    }


def homograph_score(hostname: str) -> float:
    """Score IDN, punycode, and mixed-script indicators."""

    if not hostname:
        return 0.0

    score = 0.0
    if "xn--" in hostname:
        score += 0.75

    non_ascii_count = sum(ord(ch) > 127 for ch in hostname)
    if non_ascii_count:
        score += 0.65
        score += min(0.2, non_ascii_count / max(len(hostname), 1))

    script_markers = set()
    for char in hostname:
        if ord(char) <= 127 or not char.isalpha():
            continue
        try:
            name = char.encode("unicode_escape").decode("ascii")
        except UnicodeEncodeError:
            name = ""
        if "\\u04" in name or "\\u05" in name:
            script_markers.add("cyrillic_or_hebrew")
        elif "\\u03" in name:
            script_markers.add("greek")
        else:
            script_markers.add("other_non_latin")
    if len(script_markers) > 1:
        score += 0.3

    return round(min(1.0, score), 6)


def mutation_score(tokens: list[str], url_entropy: float, digit_ratio: float, encoded_count: float) -> float:
    """Estimate generated or mutated URL behavior."""

    score = 0.0
    if url_entropy >= 4.5:
        score += 0.25
    if digit_ratio >= 0.18:
        score += 0.2
    if encoded_count >= 2:
        score += 0.2

    long_random_tokens = 0
    for token in tokens:
        if len(token) >= 12 and shannon_entropy(token) >= 3.2:
            long_random_tokens += 1
    if long_random_tokens:
        score += min(0.35, long_random_tokens * 0.15)

    return round(min(1.0, score), 6)


def character_pattern_risk(hostname: str, tokens: list[str]) -> float:
    """Score suspicious character-level patterns in host and tokens."""

    score = 0.0
    if re.search(r"(.)\1{3,}", hostname):
        score += 0.25
    if re.search(r"[a-z]+\d+[a-z]+", hostname) or re.search(r"\d+[a-z]+\d+", hostname):
        score += 0.25
    if hostname.count("-") >= 2:
        score += 0.2
    if any(len(token) >= 16 and not token.isalpha() for token in tokens):
        score += 0.25
    return round(min(1.0, score), 6)


def domain_reputation_risk(metadata: Mapping[str, Any], domain_age_days: float, reputation_score: float) -> float:
    """Convert reputation and freshness metadata into a risk-oriented score."""

    risk = 0.0
    if reputation_score <= 20:
        risk += 0.55
    elif reputation_score <= 40:
        risk += 0.3

    if 0 <= domain_age_days < 7:
        risk += 0.35
    elif 7 <= domain_age_days < 30:
        risk += 0.25
    elif 30 <= domain_age_days < 90:
        risk += 0.1

    if bool_metadata(metadata, "whois_available", default=0.0) == 0 and domain_age_days == -1:
        risk += 0.1
    return round(min(1.0, risk), 6)


def is_ip_hostname(hostname: str) -> bool:
    """Return true when the hostname is an IPv4 or IPv6 address."""

    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def estimate_registered_domain(hostname: str) -> str:
    """Estimate the registered domain without relying on external suffix data."""

    if not hostname or is_ip_hostname(hostname):
        return hostname
    labels = [label for label in hostname.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    return ".".join(labels[-2:])


def count_subdomains(hostname: str) -> int:
    """Count labels before the estimated registered domain."""

    if not hostname or is_ip_hostname(hostname):
        return 0
    labels = [label for label in hostname.split(".") if label]
    return max(0, len(labels) - 2)


def registrar_risk_score(registrar: Any) -> float:
    """Convert registrar text into a conservative numeric risk hint."""

    if registrar is None or (isinstance(registrar, float) and math.isnan(registrar)):
        return 0.0
    text = str(registrar).strip().lower()
    if not text or text == "unknown":
        return 0.0

    higher_risk_tokens = {
        "privacy",
        "proxy",
        "cheap",
        "free",
        "anonymous",
        "redacted",
    }
    if any(token in text for token in higher_risk_tokens):
        return 0.6
    return 0.2


def numeric_metadata(
    metadata: Mapping[str, Any],
    key: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read numeric metadata with consistent fallback behavior."""

    value = metadata.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if math.isnan(number):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def bool_metadata(metadata: Mapping[str, Any], key: str, default: float = -1.0) -> float:
    """Read a boolean-like metadata field as 1.0, 0.0, or an unknown default."""

    if key not in metadata:
        return default
    value = metadata.get(key, default)
    if value is None:
        return default
    if value == default:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "unknown":
            return default
        if text in {"1", "true", "valid", "yes", "y"}:
            return 1.0
        if text in {"0", "false", "invalid", "no", "n"}:
            return 0.0
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return 1.0 if number > 0 else 0.0


def extract_url_features(
    url: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Extract the numeric feature vector for one URL."""

    metadata = metadata or {}
    parsed = parse_url(url)
    normalized = parsed.normalized
    hostname = parsed.hostname
    decoded_url = unquote(normalized.lower())
    registered_domain = estimate_registered_domain(hostname)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_keys = {key.lower() for key, _ in query_pairs}
    tokens = hostname_tokens(hostname, parsed.path, parsed.query)
    brand_scores = brand_similarity_analysis(hostname, tokens)

    url_length = len(normalized)
    digit_count = sum(ch.isdigit() for ch in normalized)
    letter_count = sum(ch.isalpha() for ch in normalized)
    special_char_count = sum(ch in SPECIAL_CHARACTERS for ch in normalized)
    suspicious_word_count = sum(1 for word in SUSPICIOUS_WORDS if word in decoded_url)
    suspicious_query_key_count = sum(1 for key in query_keys if key in SUSPICIOUS_QUERY_KEYS)
    has_embedded_url = int(
        "http://" in parsed.query.lower()
        or "https://" in parsed.query.lower()
        or "http%3a" in parsed.query.lower()
        or "https%3a" in parsed.query.lower()
    )
    average_token_length = sum(len(token) for token in tokens) / len(tokens) if tokens else 0.0
    max_token_length = max((len(token) for token in tokens), default=0)
    domain_age_days = numeric_metadata(metadata, "domain_age_days", -1.0, minimum=-1.0)
    reputation_score = numeric_metadata(
        metadata,
        "reputation_score",
        50.0,
        minimum=0.0,
        maximum=100.0,
    )
    encoded_char_count = float(len(ENCODED_SEQUENCE_RE.findall(normalized)))
    digit_ratio = round(digit_count / url_length, 6) if url_length else 0.0
    entropy = shannon_entropy(normalized)

    ssl_valid = bool_metadata(metadata, "ssl_valid", default=-1.0)
    redirect_count = numeric_metadata(metadata, "redirect_count", 0.0, minimum=0.0)
    subdomain_count = count_subdomains(hostname)
    reputation_risk = domain_reputation_risk(metadata, domain_age_days, reputation_score)
    uses_https = parsed.scheme == "https"

    features: Dict[str, float] = {
        "url_length": float(url_length),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "digit_count": float(digit_count),
        "digit_ratio": digit_ratio,
        "letter_ratio": round(letter_count / url_length, 6) if url_length else 0.0,
        "special_char_count": float(special_char_count),
        "at_symbol_count": float(normalized.count("@")),
        "hyphen_count": float(normalized.count("-")),
        "dot_count": float(normalized.count(".")),
        "encoded_char_count": encoded_char_count,
        "suspicious_word_count": float(suspicious_word_count),
        "url_entropy": entropy,
        "subdomain_count": float(subdomain_count),
        "estimated_domain_length": float(len(registered_domain)),
        "has_punycode": float("xn--" in hostname),
        "domain_age_days": domain_age_days,
        "whois_available": bool_metadata(metadata, "whois_available", default=0.0),
        "registrar_risk": registrar_risk_score(metadata.get("registrar")),
        "reputation_score": reputation_score,
        "uses_https": float(uses_https),
        "ssl_valid": ssl_valid,
        "has_ip_address": float(is_ip_hostname(hostname)),
        "has_port": float(parsed.port_present),
        "redirect_count": redirect_count,
        "suspicious_query_key_count": float(suspicious_query_key_count),
        "has_embedded_url": float(has_embedded_url),
        "shortener_hint": float(hostname in SHORTENER_HINTS),
        "token_count": float(len(tokens)),
        "average_token_length": round(average_token_length, 6),
        "max_token_length": float(max_token_length),
        "brand_token_count": brand_scores["brand_token_count"],
        "max_brand_similarity": brand_scores["max_brand_similarity"],
        "typosquatting_score": brand_scores["typosquatting_score"],
        "domain_impersonation_score": brand_scores["domain_impersonation_score"],
        "brand_spoofing_score": brand_scores["brand_spoofing_score"],
        "homograph_score": homograph_score(hostname),
        "lookalike_digit_count": float(sum(ch in "0134578" for ch in hostname)),
        "url_mutation_score": mutation_score(tokens, entropy, digit_ratio, encoded_char_count),
        "char_pattern_risk": character_pattern_risk(hostname, tokens),
        "domain_reputation_risk": reputation_risk,
        "trusted_domain_match": float(trusted_domain_match(hostname)),
        "trusted_domain_normality_score": trusted_domain_normality_score(
            hostname=hostname,
            uses_https=uses_https,
            ssl_valid=ssl_valid,
            redirect_count=redirect_count,
            suspicious_query_key_count=suspicious_query_key_count,
            has_embedded_url=has_embedded_url,
            encoded_char_count=encoded_char_count,
            subdomain_count=subdomain_count,
            domain_reputation_risk_value=reputation_risk,
        ),
    }
    return features


def extract_feature_frame(
    urls: Iterable[Any],
    metadata_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract features for a sequence of URLs."""

    rows = []
    if metadata_frame is None:
        for url in urls:
            rows.append(extract_url_features(url))
    else:
        for index, url in enumerate(urls):
            metadata = metadata_frame.iloc[index].to_dict()
            rows.append(extract_url_features(url, metadata))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def explain_feature_indicators(features: Mapping[str, float]) -> list[str]:
    """Create readable risk indicators from one feature vector."""

    reasons: list[str] = []
    if features.get("has_ip_address", 0) >= 1:
        reasons.append("URL uses an IP address as the hostname")
    if features.get("uses_https", 0) == 0:
        reasons.append("URL does not use HTTPS")
    if features.get("ssl_valid", -1) == 0:
        reasons.append("SSL certificate metadata is marked invalid")
    if features.get("suspicious_word_count", 0) >= 2:
        reasons.append("URL contains multiple account or credential related terms")
    elif features.get("suspicious_word_count", 0) == 1:
        reasons.append("URL contains an account or credential related term")
    if features.get("encoded_char_count", 0) >= 2:
        reasons.append("URL contains encoded characters often used for obfuscation")
    if features.get("url_entropy", 0) >= 4.5:
        reasons.append("URL has high character entropy")
    if features.get("subdomain_count", 0) >= 3:
        reasons.append("URL has a deep subdomain structure")
    if features.get("has_punycode", 0) >= 1:
        reasons.append("Hostname contains punycode, which may indicate homograph risk")
    if features.get("domain_age_days", 999999) != -1 and features.get("domain_age_days", 999999) < 30:
        reasons.append("Domain age metadata indicates a newly registered domain")
    if features.get("redirect_count", 0) >= 2:
        reasons.append("URL metadata shows multiple redirects")
    if features.get("suspicious_query_key_count", 0) >= 1:
        reasons.append("URL query contains redirect or session related parameters")
    if features.get("has_embedded_url", 0) >= 1:
        reasons.append("URL query embeds another URL")
    if features.get("shortener_hint", 0) >= 1:
        reasons.append("Hostname resembles a URL shortener")
    if features.get("url_length", 0) >= 120:
        reasons.append("URL is unusually long")
    if features.get("brand_spoofing_score", 0) >= 0.8:
        reasons.append("Domain contains a trusted brand outside the known brand domain")
    if features.get("typosquatting_score", 0) >= 0.78:
        reasons.append("Domain is highly similar to a trusted brand and may be typosquatting")
    if features.get("domain_impersonation_score", 0) >= 0.7:
        reasons.append("Domain structure suggests brand impersonation")
    if features.get("homograph_score", 0) >= 0.7:
        reasons.append("Hostname has homograph or punycode indicators")
    if features.get("url_mutation_score", 0) >= 0.45:
        reasons.append("URL contains mutation patterns associated with generated phishing links")
    if features.get("char_pattern_risk", 0) >= 0.45:
        reasons.append("Hostname contains suspicious character-level patterns")
    if features.get("domain_reputation_risk", 0) >= 0.45:
        reasons.append("Domain reputation or freshness metadata indicates elevated risk")
    if features.get("trusted_domain_normality_score", 0) >= 0.8:
        reasons.append("Trusted domain structure and reputation indicators look normal")

    if not reasons:
        reasons.append("No strong phishing indicators were found in the extracted URL features")
    return reasons
