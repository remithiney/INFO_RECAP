import re
from html import unescape

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text):
    if not text:
        return ''
    text = unescape(text)
    text = _SCRIPT_STYLE_RE.sub(' ', text)
    text = _TAG_RE.sub(' ', text)
    text = _WS_RE.sub(' ', text)
    return text.strip()
