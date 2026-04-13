"""Text sanitization utilities — single source of truth for the RAG API.

Three sanitization layers serve different pipeline stages:

* **sanitize_output** — strips dangerous control chars from LLM responses
  before they reach the client, preserving whitespace (\\n, \\t, \\r).
* **sanitize_log** — strips ALL control chars and truncates for safe,
  single-line log output (prevents log injection / CWE-117).
* **sanitize_field** — null-safe wrapper that coerces non-string metadata
  (e.g. integer doc_ids) to str before applying a sanitizer function.

Consolidated from main.py, middleware.py, and response.py so every
sanitization decision lives in one reviewable, testable module.
"""

import re


# Strips C0 control chars EXCEPT whitespace (\t=0x09, \n=0x0a, \r=0x0d).
# Used for LLM output sanitization — preserves formatting.
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Strips ALL C0 control chars including whitespace.
# Used for log output — forces single-line, prevents log injection.
_LOG_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f]')


def sanitize_output(text: str) -> str:
    """Strip C0 control chars from LLM output (preserves \\n, \\r, \\t)."""
    return CONTROL_CHAR_RE.sub('', text)


def sanitize_log(text: str, max_len: int = 200) -> str:
    """Strip ALL control chars and truncate for safe log output."""
    return _LOG_CONTROL_RE.sub(' ', text)[:max_len]


def sanitize_field(text, sanitize_fn) -> str:
    """Apply output sanitizer to a single field, handling None and non-str types.

    JSON metadata can contain non-string values (e.g. integer doc_ids).
    Coerces to str before sanitizing to prevent TypeError in re.sub().
    Uses explicit ``is None`` instead of falsy check so valid values
    like ``0`` are preserved rather than silently dropped.
    """
    if text is None:
        return ""
    return sanitize_fn(str(text))
