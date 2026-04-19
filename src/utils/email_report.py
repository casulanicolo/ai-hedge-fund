"""
src/utils/email_report.py
──────────────────────────
DEPRECATED — replaced by:
  src/alerts/premarket_builder.py  (pre-market briefing)
  src/alerts/postmarket_builder.py (EOD digest)
  src/alerts/email_sender.py       (send_premarket / send_postmarket / send_critical_alert)

This file is kept to avoid import errors from legacy callers.
All functions are no-ops or stubs. Remove callers and delete this file in Fase 5.
"""

import logging

logger = logging.getLogger(__name__)

_DEPRECATION_MSG = (
    "[email_report] DEPRECATED — use premarket_builder / postmarket_builder instead."
)


def build_html(*args, **kwargs) -> str:  # noqa: ANN002
    logger.warning(_DEPRECATION_MSG)
    return ""


def build_plaintext(*args, **kwargs) -> str:  # noqa: ANN002
    logger.warning(_DEPRECATION_MSG)
    return ""


def send_daily_report(*args, **kwargs) -> bool:  # noqa: ANN002
    logger.warning(_DEPRECATION_MSG)
    return False
