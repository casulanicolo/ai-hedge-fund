"""
src/audit/event_types.py — Fase 8
Canonical constants for audit_trail.event_type and severity.
"""
from __future__ import annotations


class EventType:
    ORDER_SUBMIT          = "ORDER_SUBMIT"
    ORDER_FILL            = "ORDER_FILL"
    ORDER_REJECT          = "ORDER_REJECT"
    POSITION_OPEN         = "POSITION_OPEN"
    POSITION_CLOSE        = "POSITION_CLOSE"
    CIRCUIT_BREAKER       = "CIRCUIT_BREAKER_TRIGGERED"
    CIRCUIT_BREAKER_RESET = "CIRCUIT_BREAKER_RESET"
    KILL_SWITCH           = "KILL_SWITCH"
    COMPLIANCE_FAIL       = "COMPLIANCE_FAIL"
    WEIGHT_UPDATE         = "WEIGHT_UPDATE"
    MODEL_PROMOTED        = "MODEL_PROMOTED"
    ERROR                 = "ERROR"
    MONITOR_TICK          = "MONITOR_TICK"
    CLOSE_ALL             = "CLOSE_ALL"


class Severity:
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
