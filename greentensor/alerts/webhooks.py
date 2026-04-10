# -*- coding: utf-8 -*-
"""
Webhook and alerting integrations for GreenTensor.

Sends security alerts and carbon budget violations to:
- Slack
- PagerDuty
- Generic webhook (any HTTP endpoint)
- Email (SMTP)
- Console (default)
"""

import json
import time
import urllib.request
import urllib.error
from typing import Any, Callable, Optional
from greentensor.utils.logger import logger


def _post_json(url: str, payload: dict, headers: dict = None) -> bool:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status < 300
    except Exception as e:
        logger.warning(f"Webhook delivery failed: {e}")
        return False


def _format_event(event: Any) -> dict:
    """Convert any GreenTensor alert/event to a standard dict."""
    base = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source": "greentensor",
    }
    if hasattr(event, "severity"):
        base["severity"] = event.severity
    if hasattr(event, "message"):
        base["message"] = event.message
    if hasattr(event, "alert_type"):
        base["type"] = event.alert_type
    elif hasattr(event, "category"):
        base["type"] = event.category
    if hasattr(event, "mitre_technique"):
        base["mitre"] = event.mitre_technique
    return base


# ── Slack ──────────────────────────────────────────────────────────────────────

def SlackWebhook(webhook_url: str, channel: str = None) -> Callable:
    """
    Returns an on_alert handler that posts to a Slack webhook.

    Usage:
        with GreenTensor(on_alert=SlackWebhook("https://hooks.slack.com/...")) as gt:
            train()
    """
    def handler(event: Any):
        evt = _format_event(event)
        severity = evt.get("severity", "info").upper()
        emoji = {"CRITICAL": ":red_circle:", "HIGH": ":orange_circle:",
                 "MEDIUM": ":yellow_circle:", "LOW": ":white_circle:"}.get(severity, ":white_circle:")
        payload = {
            "text": f"{emoji} *GreenTensor Alert* [{severity}]",
            "attachments": [{
                "color": {"CRITICAL": "danger", "HIGH": "warning"}.get(severity, "good"),
                "fields": [
                    {"title": "Type", "value": evt.get("type", "unknown"), "short": True},
                    {"title": "Severity", "value": severity, "short": True},
                    {"title": "Message", "value": evt.get("message", ""), "short": False},
                    {"title": "Time", "value": evt.get("timestamp", ""), "short": True},
                    {"title": "MITRE", "value": evt.get("mitre", "N/A"), "short": True},
                ],
            }],
        }
        if channel:
            payload["channel"] = channel
        _post_json(webhook_url, payload)
    return handler


# ── PagerDuty ──────────────────────────────────────────────────────────────────

def PagerDutyAlert(integration_key: str, severity_map: dict = None) -> Callable:
    """
    Returns an on_alert handler that creates PagerDuty incidents.

    Usage:
        with GreenTensor(on_alert=PagerDutyAlert("your-integration-key")) as gt:
            train()
    """
    PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"
    default_map = {"critical": "critical", "high": "error",
                   "medium": "warning", "low": "info"}
    sev_map = severity_map or default_map

    def handler(event: Any):
        evt = _format_event(event)
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"GreenTensor: {evt.get('type', 'alert')} — {evt.get('message', '')[:100]}",
                "severity": sev_map.get(evt.get("severity", "low"), "info"),
                "source": "greentensor",
                "timestamp": evt.get("timestamp"),
                "custom_details": evt,
            },
        }
        _post_json(PAGERDUTY_URL, payload)
    return handler


# ── Generic webhook ────────────────────────────────────────────────────────────

def GenericWebhook(url: str, headers: dict = None, only_severity: list = None) -> Callable:
    """
    Returns an on_alert handler that POSTs JSON to any HTTP endpoint.

    Parameters
    ----------
    url             : endpoint URL
    headers         : optional HTTP headers (e.g. Authorization)
    only_severity   : only send alerts of these severities (e.g. ["critical", "high"])

    Usage:
        with GreenTensor(on_alert=GenericWebhook(
            "https://my-siem.company.com/ingest",
            headers={"Authorization": "Bearer token"},
            only_severity=["critical", "high"],
        )) as gt:
            train()
    """
    def handler(event: Any):
        evt = _format_event(event)
        if only_severity and evt.get("severity") not in only_severity:
            return
        _post_json(url, evt, headers)
    return handler


# ── Multi-handler ──────────────────────────────────────────────────────────────

def MultiAlert(*handlers: Callable) -> Callable:
    """
    Combines multiple alert handlers. All handlers receive every alert.

    Usage:
        with GreenTensor(on_alert=MultiAlert(
            SlackWebhook("https://hooks.slack.com/..."),
            PagerDutyAlert("key"),
            GenericWebhook("https://siem/ingest"),
        )) as gt:
            train()
    """
    def handler(event: Any):
        for h in handlers:
            try:
                h(event)
            except Exception as e:
                logger.warning(f"Alert handler failed: {e}")
    return handler
