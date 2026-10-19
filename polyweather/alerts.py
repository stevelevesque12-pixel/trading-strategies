"""Alerts to your phone: Telegram, ntfy push, or a Slack/Discord webhook.

Configure any combination (all optional; with none set, alerts only print):
    ALERT_TELEGRAM_TOKEN + ALERT_TELEGRAM_CHAT_ID   bot from @BotFather; chat id from @userinfobot
    ALERT_NTFY_TOPIC                                 free push via the ntfy app: pick an unguessable topic
    ALERT_NTFY_SERVER                                default https://ntfy.sh
    ALERT_WEBHOOK_URL                                Slack ({"text"}) or Discord ({"content"}) incoming webhook

Levels: "urgent" (halts, money-path failures) always sends; "info" (daily
report, cash-outs) sends; "warn" is throttled per key so a flapping error
can't flood you.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import http


@dataclass
class Alerter:
    telegram_token: Optional[str] = None
    telegram_chat: Optional[str] = None
    ntfy_topic: Optional[str] = None
    ntfy_server: str = "https://ntfy.sh"
    webhook_url: Optional[str] = None
    throttle_s: float = 1800.0
    prefix: str = "polyweather"
    _last: dict = field(default_factory=dict)
    _post: Optional[Callable] = None  # injectable for tests

    @classmethod
    def from_env(cls) -> "Alerter":
        e = os.environ
        return cls(telegram_token=e.get("ALERT_TELEGRAM_TOKEN"), telegram_chat=e.get("ALERT_TELEGRAM_CHAT_ID"),
                   ntfy_topic=e.get("ALERT_NTFY_TOPIC"), ntfy_server=e.get("ALERT_NTFY_SERVER", "https://ntfy.sh"),
                   webhook_url=e.get("ALERT_WEBHOOK_URL"))

    @property
    def channels(self) -> list[str]:
        out = []
        if self.telegram_token and self.telegram_chat:
            out.append("telegram")
        if self.ntfy_topic:
            out.append("ntfy")
        if self.webhook_url:
            out.append("webhook")
        return out

    def send(self, title: str, body: str = "", level: str = "info", key: Optional[str] = None) -> bool:
        """Returns True if at least one channel accepted it (or it was printed with none configured)."""
        if level == "warn":
            k = key or title
            if time.time() - self._last.get(k, 0.0) < self.throttle_s:
                return False
            self._last[k] = time.time()
        icon = {"urgent": "🚨", "warn": "⚠️", "info": "ℹ️"}.get(level, "")
        text = f"{icon} {self.prefix}: {title}" + (f"\n{body}" if body else "")
        print(f"  [alert:{level}] {title}" + (f" | {body}" if body else ""), flush=True)
        post = self._post or (lambda url, **kw: http.session().post(url, timeout=15, **kw).raise_for_status())
        ok = not self.channels
        if "telegram" in self.channels:
            ok |= self._try(post, f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                            json={"chat_id": self.telegram_chat, "text": text[:4000]})
        if "ntfy" in self.channels:
            safe_title = f"{self.prefix}: {title}"[:200].encode("latin-1", "replace").decode("latin-1")
            headers = {"Title": safe_title, "Priority": "urgent" if level == "urgent" else "default"}
            ok |= self._try(post, f"{self.ntfy_server.rstrip('/')}/{self.ntfy_topic}",
                            data=(body or title).encode(), headers=headers)
        if "webhook" in self.channels:
            key_name = "content" if "discord" in self.webhook_url else "text"
            ok |= self._try(post, self.webhook_url, json={key_name: text[:1900]})
        return ok

    @staticmethod
    def _try(post, url, **kw) -> bool:
        try:
            post(url, **kw)
            return True
        except Exception as exc:
            print(f"  alert delivery failed ({url.split('/')[2]}): {exc}", flush=True)
            return False
