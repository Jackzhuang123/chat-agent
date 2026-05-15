#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网络环境辅助：确保本地服务不被系统代理劫持。"""

import os
from typing import Iterable

import httpx


_LOCAL_NO_PROXY_HOSTS = ("127.0.0.1", "localhost", "::1")


def ensure_local_no_proxy(extra_hosts: Iterable[str] = ()) -> None:
    """Append localhost-style hosts to NO_PROXY/no_proxy without clobbering user config."""
    required = list(_LOCAL_NO_PROXY_HOSTS) + [host for host in extra_hosts if host]
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        items = [item.strip() for item in existing.split(",") if item.strip()]
        seen = set(items)
        changed = False
        for host in required:
            if host not in seen:
                items.append(host)
                seen.add(host)
                changed = True
        if changed or existing:
            os.environ[env_name] = ",".join(items)


def should_bypass_api_proxy() -> bool:
    """Default to bypassing proxy for external API calls unless user explicitly enables it."""
    value = os.environ.get("CHAT_AGENT_USE_PROXY_FOR_API", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def create_httpx_client_for_api(timeout: float = 60.0) -> httpx.Client:
    """Create an httpx client with predictable proxy behavior."""
    trust_env = not should_bypass_api_proxy()
    return httpx.Client(timeout=timeout, trust_env=trust_env)
