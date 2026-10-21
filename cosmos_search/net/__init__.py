"""Shared HTTP and TAP clients: User-Agent, throttling, backoff, response cache.

Every outbound request in the project goes through `get_client()`. See docs/infra/net.md.
"""

from .blind import BlindModeError, blind, check_blind
from .cache import ResponseCache
from .client import USER_AGENT, BudgetExhausted, Client, HostBlocked, IntegrityError, get_client
from .policy import POLICIES, HostPolicy
from .s3 import S3Entry, s3_list
from .tap import Tap

__all__ = [
    "BlindModeError", "BudgetExhausted", "Client", "HostBlocked", "HostPolicy", "IntegrityError",
    "POLICIES", "ResponseCache", "S3Entry", "Tap", "USER_AGENT", "blind", "check_blind", "get_client",
    "s3_list",
]
