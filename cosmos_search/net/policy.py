"""Per-host request policies.

Numbers come from what each service documents (docs/infra/net.md lists them with sources). Where a
service documents nothing, the default is two concurrent connections and no minimum interval.
"""

from dataclasses import dataclass, field
from urllib.parse import urlsplit

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HostPolicy:
    max_concurrent: int = 2
    min_interval: float = 0.0  # seconds between request starts, across all threads
    retry_statuses: frozenset = RETRY_STATUSES
    max_retries: int = 6
    backoff_base: float = 1.0
    backoff_cap: float = 120.0
    # Statuses that mean "you are being blacklisted"; enough of them in a row opens the breaker.
    block_statuses: frozenset = frozenset()
    breaker_threshold: int = 3
    breaker_cooldown: float = 600.0
    # Stop when the X-RateLimit-Remaining header falls to this (ADS daily quota). None = ignore.
    budget_reserve: int | None = None
    # Share one limiter across every host that matches this rule (CDS limits are per client IP).
    shared: bool = False
    timeout: float = 300.0


_CDS = HostPolicy(
    max_concurrent=2, min_interval=0.5, block_statuses=frozenset({403, 429}),
    breaker_threshold=2, breaker_cooldown=3600.0, shared=True,
)

# Rule keys are "host-suffix" or "host-suffix/path-prefix"; the most specific match wins.
POLICIES = {
    # CDS: > 6 queries/s from one IP gets a temporary blacklist. We stay at 2/s across all CDS hosts.
    "cds.unistra.fr": _CDS,
    "u-strasbg.fr": _CDS,
    # Gaia archive: <= 5 executing jobs per user (enforced in tap.py), nothing documented per request.
    "gea.esac.esa.int": HostPolicy(max_concurrent=4, min_interval=0.1),
    # Gaia bulk CDN and its S3-style listing endpoint: static files behind a CDN.
    "cdn.gea.esac.esa.int": HostPolicy(max_concurrent=8),
    "cdn77-storage.com": HostPolicy(max_concurrent=8),
    # Public S3 buckets (stpubdata, nasa-irsa-wise) are built for parallel reads.
    "amazonaws.com": HostPolicy(max_concurrent=32, backoff_base=0.5),
    # TESScut: > 10 simultaneous calls per user returns 503.
    "mast.stsci.edu/tesscut": HostPolicy(max_concurrent=8),
    "mast.stsci.edu": HostPolicy(max_concurrent=4),
    # ADS: 5,000 queries/day per token; remaining count is in the response headers.
    "api.adsabs.harvard.edu": HostPolicy(max_concurrent=2, min_interval=0.2, budget_reserve=50),
    # IRSA documents no limit; its ZTF light-curve service takes ~20 s a call, so a few run at once. Calls that hang
    # are cut short and retried.
    "irsa.ipac.caltech.edu": HostPolicy(max_concurrent=4, timeout=120.0),
    # HATS catalogues are static Parquet.
    "data.lsdb.io": HostPolicy(max_concurrent=8),
}

DEFAULT = HostPolicy()


def match(url, policies=POLICIES):
    """Return (rule_key, policy) for a URL. rule_key is None when the default applies."""
    parts = urlsplit(url)
    host, path = (parts.hostname or "").lower(), parts.path or "/"
    best, best_rank = None, (-1, -1)
    for rule in policies:
        rule_host, _, rule_path = rule.partition("/")
        if not (host == rule_host or host.endswith("." + rule_host)):
            continue
        if rule_path and not path.lstrip("/").startswith(rule_path):
            continue
        rank = (len(rule_path) if rule_path else 0, len(rule_host))
        if rank > best_rank:
            best, best_rank = rule, rank
    return best, policies.get(best, DEFAULT)


def limiter_key(url, policies=POLICIES):
    """Requests with the same key share one concurrency and rate limiter."""
    rule, policy = match(url, policies)
    host = (urlsplit(url).hostname or "").lower()
    return (rule if policy.shared else f"{host}|{rule}"), policy
