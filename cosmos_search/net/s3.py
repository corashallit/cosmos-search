"""Anonymous listing of public S3-style buckets (stpubdata, nasa-irsa-wise, the Gaia CDN's storage)."""

from dataclasses import dataclass
from xml.etree import ElementTree

from .client import get_client


@dataclass(frozen=True)
class S3Entry:
    key: str
    size: int = 0
    is_prefix: bool = False


def s3_list(bucket_url, prefix="", delimiter=None, *, client=None, page_size=1000):
    """Yield S3Entry for every object (and, with a delimiter, every common prefix) under prefix.

    Uses ListObjectsV2 and follows continuation tokens. Endpoints that only speak V1 (no
    NextContinuationToken in a truncated page) are continued with start-after the last key.
    """
    client = client or get_client()
    url = bucket_url.rstrip("/") + "/"
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(page_size)}
    if delimiter:
        params["delimiter"] = delimiter
    while True:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
        last = None
        for c in root.iter(f"{ns}Contents"):
            key = c.findtext(f"{ns}Key")
            last = max(last or key, key)
            yield S3Entry(key, int(c.findtext(f"{ns}Size") or 0))
        for p in root.iter(f"{ns}CommonPrefixes"):
            key = p.findtext(f"{ns}Prefix")
            last = max(last or key, key)
            yield S3Entry(key, 0, True)
        if (root.findtext(f"{ns}IsTruncated") or "").lower() != "true" or last is None:
            return
        token = root.findtext(f"{ns}NextContinuationToken")
        params.pop("continuation-token", None)
        params.pop("start-after", None)
        params.pop("marker", None)
        if token:
            params["continuation-token"] = token
        else:
            params["start-after"] = last
            params["marker"] = last
