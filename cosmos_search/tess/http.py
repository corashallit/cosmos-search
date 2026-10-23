"""Outbound access for the TESS arm, through the shared client in cosmos_search.net."""

from cosmos_search.net import get_client, s3_list  # noqa: F401

STPUBDATA = "https://stpubdata.s3.amazonaws.com"


def get_bytes(url, missing_ok=True, cache=None):
    """Body of url, or None on 404 when missing_ok. Uncached in containers (light curves are streamed and
    discarded); kept in the persistent cache where the machine has a local data store (net/local.py), so a
    rerun reads them from disk first."""
    from cosmos_search.net.local import persistent

    return get_client().get_bytes(url, cache=persistent() if cache is None else cache, missing_ok=missing_ok)
