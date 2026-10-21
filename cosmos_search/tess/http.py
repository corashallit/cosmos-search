"""Outbound access for the TESS arm, through the shared client in cosmos_search.net."""

from cosmos_search.net import get_client, s3_list  # noqa: F401

STPUBDATA = "https://stpubdata.s3.amazonaws.com"


def get_bytes(url, missing_ok=True):
    """Body of url, uncached (light curves are streamed and discarded), or None on 404 when missing_ok."""
    return get_client().get_bytes(url, cache=False, missing_ok=missing_ok)
