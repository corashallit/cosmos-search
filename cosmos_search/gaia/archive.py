"""Gaia archive endpoints, login and the shared TAP handle."""

import os
import threading

from cosmos_search.net import Tap, get_client

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LOGIN_URL = "https://gea.esac.esa.int/tap-server/login"
DATALINK_URL = "https://gea.esac.esa.int/data-server/data"
CDN_URL = "https://cdn.gea.esac.esa.int"
CDN_LIST_URL = "https://gaia.eu-1.cdn77-storage.com"

# The archive runs at most 5 jobs per user (2026-10-14 release); keep one free for interactive use.
JOB_SLOTS = 4

_lock = threading.Lock()
_tap = None
_logged_in = False


def login(client=None, *, force=False):
    """Cookie login with GAIA_USER / GAIA_PASSWORD. Without them, stay anonymous (row limits apply)."""
    global _logged_in
    client = client or get_client()
    user, password = os.environ.get("GAIA_USER"), os.environ.get("GAIA_PASSWORD")
    with _lock:
        if (_logged_in and not force) or not (user and password):
            return _logged_in
        resp = client.post(LOGIN_URL, data={"username": user, "password": password})
        resp.raise_for_status()
        _logged_in = True
        return True


def tap():
    global _tap
    login()
    with _lock:
        if _tap is None:
            _tap = Tap(TAP_URL, job_slots=JOB_SLOTS)
        return _tap
