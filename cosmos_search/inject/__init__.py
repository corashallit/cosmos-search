"""Injection and decoy generation.

The distributions live here, in public. The draws do not: the CLI takes its
seed from os.urandom, writes truth only as age ciphertext, and prints counts.
It is meant to run in the cosmos-search-truth repo's Actions workflow, never
in an analysis session. See docs/scoping/blinding.md.
"""
