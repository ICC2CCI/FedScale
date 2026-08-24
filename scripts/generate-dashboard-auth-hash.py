#!/usr/bin/env python3
"""Generate a PBKDF2 hash for the dashboard Basic Auth Secret."""

from __future__ import annotations

import base64
import getpass
import hashlib
import secrets


ITERATIONS = 310_000


def main() -> None:
    password = getpass.getpass("Dashboard password: ")
    confirmation = getpass.getpass("Repeat password: ")
    if not password or password != confirmation:
        raise SystemExit("Passwords are empty or do not match.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    print("pbkdf2_sha256$%d$%s$%s" % (ITERATIONS, encode(salt), encode(digest)))


if __name__ == "__main__":
    main()
