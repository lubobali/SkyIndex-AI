"""One-time secret setup for SkyIndex-AI.

Creates the Databricks secret scope the deployed app reads its Lakebase
connection URL from. Run once, from a Databricks notebook or anywhere the
Databricks SDK is authenticated:

    python setup_secrets.py

The URL is read with getpass, so it is never echoed, never written to disk,
and never lands in shell history.
"""

from __future__ import annotations

import getpass
import sys

# Workspace-wide namespace. On a shared workspace a generic name like
# "database" may already exist under someone else's ownership, and writing to
# a scope you do not own fails with PermissionDenied - which reads like an
# authentication problem and is not one.
SCOPE = "lubo-skyindex"
KEY = "lakebase-url"


def clean_url(value: str) -> str:
    """Strip whitespace from inside a URL-form connection string.

    Pasting a long URL into a masked prompt can introduce line-wrap whitespace
    mid-string. The result looks right but produces a hostname containing
    spaces, which fails DNS with "Name or service not known" - an error that
    reads like a network fault and sends the investigation the wrong way.
    """
    stripped = value.strip()
    if stripped.startswith(("postgresql://", "postgres://")):
        return "".join(stripped.split())
    return stripped


def main() -> int:
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()

    print(f"Creating secret scope '{SCOPE}' (skipped if it already exists)...")
    try:
        client.secrets.create_scope(scope=SCOPE)
        print("  created")
    except Exception as exc:
        # RESOURCE_ALREADY_EXISTS is the expected path on re-runs.
        if "RESOURCE_ALREADY_EXISTS" in str(exc):
            print("  already exists")
        else:
            print(f"  could not create scope: {exc}")
            return 1

    url = clean_url(
        getpass.getpass(
            "\nLakebase connection URL\n"
            "  postgresql://<role>:<password>@<host>:5432/databricks_postgres?sslmode=require\n"
            "> "
        )
    )

    if not url:
        print("No URL entered - nothing stored.")
        return 1
    if not url.startswith(("postgresql://", "postgres://")):
        print("That does not look like a Postgres URL. Nothing stored.")
        return 1
    if "@" not in url or ":" not in url.split("@")[0][13:]:
        print(
            "The URL has no password in it. The OAuth connection string shown by\n"
            "default expires after an hour - use a native Postgres role with a\n"
            "static password instead (Branch overview -> Roles & Databases)."
        )
        return 1

    client.secrets.put_secret(scope=SCOPE, key=KEY, string_value=url)
    print(f"\nStored {SCOPE}/{KEY}.")
    print("app.yaml already points at this scope, so the app needs no further config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
