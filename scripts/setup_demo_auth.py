#!/usr/bin/env python3
"""
One-time setup for the demo's access control.

1. Logs you in to Modal if needed (opens your browser).
2. Asks for the password people will type to open the demo UI.
3. Creates a Modal proxy auth token.
4. Stores both in the Modal secret "demo-auth" (MODAL_PROXY_TOKEN, DEMO_PASSWORD).
5. Redeploys inferencing.py and streamlitserve.py.
6. Checks that the backend now rejects requests without the token.

Usage (inside the repo, with the Python env that has Modal installed):
    git switch dev && python scripts/setup_demo_auth.py

Re-run it any time to change the password; it also issues a fresh token.
"""

import getpass
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_NAME = "demo-auth"
TOKEN_NAME = "vllm-rag-demo-ui"
TOKENS_PAGE = "https://modal.com/settings/proxy-auth-tokens"
BACKEND_MODELS_URL = "https://{workspace}--vllm-chat-rag-ragllmconfig-serve.modal.run/v1/models"

MODAL_CMD = [shutil.which("modal")] if shutil.which("modal") else [sys.executable, "-m", "modal"]


def modal(*args, capture=False):
    """Run a Modal CLI command from the repo root."""
    return subprocess.run([*MODAL_CMD, *args], cwd=ROOT, text=True, capture_output=capture)


def step(msg):
    print(f"\n==> {msg}", flush=True)


def ensure_logged_in():
    step("Checking your Modal login")
    if modal("app", "list", capture=True).returncode == 0:
        print("Logged in.")
        return
    print("Not logged in. Your browser will open: approve the login there.")
    if modal("setup").returncode != 0 or modal("app", "list", capture=True).returncode != 0:
        sys.exit("Modal login failed. Run `modal setup` yourself, then re-run this script.")


def ask_password():
    step("Pick the password people will type to open the demo UI")
    while True:
        password = getpass.getpass("Demo password (hidden): ")
        if len(password) < 6:
            print("Use at least 6 characters.")
            continue
        if getpass.getpass("Repeat it: ") == password:
            return password
        print("They didn't match. Try again.")


def create_proxy_token():
    step("Creating a Modal proxy auth token")
    res = modal("workspace", "proxy-tokens", "create", "--name", TOKEN_NAME, "--json", capture=True)
    if res.returncode == 0:
        try:
            data = json.loads(res.stdout)
            print("Created.")
            return f'{data["Modal-Key"]}.{data["Modal-Secret"]}'
        except (ValueError, KeyError):
            pass
    # Modal CLI older than 1.5.1 has no `workspace proxy-tokens` command.
    print(
        "Your Modal CLI can't create proxy tokens (needs 1.5.1+). Create one in the dashboard:\n"
        f"  {TOKENS_PAGE}\n"
        "then paste its two values here (input is hidden)."
    )
    key = getpass.getpass("Modal-Key (starts with wk-): ").strip()
    secret = getpass.getpass("Modal-Secret (starts with ws-): ").strip()
    if not (key.startswith("wk-") and secret.startswith("ws-")):
        sys.exit("Those don't look like proxy token values (wk-... / ws-...). Nothing was changed.")
    return f"{key}.{secret}"


def store_secret(token, password):
    step(f'Storing both in the Modal secret "{SECRET_NAME}"')
    res = modal(
        "secret", "create", SECRET_NAME,
        f"MODAL_PROXY_TOKEN={token}",
        f"DEMO_PASSWORD={password}",
        "--force",
        capture=True,
    )
    if res.returncode != 0:
        sys.exit(f"Couldn't create the secret:\n{res.stderr or res.stdout}")
    print("Stored.")


def deploy():
    for file in ("inferencing.py", "streamlitserve.py"):
        step(f"Deploying {file}")
        if modal("deploy", file).returncode != 0:
            sys.exit(f"Deploying {file} failed (see above). Fix it, then run `modal deploy {file}`.")


def verify():
    step("Checking the backend refuses requests without the token")
    workspace = os.getenv("MODAL_WORKSPACE", "dim-lambros")
    url = BACKEND_MODELS_URL.format(workspace=workspace)
    try:
        urllib.request.urlopen(url, timeout=30)
        print(f"WARNING: {url} answered without a token. Check the deploy output above.")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("Locked: requests without the token get 401.")
        else:
            print(f"Expected 401 but got {e.code} from {url}. "
                  "If your workspace isn't 'dim-lambros', set MODAL_WORKSPACE and check by hand.")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Couldn't reach {url} ({e}). Check it by hand: it should return 401.")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep messages in order with Modal's own output
    print("Access-control setup for the vLLM RAG demo on Modal.")
    ensure_logged_in()
    password = ask_password()
    token = create_proxy_token()
    store_secret(token, password)
    deploy()
    verify()
    print("\nDone. Open the Streamlit URL printed above and enter your demo password.")


if __name__ == "__main__":
    main()
