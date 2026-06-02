"""
One-time Reddit login → saves an authenticated session (storage_state) the
scraper can reuse, so it browses as logged-in (to get past the datacenter-IP
"blocked by network security" page).

USAGE — run on a machine WITH a display (your laptop), so you can handle any
captcha / 2FA in the visible window:

    pip install playwright && playwright install chromium      # if not already
    python3 login_reddit.py                                    # prompts for password
    # then copy the saved session to the VM:
    scp reddit_state.json ubuntu@<vm-ip>:~/Documents/agent-reddit-backend/

Env knobs:
    REDDIT_USERNAME        (else prompts)
    REDDIT_PASSWORD        (else prompts securely; avoids shell history)
    REDDIT_STORAGE_STATE   output path (default: reddit_state.json)
    HEADLESS=1             run headless (only works if NO captcha/2FA appears)

⚠️  reddit_state.json holds your live session — treat it like a password.
    It's gitignored; never commit it. Re-run this when the session expires.
"""

import getpass
import os
from playwright.sync_api import sync_playwright

USER = os.environ.get("REDDIT_USERNAME") or input("Reddit username: ").strip()
PASS = os.environ.get("REDDIT_PASSWORD") or getpass.getpass("Reddit password: ")
OUT = os.environ.get("REDDIT_STORAGE_STATE", "reddit_state.json")
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _try(page, selectors, value=None):
    """Best-effort: fill (if value) or click the first selector that exists."""
    for sel in selectors:
        try:
            if value is None:
                page.click(sel, timeout=4000)
            else:
                page.fill(sel, value, timeout=4000)
            return True
        except Exception:
            continue
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()
        page.goto("https://www.reddit.com/login/", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)

        # Best-effort autofill (Reddit's login markup changes / uses web components;
        # if these miss, just finish the login by hand in the window).
        filled_u = _try(page, ['input[name="username"]', '#login-username', '#loginUsername'], USER)
        filled_p = _try(page, ['input[name="password"]', '#login-password', '#loginPassword'], PASS)
        if filled_u and filled_p:
            _try(page, ['button[type="submit"]', 'button:has-text("Log In")', 'button:has-text("Log in")'])
        else:
            print("(couldn't auto-fill the form — log in manually in the window)")

        print("\n→ Finish logging in inside the browser window (solve any captcha / 2FA).")
        input("→ Once you're logged in, press Enter HERE to save the session... ")

        ctx.storage_state(path=OUT)
        print(f"\n✓ Saved authenticated session to: {OUT}")
        browser.close()


if __name__ == "__main__":
    main()
