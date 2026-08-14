"""
Test Angel One SmartAPI Authentication.
Verifies API Key, Client Code, Password, and TOTP authentication.
Run: python scripts/test_angel_login.py
"""

import sys
from pathlib import Path
import pyotp

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import BROKER_CONFIG


def test_login():
    print("=" * 60)
    print("Testing Angel One SmartAPI Login...")
    print("=" * 60)

    cfg = BROKER_CONFIG["ANGEL_ONE"]
    api_key = cfg["API_KEY"]
    client_code = cfg["CLIENT_CODE"]
    password = cfg["PASSWORD"]
    totp_key = cfg["TOTP_KEY"]

    if not api_key or not client_code or not password or not totp_key:
        print("[WARNING] Broker credentials missing in config/settings.py!")
        print("Please fill in ANGEL_ONE credentials in config/settings.py or environment variables:")
        print("  - API_KEY")
        print("  - CLIENT_CODE")
        print("  - PASSWORD (MPIN)")
        print("  - TOTP_KEY")
        return None

    try:
        from SmartApi import SmartConnect

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_key).now()

        data = smart_api.generateSession(client_code, password, totp)
        if data and data.get("status"):
            jwt_token = data["data"]["jwtToken"]
            feed_token = smart_api.getfeedToken()
            print("[SUCCESS] Angel One Session Established!")
            print(f"  - Client Code: {client_code}")
            print(f"  - Feed Token : {feed_token[:15]}...")
            print("=" * 60)
            return smart_api, data["data"]
        else:
            print(f"[FAILED] Login failed: {data}")
            return None
    except Exception as e:
        print(f"[ERROR] Connection Exception: {e}")
        return None


if __name__ == "__main__":
    test_login()
