import os
import requests
import subprocess
import re

# Robust fallback: handle empty string or unset env var
_env_topic = os.getenv("NTFY_TOPIC", "").strip()
DEFAULT_NTFY_TOPIC = _env_topic if _env_topic else "ben-secret107468"


def clean_header_text(text: str) -> str:
    """
    HTTP headers only support ASCII / Latin-1 in Python requests.
    This safely strips characters outside ASCII to prevent Latin-1 encoding errors.
    """
    if not text:
        return ""
    return re.sub(r'[^\x20-\x7E]', '', text).strip()


def send_ntfy(message: str, title: str = "Trade Signal",
              topic: str = None, priority: str = "urgent",
              tags: str = "chart_with_upwards_trend,bell") -> bool:
    """
    Sends an instant push notification to your phone via ntfy.sh.
    100% free, no login or account required, works from anywhere (Mac, Linux, GitHub Actions).
    """
    channel = (topic or DEFAULT_NTFY_TOPIC).strip()
    if not channel:
        channel = "ben-secret107468"

    url = f"https://ntfy.sh/{channel}"

    safe_title = clean_header_text(title) or "Trade Signal"
    safe_tags = clean_header_text(tags) or "bell"

    headers = {
        "Title": safe_title,
        "Priority": priority,
        "Tags": safe_tags
    }

    try:
        response = requests.post(
            url,
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print(f"📱 Push alert delivered to: https://ntfy.sh/{channel}")
            return True
        else:
            print(f"❌ ntfy error ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to reach ntfy.sh: {e}")
        return False


def send_imessage(recipient: str, message: str) -> bool:
    """Sends an iMessage directly via macOS Messages app (Mac only)."""
    if not recipient or recipient.startswith("+1234567890"):
        return False

    clean_message = (
        message
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\n', '\\n')
    )

    applescript = f'''
    tell application "Messages"
        try
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{recipient}" of targetService
            send "{clean_message}" to targetBuddy
            return "SUCCESS"
        on error errMsg
            return "ERROR: " & errMsg
        end try
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript],
                                capture_output=True, text=True, check=True)
        return "ERROR:" not in result.stdout
    except Exception:
        return False


def send_alert(message: str, title: str = "Trade Signal",
               imessage_recipient: str = None,
               priority: str = "urgent",
               tags: str = "chart_with_upwards_trend,bell") -> bool:
    """
    Universal alert dispatcher:
    1. Sends instant lock-screen push notification via ntfy.sh (primary).
    2. Optionally mirrors to iMessage if on Mac and configured.
    """
    ntfy_success = send_ntfy(message, title=title, priority=priority, tags=tags)

    if imessage_recipient and not imessage_recipient.startswith("+1234567890"):
        send_imessage(imessage_recipient, message)

    return ntfy_success


if __name__ == "__main__":
    print("=" * 55)
    print("📱 Notification Test")
    print(f"Target Topic: https://ntfy.sh/{DEFAULT_NTFY_TOPIC}")
    print("=" * 55)

    test_msg = (
        "🟢 BUY SIGNAL: BTC/USDT\n"
        "Price: $81,350.00\n"
        "Stop-Loss: $79,800.00 (-1.91%)\n"
        "Target 1:  $83,675.00 (+2.86%)\n\n"
        "Thesis: 1h Macro Uptrend confirmed + 15m RSI Pullback."
    )

    res = send_alert(test_msg, title="🟢 BUY BTC/USDT @ $81,350.00 | SL: $79,800.00")
    print("Delivered:", res)
