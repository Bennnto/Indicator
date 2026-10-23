import os
import requests
import subprocess
import re


# Default ntfy.sh topic (override with environment variable NTFY_TOPIC if set)
DEFAULT_NTFY_TOPIC = os.getenv("NTFY_TOPIC", "ben-secret107468")


def clean_header_text(text: str) -> str:
    """
    HTTP headers only support ASCII / Latin-1 in Python requests.
    This safely strips emojis from header values to prevent Latin-1 encoding errors.
    """
    if not text:
        return ""
    # Remove characters outside ASCII printable range
    return re.sub(r'[^\x20-\x7E]', '', text).strip()


def send_ntfy(message: str, title: str = "Crypto Trade Alert",
              topic: str = None, priority: str = "urgent",
              tags: str = "chart_with_upwards_trend,bell") -> bool:
    """
    Sends an instant push notification to your iPhone / Android via ntfy.sh.
    100% free, no login or account required, works from anywhere (Mac, Linux, GitHub Actions).
    """
    channel = topic or DEFAULT_NTFY_TOPIC
    url = f"https://ntfy.sh/{channel}"

    # Ensure title and tags are clean ASCII for HTTP headers
    safe_title = clean_header_text(title) or "Trade Signal"
    safe_tags = clean_header_text(tags) or "bell"

    headers = {
        "Title": safe_title,
        "Priority": priority,
        "Tags": safe_tags
    }

    try:
        # Message body is sent as raw UTF-8 bytes — emojis and formatting fully work here!
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
               imessage_recipient: str = None) -> bool:
    """
    Universal alert dispatcher:
    1. Sends instant lock-screen push notification via ntfy.sh (primary).
    2. Optionally mirrors to iMessage if on Mac and configured.
    """
    ntfy_success = send_ntfy(message, title=title)

    if imessage_recipient and not imessage_recipient.startswith("+1234567890"):
        send_imessage(imessage_recipient, message)

    return ntfy_success


if __name__ == "__main__":
    print("=" * 55)
    print("📲 Notification Test")
    print(f"Target Topic: https://ntfy.sh/{DEFAULT_NTFY_TOPIC}")
    print("=" * 55)

    test_msg = (
        "🟢 BUY SIGNAL: BTC/USDT\n"
        "Price: $81,350.00\n"
        "Stop-Loss: $79,800.00 (-1.91%)\n"
        "Target 1:  $83,675.00 (+2.86%)\n\n"
        "Thesis: 1h Macro Uptrend confirmed + 15m RSI Pullback."
    )

    print("Sending test notification...")
    ok = send_ntfy(test_msg, title="BTC/USDT Buy Signal (Test)")
    if ok:
        print("✅ Success! Check your phone.")
