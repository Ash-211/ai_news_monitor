"""
sender.py — WhatsApp message sender via Meta Cloud API.

Sends text replies to users through the WhatsApp Business Cloud API.
Uses httpx for async HTTP requests.

Required environment variables:
    WHATSAPP_TOKEN     – Permanent access token from Meta Developer dashboard.
    WHATSAPP_PHONE_ID  – Phone number ID from the WhatsApp Business API setup.
"""

import os
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("whatsapp_bot.sender")

GRAPH_API_VERSION = "v21.0"
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_ID}/messages"


async def send_whatsapp_message(to: str, body: str) -> dict:
    """
    Send a plain-text WhatsApp message to a recipient.

    Args:
        to:   Recipient phone number in international format (e.g. "919876543210").
        body: The message text to send. WhatsApp supports *bold*, _italic_,
              ~strikethrough~, and ```monospace``` formatting.

    Returns:
        The JSON response from Meta's API, or an error dict on failure.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logger.error("❌ WHATSAPP_TOKEN or WHATSAPP_PHONE_ID not set in environment.")
        return {"error": "WhatsApp credentials not configured."}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    # WhatsApp has a 4096 character limit per message.
    # If the body is longer, we split it into multiple messages.
    MAX_LEN = 4096
    chunks = _split_message(body, MAX_LEN)

    last_response = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": True, "body": chunk},
            }

            try:
                resp = await client.post(BASE_URL, headers=headers, json=payload)
                last_response = resp.json()

                if resp.status_code != 200:
                    logger.error(
                        "⚠️ WhatsApp API error (status %s): %s",
                        resp.status_code,
                        last_response,
                    )
                else:
                    logger.info("✅ Message sent to %s (chunk %d/%d)", to, chunks.index(chunk) + 1, len(chunks))

            except httpx.HTTPError as e:
                logger.exception("❌ HTTP error sending WhatsApp message: %s", e)
                return {"error": str(e)}

    return last_response


def _split_message(text: str, max_len: int) -> list[str]:
    """
    Split a long message into chunks that fit within WhatsApp's character limit.
    Tries to split on newlines to preserve formatting.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at the last newline before the limit
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            # No good newline found, split at a space
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            # No space found either, hard-split
            split_at = max_len

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks
