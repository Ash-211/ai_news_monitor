"""
Discord Bot — Main Entry Point
Connects to Discord, loads command cogs, and runs a lightweight
HTTP server on $PORT for health checks. Includes a self-ping
mechanism that keeps the Render service alive during active hours
(10 AM – 9 PM IST) without needing an external uptime bot.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
import discord
from discord.ext import commands, tasks
from aiohttp import web, ClientSession
from dotenv import load_dotenv

load_dotenv()

# ── Self-Ping Configuration ───────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
ACTIVE_START_HOUR = 10  # 10 AM IST
ACTIVE_END_HOUR = 21    # 9 PM IST
_logger = logging.getLogger("bot.selfping")

# ── Bot Setup ─────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


# ── Events ────────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    print(f"✅ Bot is online as {bot.user} (ID: {bot.user.id})")
    print(f"   Connected to {len(bot.guilds)} server(s)")

    # Sync slash commands with Discord
    try:
        synced = await bot.tree.sync()
        print(f"   Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"   ⚠️  Failed to sync commands: {e}")


# ── Lightweight HTTP server for health checks ─────────────────────────────────
async def handle_ping(request):
    """Health-check endpoint — used by self-ping and GitHub Actions wake-up."""
    now_ist = datetime.now(IST).strftime("%I:%M %p IST")
    return web.Response(text=f"🤖 AI News Bot is alive! ({now_ist})", status=200)


async def start_keepalive_server():
    """Start a tiny web server on the port Render assigns (or 8080)."""
    http_app = web.Application()
    http_app.router.add_get("/", handle_ping)
    http_app.router.add_get("/ping", handle_ping)

    port = int(os.getenv("PORT", 8080))
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"   Keep-alive server running on port {port}")


# ── Self-Ping Background Task ─────────────────────────────────────────────────
async def self_ping_loop():
    """
    Pings this bot's own HTTP endpoint every 4 minutes during active hours
    (10 AM - 9 PM IST) to prevent Render from spinning it down.
    Outside active hours, it does nothing and Render sleeps naturally.
    """
    await asyncio.sleep(30)  # Wait for HTTP server to fully start

    service_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if not service_url:
        _logger.warning("RENDER_EXTERNAL_URL not set — self-ping disabled for bot.")
        return

    ping_url = f"{service_url.rstrip('/')}/ping"
    _logger.info(f"Bot self-ping started. Target: {ping_url}")

    async with ClientSession() as session:
        while True:
            try:
                now_ist = datetime.now(IST)
                if ACTIVE_START_HOUR <= now_ist.hour < ACTIVE_END_HOUR:
                    async with session.get(ping_url, timeout=15) as resp:
                        _logger.info(f"✅ Bot self-ping OK at {now_ist.strftime('%I:%M %p IST')} (status={resp.status})")
                else:
                    _logger.info(f"😴 Bot outside active hours ({now_ist.strftime('%I:%M %p IST')}). Letting Render sleep.")
            except Exception as e:
                _logger.warning(f"⚠️ Bot self-ping failed: {e}")
            await asyncio.sleep(240)  # 4 minutes


# ── Load Cogs & Run ──────────────────────────────────────────────────────────
async def main():
    """Loads cogs and starts both the bot and the keep-alive server."""
    # Load command cogs
    await bot.load_extension("src.bot.cogs.news_commands")
    await bot.load_extension("src.bot.cogs.scheduler")
    await bot.load_extension("src.bot.cogs.verify_commands")
    print("   Cogs loaded: news_commands, scheduler, verify_commands")

    # Start the keep-alive web server in the background
    await start_keepalive_server()

    # Start self-ping background task (keeps Render alive during 10 AM - 9 PM IST)
    asyncio.create_task(self_ping_loop())

    # Start the bot (this blocks until the bot disconnects)
    token = os.getenv("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ ERROR: DISCORD_BOT_TOKEN or DISCORD_TOKEN not found in environment")
        print("   Get your token from https://discord.com/developers/applications")
        return

    await bot.start(token)


if __name__ == "__main__":
    asyncio.run(main())
