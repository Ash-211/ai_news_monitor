"""
Discord Bot — Main Entry Point
Connects to Discord, loads command cogs, and runs a lightweight
HTTP server on $PORT so Render's health-check (and UptimeRobot)
can keep the free-tier instance awake 24/7.
"""

import os
import asyncio
import discord
from discord.ext import commands
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

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


# ── Lightweight HTTP server for Render keep-alive ─────────────────────────────
async def handle_ping(request):
    """Health-check endpoint — UptimeRobot hits this every 14 min."""
    return web.Response(text="🤖 AI News Bot is alive!", status=200)


async def start_keepalive_server():
    """Start a tiny web server on the port Render assigns (or 8080)."""
    app = web.Application()
    app.router.add_get("/", handle_ping)
    app.router.add_get("/ping", handle_ping)

    port = int(os.getenv("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"   Keep-alive server running on port {port}")


# ── Load Cogs & Run ──────────────────────────────────────────────────────────
async def main():
    """Loads cogs and starts both the bot and the keep-alive server."""
    # Load command cogs
    await bot.load_extension("src.bot.cogs.news_commands")
    await bot.load_extension("src.bot.cogs.scheduler")
    print("   Cogs loaded: news_commands, scheduler")

    # Start the keep-alive web server in the background
    await start_keepalive_server()

    # Start the bot (this blocks until the bot disconnects)
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("❌ ERROR: DISCORD_BOT_TOKEN not found in .env")
        print("   Get your token from https://discord.com/developers/applications")
        return

    await bot.start(token)


if __name__ == "__main__":
    asyncio.run(main())
