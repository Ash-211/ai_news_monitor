"""
bot.py — Main entry point for the AI News Pipeline Discord bot.

Responsibilities:
  1. Load environment variables (DISCORD_TOKEN, PORT).
  2. Initialise a discord.ext.commands.Bot with slash-command support.
  3. Load cogs (news_commands & scheduler).
  4. Start a lightweight aiohttp web server for UptimeRobot / Render
     health-check pings.
  5. Sync the application command tree on ready.

Run with:
    python -m src.bot.bot
"""

import os
import logging
import asyncio

import discord
from discord.ext import commands
from dotenv import load_dotenv
from aiohttp import web

# ── Environment ────────────────────────────────────────────────────────────
load_dotenv()

DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
PORT: int = int(os.environ.get("PORT", 8080))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_news_bot")

# ── Bot Setup ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# ── Aiohttp Health-Check Server ────────────────────────────────────────────

async def _health_ping(request: web.Request) -> web.Response:
    """Minimal /ping endpoint so UptimeRobot can keep Render alive."""
    return web.json_response({"status": "alive"})


async def _start_health_server() -> None:
    """Start the aiohttp server in the background (non-blocking)."""
    app = web.Application()
    app.router.add_get("/ping", _health_ping)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info("Health-check server listening on port %s", PORT)

# ── Bot Events ─────────────────────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    """Fires once the bot has connected and is ready."""
    logger.info("Logged in as %s (ID: %s)", bot.user.name, bot.user.id)

    # Sync slash commands globally so they appear in all guilds.
    try:
        synced = await bot.tree.sync()
        logger.info("Synced %d application command(s).", len(synced))
    except Exception as exc:
        logger.exception("Failed to sync command tree: %s", exc)

    # Start the lightweight health-check web server.
    await _start_health_server()
    logger.info("Bot is fully operational — %s is online!", bot.user.name)

# ── Cog Loading ────────────────────────────────────────────────────────────

async def _load_cogs() -> None:
    """Dynamically load all cogs before the bot starts."""
    cog_modules = [
        "src.bot.cogs.news_commands",
        "src.bot.cogs.scheduler",
    ]
    for cog in cog_modules:
        try:
            await bot.load_extension(cog)
            logger.info("Loaded cog: %s", cog)
        except Exception as exc:
            logger.exception("Failed to load cog %s: %s", cog, exc)

# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    """Async entry point — loads cogs then starts the bot."""
    if not DISCORD_TOKEN:
        logger.critical("DISCORD_TOKEN is not set. Exiting.")
        return

    async with bot:
        await _load_cogs()
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
