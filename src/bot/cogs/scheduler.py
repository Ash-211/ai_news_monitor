"""
scheduler.py — Daily news broadcast scheduler cog.

Uses discord.ext.tasks to fire once a day at 09:30 UTC (3:00 PM IST).
For every row in the DiscordSubscription table it sends up to 5 articles
to the subscribed channel, filtered by the subscription's category.

Reuses build_article_embed / build_article_view from news_commands to
keep embed formatting consistent across the bot.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import List

import discord
from discord.ext import commands, tasks

from src.ingestion.database import Article, DiscordSubscription, get_session
from src.bot.cogs.news_commands import (
    build_article_embed,
    build_article_view,
    _fetch_articles_for_date,
    COLOR_DEFAULT,
)

logger = logging.getLogger("ai_news_bot.scheduler")

# Maximum articles to broadcast per channel per day.
MAX_BROADCAST_ARTICLES = 5

# 09:30 UTC  ≡  3:00 PM IST
BROADCAST_TIME = _dt.time(hour=9, minute=30, tzinfo=_dt.timezone.utc)


class DailyScheduler(commands.Cog):
    """Cog that broadcasts daily news to all subscribed Discord channels."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def cog_load(self) -> None:
        """Start the task loop when the cog is loaded."""
        self.daily_news_broadcast.start()
        logger.info(
            "Daily broadcast loop started — scheduled at %s UTC.",
            BROADCAST_TIME.strftime("%H:%M"),
        )

    async def cog_unload(self) -> None:
        """Gracefully cancel the task loop when the cog is unloaded."""
        self.daily_news_broadcast.cancel()
        logger.info("Daily broadcast loop stopped.")

    # ── Task Loop ───────────────────────────────────────────────────────────

    @tasks.loop(time=BROADCAST_TIME)
    async def daily_news_broadcast(self) -> None:
        """
        Core broadcast routine — runs once every day at BROADCAST_TIME.

        Steps:
            1. Query all subscriptions from the database.
            2. For each subscription, fetch today's articles (respecting category).
            3. Send up to MAX_BROADCAST_ARTICLES as embeds to the target channel.
            4. Log successes and handle errors (deleted channels, missing perms, etc.).
        """
        logger.info("🕤 Daily news broadcast triggered.")

        session = get_session()
        try:
            subscriptions: List[DiscordSubscription] = session.query(DiscordSubscription).all()
        finally:
            session.close()

        if not subscriptions:
            logger.info("No active subscriptions — nothing to broadcast.")
            return

        logger.info("Broadcasting to %d subscription(s).", len(subscriptions))

        for sub in subscriptions:
            try:
                channel = self.bot.get_channel(int(sub.channel_id))

                # If the channel isn't in cache, attempt to fetch it.
                if channel is None:
                    try:
                        channel = await self.bot.fetch_channel(int(sub.channel_id))
                    except (discord.NotFound, discord.Forbidden):
                        logger.warning(
                            "Channel %s (server %s) not found or inaccessible — skipping.",
                            sub.channel_id,
                            sub.server_id,
                        )
                        continue

                # Fetch today's articles (filtered by subscription category).
                articles = _fetch_articles_for_date(
                    target_date=_dt.date.today(),
                    category=sub.category,
                    limit=MAX_BROADCAST_ARTICLES,
                )

                if not articles:
                    logger.info(
                        "No articles for category '%s' today — skipping channel %s.",
                        sub.category,
                        sub.channel_id,
                    )
                    continue

                # Send a header message.
                category_label = sub.category if sub.category and sub.category.lower() != "all" else "All Categories"
                header = (
                    f"📰 **Daily News Digest** — {category_label}\n"
                    f"*{_dt.date.today().strftime('%A, %B %d, %Y')}* • {len(articles)} article(s)"
                )
                await channel.send(header)

                # Send each article as its own embed + button.
                for article in articles:
                    await channel.send(
                        embed=build_article_embed(article),
                        view=build_article_view(article),
                    )

                logger.info(
                    "✅ Sent %d article(s) to channel %s (server %s, category: %s).",
                    len(articles),
                    sub.channel_id,
                    sub.server_id,
                    sub.category,
                )

            except discord.Forbidden:
                logger.warning(
                    "Missing permissions to send in channel %s (server %s).",
                    sub.channel_id,
                    sub.server_id,
                )
            except discord.HTTPException as exc:
                logger.error(
                    "HTTP error broadcasting to channel %s: %s",
                    sub.channel_id,
                    exc,
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected error broadcasting to channel %s: %s",
                    sub.channel_id,
                    exc,
                )

        logger.info("🏁 Daily broadcast cycle complete.")

    @daily_news_broadcast.before_loop
    async def _wait_until_ready(self) -> None:
        """Ensure the bot is fully connected before the first broadcast."""
        await self.bot.wait_until_ready()
        logger.info("Bot is ready — scheduler armed.")


# ── Cog setup hook (required by discord.py) ─────────────────────────────────

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(DailyScheduler(bot))
    logger.info("DailyScheduler cog loaded.")
