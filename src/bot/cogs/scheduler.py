"""
Scheduler Cog
Runs a background loop that broadcasts the daily news briefing
to all subscribed Discord channels at 3:00 PM UTC every day.
"""

import discord
from discord.ext import commands, tasks
from datetime import datetime, date, timedelta, time as dt_time
from sqlalchemy.orm import sessionmaker
from src.ingestion.database import Article, DiscordSubscription, get_engine


# ── Constants ─────────────────────────────────────────────────────────────────
DAILY_DROP_TIME = dt_time(hour=3, minute=30)  # 3:30 AM UTC (9:00 AM IST)
MAX_ARTICLES = 5
EMBED_COLOR_PRIMARY = 0x5865F2
EMBED_COLOR_SUCCESS = 0x57F287
EMBED_COLOR_WARNING = 0xFEE75C
EMBED_COLOR_ERROR = 0xED4245


def _get_session():
    """Create a fresh database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


class DailyScheduler(commands.Cog):
    """Background task that sends the daily news drop."""

    def __init__(self, bot):
        self.bot = bot
        self.daily_drop.start()

    def cog_unload(self):
        self.daily_drop.cancel()

    @tasks.loop(time=DAILY_DROP_TIME)
    async def daily_drop(self):
        """Triggered at 3:30 AM UTC (9:00 AM IST) every day."""
        print(f"[{datetime.utcnow()}] 🕒 Daily drop triggered!")

        # Run automated ingestion and pipeline before posting
        print("  Running automated ingestion and AI pipeline...")
        try:
            import asyncio
            from src.ingestion.scheduler import run_jobs
            # run_jobs is blocking, so run it in a separate thread so it doesn't freeze the bot
            await asyncio.to_thread(run_jobs)
            print("  Ingestion complete.")
        except Exception as e:
            print(f"  ❌ Automated ingestion failed: {e}")

        session = _get_session()
        try:
            # Get all subscribed channels
            subscriptions = session.query(DiscordSubscription).all()

            if not subscriptions:
                print("  No channels subscribed. Skipping daily drop.")
                return

            print(f"  Broadcasting to {len(subscriptions)} channel(s)...")

            for sub in subscriptions:
                try:
                    channel = self.bot.get_channel(int(sub.channel_id))
                    if channel is None:
                        # Try fetching if not in cache
                        try:
                            channel = await self.bot.fetch_channel(int(sub.channel_id))
                        except Exception:
                            print(f"  ⚠️  Channel {sub.channel_id} not found. Skipping.")
                            continue

                    # Build query for today's articles
                    today = date.today()
                    tomorrow = today + timedelta(days=1)

                    query = session.query(Article).filter(
                        Article.published_at >= datetime.combine(today, datetime.min.time()),
                        Article.published_at < datetime.combine(tomorrow, datetime.min.time()),
                    )

                    # Apply category filter if the subscription has one
                    if sub.category:
                        query = query.filter(Article.category == sub.category)

                    articles = (
                        query.order_by(Article.credibility_score.desc())
                        .limit(MAX_ARTICLES)
                        .all()
                    )

                    if not articles:
                        # Fallback: latest articles (with category filter if applicable)
                        fallback_query = session.query(Article)
                        if sub.category:
                            fallback_query = fallback_query.filter(Article.category == sub.category)
                        articles = (
                            fallback_query.order_by(Article.published_at.desc())
                            .limit(MAX_ARTICLES)
                            .all()
                        )

                    if not articles:
                        continue  # Nothing to send

                    # Send header embed
                    cat_label = f" — {sub.category}" if sub.category else ""
                    header = discord.Embed(
                        title=f"📰 Daily News Briefing{cat_label} — {today.strftime('%B %d, %Y')}",
                        description=f"Top {len(articles)} AI-analyzed article(s) delivered to you automatically",
                        color=EMBED_COLOR_PRIMARY,
                    )
                    header.set_footer(text="AI News Monitor • Automated Daily Drop")
                    await channel.send(embed=header)

                    # Send each article as a separate embed with a button
                    for i, article in enumerate(articles, 1):
                        score = article.credibility_score or 0.5
                        if score >= 0.7:
                            badge = "✅ Verified"
                            color = EMBED_COLOR_SUCCESS
                        elif score >= 0.4:
                            badge = "⚠️ Uncertain"
                            color = EMBED_COLOR_WARNING
                        else:
                            badge = "🚨 Flagged"
                            color = EMBED_COLOR_ERROR

                        summary = ""
                        if article.clean_content:
                            summary = article.clean_content[:300] + "..." if len(article.clean_content) > 300 else article.clean_content
                        elif article.raw_content:
                            summary = article.raw_content[:300] + "..." if len(article.raw_content) > 300 else article.raw_content
                        else:
                            summary = "No summary available."

                        title = article.title or "Untitled Article"
                        if len(title) > 250:
                            title = title[:247] + "..."

                        embed = discord.Embed(
                            title=f"{i}. {title}",
                            description=summary,
                            color=color,
                            url=article.url,
                        )
                        embed.add_field(name="📊 Credibility", value=f"{badge} ({int(score * 100)}%)", inline=True)
                        embed.add_field(name="📂 Category", value=article.category or "General", inline=True)
                        embed.add_field(name="📰 Source", value=article.source or "Unknown", inline=True)

                        if article.published_at:
                            embed.timestamp = article.published_at

                        # Button to read the full article
                        view = discord.ui.View()
                        if article.url:
                            view.add_item(discord.ui.Button(
                                label="📖 Read Full Article",
                                url=article.url,
                                style=discord.ButtonStyle.link,
                            ))

                        await channel.send(embed=embed, view=view)

                    print(f"  ✅ Sent {len(articles)} articles to channel {sub.channel_id}")

                except Exception as e:
                    print(f"  ❌ Error sending to channel {sub.channel_id}: {e}")

        except Exception as e:
            print(f"  ❌ Daily drop error: {e}")
        finally:
            session.close()

    @daily_drop.before_loop
    async def before_daily_drop(self):
        """Wait until the bot is fully connected before starting the loop."""
        await self.bot.wait_until_ready()
        print(f"   Scheduler ready. Daily drop set for {DAILY_DROP_TIME} UTC")


# ── Cog Setup ─────────────────────────────────────────────────────────────────
async def setup(bot):
    await bot.add_cog(DailyScheduler(bot))
