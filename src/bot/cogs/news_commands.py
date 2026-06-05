"""
news_commands.py — Slash-command cog for the AI News Pipeline bot.

Commands:
    /newsdaily         – Show today's top articles.
    /news              – Search articles by date and/or tag.
    /newscategories    – Interactive category picker for today's news.
    /setup_daily       – Subscribe this channel to the daily news broadcast.
    /remove_daily      – Unsubscribe this channel from the daily news broadcast.

Helpers (importable by other cogs):
    build_article_embed(article)  – Returns a styled discord.Embed.
    build_article_view(article)   – Returns a discord.ui.View with a URL button.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List

import discord
from discord import app_commands
from discord.ext import commands
from sqlalchemy import func

from src.ingestion.database import Article, DiscordSubscription, get_session

logger = logging.getLogger("ai_news_bot.news_commands")

# ── Colour palette for credibility badges ──────────────────────────────────
COLOR_CREDIBLE   = discord.Colour(0x2ECC71)  # Green  – score >= 0.6
COLOR_UNCERTAIN  = discord.Colour(0xF39C12)  # Orange – 0.4 <= score < 0.6
COLOR_FLAGGED    = discord.Colour(0xE74C3C)  # Red    – score < 0.4
COLOR_DEFAULT    = discord.Colour(0x3498DB)  # Blue   – score is None

MAX_EMBED_ARTICLES = 10  # Cap for interactive commands
MAX_SUMMARY_LEN    = 200

# ════════════════════════════════════════════════════════════════════════════
#  Helper Functions (shared with scheduler cog)
# ════════════════════════════════════════════════════════════════════════════

def _credibility_colour(score: Optional[float]) -> discord.Colour:
    """Return an embed colour based on the article's credibility score."""
    if score is None:
        return COLOR_DEFAULT
    if score >= 0.6:
        return COLOR_CREDIBLE
    if score >= 0.4:
        return COLOR_UNCERTAIN
    return COLOR_FLAGGED


def _credibility_label(score: Optional[float]) -> str:
    """Human-friendly label for the credibility score."""
    if score is None:
        return "N/A"
    if score >= 0.6:
        return f"✅ {score:.0%}"
    if score >= 0.4:
        return f"⚠️ {score:.0%}"
    return f"🚩 {score:.0%}"


def _article_summary(article: Article) -> str:
    """Pick the best available summary text, truncated to MAX_SUMMARY_LEN."""
    text = (
        article.summary_abstractive
        or article.summary_extractive
        or article.clean_content
        or article.raw_content
        or "No summary available."
    )
    if len(text) > MAX_SUMMARY_LEN:
        text = text[:MAX_SUMMARY_LEN].rsplit(" ", 1)[0] + "…"
    return text


def build_article_embed(article: Article) -> discord.Embed:
    """
    Build a beautifully formatted Discord embed for a single article.

    Layout:
        Title        → embed title (linked to article URL)
        Description  → truncated summary
        Fields       → Category, Credibility
        Footer       → source & published date
        Colour       → based on credibility score
    """
    embed = discord.Embed(
        title=article.title or "Untitled Article",
        url=article.url,
        description=_article_summary(article),
        colour=_credibility_colour(article.credibility_score),
        timestamp=article.published_at or datetime.utcnow(),
    )

    # Category badge
    category_display = article.category or "Uncategorised"
    embed.add_field(name="📂 Category", value=category_display, inline=True)

    # Credibility badge
    embed.add_field(
        name="🛡️ Credibility",
        value=_credibility_label(article.credibility_score),
        inline=True,
    )

    # Source & timestamp footer
    source_text = article.source or "Unknown Source"
    embed.set_footer(text=f"Source: {source_text} • AI News Pipeline")

    return embed


def build_article_view(article: Article) -> discord.ui.View:
    """
    Return a View with a single URL button linking to the full article.
    URL buttons do not require an interaction callback.
    """
    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            label="Read Full Article",
            style=discord.ButtonStyle.link,
            url=article.url,
            emoji="📰",
        )
    )
    return view


# ════════════════════════════════════════════════════════════════════════════
#  Internal DB query helpers & Pagination View
# ════════════════════════════════════════════════════════════════════════════

def fetch_articles(
    session,
    category: Optional[str] = None,
    target_date: Optional[date] = None,
    tag: Optional[str] = None,
    offset: int = 0,
    limit: int = 5,
) -> List[Article]:
    """
    Unified query function supporting category, date, and keyword search filters, with pagination offset.
    Requires an active database session.
    """
    query = session.query(Article)
    
    if target_date:
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        query = query.filter(Article.published_at >= start, Article.published_at < end)
        
    if category and category.lower() != "all":
        query = query.filter(func.lower(Article.category) == category.lower())
        
    if tag:
        tag_term = tag.strip()
        query = query.filter(
            (Article.title.ilike(f"%{tag_term}%")) |
            (Article.keywords.ilike(f"%{tag_term}%")) |
            (Article.category.ilike(f"%{tag_term}%"))
        )
        
    return query.order_by(Article.published_at.desc(), Article.id.desc()).offset(offset).limit(limit).all()


def get_fallback_date(session, category: Optional[str] = None, tag: Optional[str] = None) -> Optional[date]:
    """
    Finds the date of the most recent article, optionally filtered by category or tag.
    Requires an active database session.
    """
    query = session.query(Article)
    if category and category.lower() != "all":
        query = query.filter(func.lower(Article.category) == category.lower())
    if tag:
        tag_term = tag.strip()
        query = query.filter(
            (Article.title.ilike(f"%{tag_term}%")) |
            (Article.keywords.ilike(f"%{tag_term}%")) |
            (Article.category.ilike(f"%{tag_term}%"))
        )
    latest = query.order_by(Article.published_at.desc(), Article.id.desc()).first()
    return latest.published_at.date() if latest and latest.published_at else None


class NewsPaginationView(discord.ui.View):
    """
    A view containing:
    1. A URL link button to "Read Full Article".
    2. A "Show More ➡️" button to load the next page of 5 articles.
    """
    def __init__(
        self,
        command_type: str,  # "daily", "category", "search"
        offset: int,
        category: Optional[str] = None,
        search_date: Optional[date] = None,
        search_tag: Optional[str] = None,
        article_url: Optional[str] = None,
        start_index: int = 1,
    ) -> None:
        super().__init__(timeout=None)
        self.command_type = command_type
        self.offset = offset
        self.category = category
        self.search_date = search_date
        self.search_tag = search_tag
        self.start_index = start_index

        if article_url:
            self.add_item(
                discord.ui.Button(
                    label="Read Full Article",
                    style=discord.ButtonStyle.link,
                    url=article_url,
                    emoji="📰",
                )
            )

        self.show_more_btn = discord.ui.Button(
            label="Show More ➡️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"show_more:{command_type}:{offset}",
        )
        self.show_more_btn.callback = self.show_more_callback
        self.add_item(self.show_more_btn)

    async def show_more_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        self.remove_item(self.show_more_btn)
        await interaction.message.edit(view=self)

        limit = 6
        session = get_session()
        try:
            batch = fetch_articles(
                session,
                category=self.category,
                target_date=self.search_date,
                tag=self.search_tag,
                offset=self.offset,
                limit=limit,
            )

            if not batch:
                await interaction.followup.send(content="📭 No more articles found.", ephemeral=True)
                return

            display_count = min(len(batch), 5)
            for i in range(display_count):
                article = batch[i]
                current_index = self.start_index + i
                is_last = (i == display_count - 1)
                has_more = (len(batch) > 5)

                if is_last and has_more:
                    view = NewsPaginationView(
                        command_type=self.command_type,
                        offset=self.offset + 5,
                        category=self.category,
                        search_date=self.search_date,
                        search_tag=self.search_tag,
                        article_url=article.url,
                        start_index=current_index + 1,
                    )
                else:
                    view = build_article_view(article)
                
                await interaction.followup.send(
                    embed=build_article_embed(article), 
                    view=view
                )
        finally:
            session.close()



# ════════════════════════════════════════════════════════════════════════════
#  Category Select Menu
# ════════════════════════════════════════════════════════════════════════════

class CategorySelect(discord.ui.Select):
    """Dropdown menu that lists all available news categories."""

    def __init__(self, options_list: list[discord.SelectOption]) -> None:
        super().__init__(
            placeholder="🔍 Select a news category...",
            min_values=1,
            max_values=1,
            options=options_list,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        chosen = self.values[0]
        await interaction.response.defer()

        session = get_session()
        try:
            # Check if today has articles in this category, otherwise fall back to latest date
            target_date = date.today()
            test_articles = fetch_articles(session, category=chosen, target_date=target_date, limit=1)
            
            fallback_msg = ""
            if not test_articles:
                fallback = get_fallback_date(session, category=chosen)
                if fallback:
                    target_date = fallback
                    fallback_msg = f" (No news today. Showing news from **{target_date}** instead)"
                else:
                    target_date = None

            if not target_date:
                embed = discord.Embed(
                    title="📭 No News Found",
                    description=f"No articles found for **{chosen}** in the database.",
                    colour=COLOR_DEFAULT,
                )
                await interaction.followup.send(embed=embed, ephemeral=True)
                return

            # Fetch page 1 (limit 6)
            batch = fetch_articles(session, category=chosen, target_date=target_date, offset=0, limit=6)
            
            await interaction.followup.send(
                content=f"📂 **{chosen}** — {min(len(batch), 5)} article(s) shown{fallback_msg}:",
                ephemeral=False,
            )

            display_count = min(len(batch), 5)
            for i in range(display_count):
                article = batch[i]
                is_last = (i == display_count - 1)
                has_more = (len(batch) > 5)

                if is_last and has_more:
                    view = NewsPaginationView(
                        command_type="category",
                        offset=5,
                        category=chosen,
                        search_date=target_date,
                        article_url=article.url,
                    )
                else:
                    view = build_article_view(article)

                await interaction.followup.send(
                    embed=build_article_embed(article),
                    view=view,
                )
        finally:
            session.close()


class CategoryView(discord.ui.View):
    """Wrapper view for the CategorySelect dropdown."""

    def __init__(self, options_list: list[discord.SelectOption]) -> None:
        super().__init__(timeout=120)
        self.add_item(CategorySelect(options_list))


# ════════════════════════════════════════════════════════════════════════════
#  Cog
# ════════════════════════════════════════════════════════════════════════════

class NewsCommands(commands.Cog):
    """Slash commands for browsing and subscribing to AI-curated news."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ── /newsdaily ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="newsdaily",
        description="Get today's top news articles.",
    )
    async def newsdaily(self, interaction: discord.Interaction) -> None:
        """Fetch and display up to 5 articles with pagination."""
        await interaction.response.defer()

        session = get_session()
        try:
            target_date = date.today()
            test_articles = fetch_articles(session, target_date=target_date, limit=1)
            
            fallback_msg = ""
            if not test_articles:
                fallback = get_fallback_date(session)
                if fallback:
                    target_date = fallback
                    fallback_msg = f" (No news today. Showing news from **{target_date}** instead)"
                else:
                    target_date = None

            if not target_date:
                embed = discord.Embed(
                    title="📭 No News",
                    description="No news articles have been found in the database. Check back later!",
                    colour=COLOR_DEFAULT,
                    timestamp=datetime.utcnow(),
                )
                embed.set_footer(text="AI News Pipeline")
                await interaction.followup.send(embed=embed)
                return

            batch = fetch_articles(session, target_date=target_date, offset=0, limit=6)

            await interaction.followup.send(
                content=f"📰 **Daily News** — {min(len(batch), 5)} article(s) shown{fallback_msg}:"
            )

            display_count = min(len(batch), 5)
            for i in range(display_count):
                article = batch[i]
                is_last = (i == display_count - 1)
                has_more = (len(batch) > 5)

                if is_last and has_more:
                    view = NewsPaginationView(
                        command_type="daily",
                        offset=5,
                        search_date=target_date,
                        article_url=article.url,
                    )
                else:
                    view = build_article_view(article)

                await interaction.followup.send(
                    embed=build_article_embed(article),
                    view=view,
                )
        finally:
            session.close()

    # ── /news ───────────────────────────────────────────────────────────────

    @app_commands.command(
        name="news",
        description="Search articles by date and/or category tag.",
    )
    @app_commands.describe(
        date="Date to search (YYYY-MM-DD format).",
        tag="Category / tag to filter by.",
    )
    async def news(
        self,
        interaction: discord.Interaction,
        date: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Flexible article search with optional date and tag filters."""

        # At least one filter is required.
        if not date and not tag:
            embed = discord.Embed(
                title="❌ Missing Parameters",
                description="Please provide a **date** or **tag** (or both) to search.",
                colour=COLOR_FLAGGED,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        # Parse and validate the date string (if provided).
        target_date: Optional[date] = None
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                embed = discord.Embed(
                    title="❌ Invalid Date",
                    description="Invalid date format. Please use **YYYY-MM-DD**.",
                    colour=COLOR_FLAGGED,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

        await interaction.response.defer()

        session = get_session()
        try:
            # Fetch page 1 (limit 6)
            batch = fetch_articles(session, category=tag, target_date=target_date, offset=0, limit=6)

            if not batch:
                desc_parts: list[str] = []
                if target_date:
                    desc_parts.append(f"date **{target_date}**")
                if tag:
                    desc_parts.append(f"tag **{tag}**")
                criteria = " and ".join(desc_parts)

                embed = discord.Embed(
                    title="📭 No Results",
                    description=f"No news found for {criteria}.",
                    colour=COLOR_DEFAULT,
                )
                await interaction.followup.send(embed=embed)
                return

            header_parts: list[str] = []
            if target_date:
                header_parts.append(str(target_date))
            if tag:
                header_parts.append(tag)

            await interaction.followup.send(
                content=f"📰 **News** ({', '.join(header_parts)}) — {min(len(batch), 5)} result(s) shown:"
            )

            display_count = min(len(batch), 5)
            for i in range(display_count):
                article = batch[i]
                is_last = (i == display_count - 1)
                has_more = (len(batch) > 5)

                if is_last and has_more:
                    view = NewsPaginationView(
                        command_type="search",
                        offset=5,
                        search_date=target_date,
                        search_tag=tag,
                        article_url=article.url,
                    )
                else:
                    view = build_article_view(article)

                await interaction.followup.send(
                    embed=build_article_embed(article),
                    view=view,
                )
        finally:
            session.close()

    # ── /newscategories ─────────────────────────────────────────────────────

    @app_commands.command(
        name="newscategories",
        description="Browse today's news by category.",
    )
    async def newscategories(self, interaction: discord.Interaction) -> None:
        """Show a dropdown of all categories with article counts."""
        session = get_session()
        try:
            today = date.today()
            start = datetime.combine(today, datetime.min.time())
            end = datetime.combine(today + timedelta(days=1), datetime.min.time())

            rows = (
                session.query(Article.category, func.count(Article.id))
                .filter(Article.published_at >= start, Article.published_at < end)
                .group_by(Article.category)
                .all()
            )
            
            # If no news today, get counts from the fallback date
            if not rows:
                fallback_date = get_fallback_date(session)
                if fallback_date:
                    start_fb = datetime.combine(fallback_date, datetime.min.time())
                    end_fb = datetime.combine(fallback_date + timedelta(days=1), datetime.min.time())
                    rows = (
                        session.query(Article.category, func.count(Article.id))
                        .filter(Article.published_at >= start_fb, Article.published_at < end_fb)
                        .group_by(Article.category)
                        .all()
                    )
        finally:
            session.close()

        if not rows:
            embed = discord.Embed(
                title="📭 No Categories",
                description="No articles have been found in the database.",
                colour=COLOR_DEFAULT,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        options: list[discord.SelectOption] = []
        for cat, count in rows:
            label = cat or "Uncategorised"
            options.append(
                discord.SelectOption(
                    label=f"{label} ({count})",
                    value=label,
                    description=f"{count} article(s)",
                )
            )

        # Discord select menus allow at most 25 options.
        options = options[:25]

        view = CategoryView(options)
        embed = discord.Embed(
            title="📂 Categories",
            description="Pick a category from the dropdown to view recent articles.",
            colour=COLOR_DEFAULT,
        )
        await interaction.response.send_message(embed=embed, view=view)


    # ── /setup_daily ────────────────────────────────────────────────────────

    @app_commands.command(
        name="setup_daily",
        description="Subscribe this channel to the daily news broadcast.",
    )
    @app_commands.describe(
        category="News category to receive (default: all).",
    )
    @app_commands.checks.has_permissions(manage_channels=True)
    async def setup_daily(
        self,
        interaction: discord.Interaction,
        category: Optional[str] = "all",
    ) -> None:
        """Register or update a daily news subscription for this channel."""
        guild_id = str(interaction.guild_id)
        channel_id = str(interaction.channel_id)

        session = get_session()
        try:
            existing = (
                session.query(DiscordSubscription)
                .filter_by(server_id=guild_id, channel_id=channel_id)
                .first()
            )

            if existing:
                existing.category = category
                action = "updated"
            else:
                sub = DiscordSubscription(
                    server_id=guild_id,
                    channel_id=channel_id,
                    category=category,
                )
                session.add(sub)
                action = "created"

            session.commit()
        finally:
            session.close()

        embed = discord.Embed(
            title="✅ Daily News Subscription",
            description=(
                f"Subscription **{action}** for this channel.\n\n"
                f"**Category:** {category}\n"
                f"**Schedule:** Every day at 3:00 PM IST (9:30 AM UTC)"
            ),
            colour=COLOR_CREDIBLE,
            timestamp=datetime.utcnow(),
        )
        embed.set_footer(text="AI News Pipeline • /remove_daily to unsubscribe")
        await interaction.response.send_message(embed=embed)

    @setup_daily.error
    async def setup_daily_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        """Handle permission errors for /setup_daily."""
        if isinstance(error, app_commands.MissingPermissions):
            embed = discord.Embed(
                title="🔒 Permission Denied",
                description="You need the **Manage Channels** permission to use this command.",
                colour=COLOR_FLAGGED,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            logger.exception("Unhandled error in /setup_daily: %s", error)

    # ── /remove_daily ───────────────────────────────────────────────────────

    @app_commands.command(
        name="remove_daily",
        description="Unsubscribe this channel from the daily news broadcast.",
    )
    @app_commands.checks.has_permissions(manage_channels=True)
    async def remove_daily(self, interaction: discord.Interaction) -> None:
        """Remove the daily subscription for this channel."""
        guild_id = str(interaction.guild_id)
        channel_id = str(interaction.channel_id)

        session = get_session()
        try:
            existing = (
                session.query(DiscordSubscription)
                .filter_by(server_id=guild_id, channel_id=channel_id)
                .first()
            )

            if not existing:
                embed = discord.Embed(
                    title="📭 No Subscription",
                    description="This channel does not have an active daily news subscription.",
                    colour=COLOR_DEFAULT,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            session.delete(existing)
            session.commit()
        finally:
            session.close()

        embed = discord.Embed(
            title="🗑️ Subscription Removed",
            description="This channel will no longer receive daily news broadcasts.",
            colour=COLOR_UNCERTAIN,
            timestamp=datetime.utcnow(),
        )
        embed.set_footer(text="AI News Pipeline • /setup_daily to re-subscribe")
        await interaction.response.send_message(embed=embed)

    @remove_daily.error
    async def remove_daily_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        """Handle permission errors for /remove_daily."""
        if isinstance(error, app_commands.MissingPermissions):
            embed = discord.Embed(
                title="🔒 Permission Denied",
                description="You need the **Manage Channels** permission to use this command.",
                colour=COLOR_FLAGGED,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            logger.exception("Unhandled error in /remove_daily: %s", error)


# ── Cog setup hook (required by discord.py) ─────────────────────────────────

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(NewsCommands(bot))
    logger.info("NewsCommands cog loaded.")
