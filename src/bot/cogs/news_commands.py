"""
News Commands Cog
Implements all custom slash commands for the AI News Discord Bot:
  /newsdaily       — Today's top news
  /news            — Search by date or tag
  /newscategories  — Interactive category dropdown
  /setup_daily     — Admin command to register channel for daily drops
  /remove_daily    — Admin command to unregister channel
"""

import discord
from discord.ext import commands
from discord import app_commands
from datetime import datetime, date, timedelta
from typing import Optional, List
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from src.ingestion.database import Article, DiscordSubscription, get_engine
import os


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ARTICLES_PER_EMBED = 5        # How many articles to show per command
EMBED_COLOR_PRIMARY = 0x5865F2    # Discord blurple
EMBED_COLOR_SUCCESS = 0x57F287    # Green
EMBED_COLOR_ERROR = 0xED4245      # Red
EMBED_COLOR_WARNING = 0xFEE75C    # Yellow


# ── Helper Functions ──────────────────────────────────────────────────────────
# ── Helper Functions & Pagination View ──────────────────────────────────────────
def _get_session():
    """Create a fresh database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def fetch_articles(
    category: str = None,
    target_date: date = None,
    tag: str = None,
    offset: int = 0,
    limit: int = 5,
) -> list:
    """
    Unified query function supporting category, date, and keyword search filters, with pagination offset.
    """
    session = _get_session()
    try:
        query = session.query(Article)
        
        if target_date:
            next_day = target_date + timedelta(days=1)
            query = query.filter(
                Article.published_at >= datetime.combine(target_date, datetime.min.time()),
                Article.published_at < datetime.combine(next_day, datetime.min.time()),
            )
            
        if category and category.lower() != "all":
            query = query.filter(Article.category.ilike(f"%{category.strip()}%"))
            
        if tag:
            tag_term = tag.strip()
            query = query.filter(
                (Article.title.ilike(f"%{tag_term}%")) |
                (Article.keywords.ilike(f"%{tag_term}%")) |
                (Article.category.ilike(f"%{tag_term}%"))
            )
            
        return query.order_by(Article.published_at.desc()).offset(offset).limit(limit).all()
    finally:
        session.close()


def get_fallback_date(category: str = None, tag: str = None) -> Optional[date]:
    """
    Finds the date of the most recent article, optionally filtered by category or tag.
    """
    session = _get_session()
    try:
        query = session.query(Article)
        if category and category.lower() != "all":
            query = query.filter(Article.category.ilike(f"%{category.strip()}%"))
        if tag:
            tag_term = tag.strip()
            query = query.filter(
                (Article.title.ilike(f"%{tag_term}%")) |
                (Article.keywords.ilike(f"%{tag_term}%")) |
                (Article.category.ilike(f"%{tag_term}%"))
            )
        latest = query.order_by(Article.published_at.desc()).first()
        return latest.published_at.date() if latest and latest.published_at else None
    finally:
        session.close()


def _build_article_embed(article, index: int = None) -> discord.Embed:
    """
    Builds a compact, beautiful Discord embed for a single article.
    Shows: Title, Summary, Credibility badge, Category, and a link button.
    """
    # Determine credibility badge
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

    # Build summary text (truncated to fit Discord limits)
    summary = ""
    if article.clean_content:
        summary = article.clean_content[:300] + "..." if len(article.clean_content) > 300 else article.clean_content
    elif article.raw_content:
        summary = article.raw_content[:300] + "..." if len(article.raw_content) > 300 else article.raw_content
    else:
        summary = "No summary available."

    # Title with optional index number
    title = article.title or "Untitled Article"
    if index is not None:
        title = f"{index}. {title}"
    if len(title) > 253:
        title = title[:250] + "..."

    embed = discord.Embed(
        title=title,
        description=summary,
        color=color,
        url=article.url,
    )

    embed.add_field(name="📊 Credibility", value=f"{badge} ({int(score * 100)}%)", inline=True)
    embed.add_field(name="📂 Category", value=article.category or "General", inline=True)
    embed.add_field(name="📰 Source", value=article.source or "Unknown", inline=True)

    if article.published_at:
        embed.timestamp = article.published_at

    embed.set_footer(text="AI News Monitor • Powered by DistilBERT")

    return embed


def _build_article_view(article) -> discord.ui.View:
    """Creates a View with a 'Read Full Article' button linking to the source."""
    view = discord.ui.View()
    if article.url:
        view.add_item(discord.ui.Button(
            label="📖 Read Full Article",
            url=article.url,
            style=discord.ButtonStyle.link,
        ))
    return view


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

        # 1. Read Full Article Button
        if article_url:
            self.add_item(
                discord.ui.Button(
                    label="📖 Read Full Article",
                    url=article_url,
                    style=discord.ButtonStyle.link,
                )
            )

        # 2. Show More Button
        self.show_more_btn = discord.ui.Button(
            label="Show More ➡️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"show_more:{command_type}:{offset}",
        )
        self.show_more_btn.callback = self.show_more_callback
        self.add_item(self.show_more_btn)

    async def show_more_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()

        # Disable/remove the Show More button on this message to prevent re-clicks
        self.remove_item(self.show_more_btn)
        await interaction.message.edit(view=self)

        limit = 6
        batch = fetch_articles(
            category=self.category,
            target_date=self.search_date,
            tag=self.search_tag,
            offset=self.offset,
            limit=limit,
        )

        if not batch:
            await interaction.followup.send(
                content="📭 No more articles found.", ephemeral=True
            )
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
                view = _build_article_view(article)

            await interaction.followup.send(
                embed=_build_article_embed(article, index=current_index),
                view=view,
            )


# ── Category Select Menu ─────────────────────────────────────────────────────
class CategorySelect(discord.ui.Select):
    """Dropdown menu that lists all available news categories."""

    def __init__(self, categories: list):
        options = []
        for cat in categories:
            options.append(discord.SelectOption(
                label=cat,
                value=cat,
                description=f"View today's {cat} news",
                emoji="📰",
            ))
        super().__init__(
            placeholder="🔍 Select a news category...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        selected_category = self.values[0]
        await interaction.response.defer(thinking=True)

        target_date = date.today()
        test_articles = fetch_articles(category=selected_category, target_date=target_date, limit=1)
        
        fallback_msg = ""
        if not test_articles:
            fallback = get_fallback_date(category=selected_category)
            if fallback:
                target_date = fallback
                fallback_msg = f" (No news today. Showing the most recent articles from **{target_date.strftime('%B %d, %Y')}** instead)"
            else:
                target_date = None

        if not target_date:
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ No News Found",
                    description=f"No articles found for category **{selected_category}**.",
                    color=EMBED_COLOR_ERROR,
                )
            )
            return

        batch = fetch_articles(category=selected_category, target_date=target_date, offset=0, limit=6)

        header = discord.Embed(
            title=f"📂 {selected_category} News",
            description=f"Showing top {min(len(batch), 5)} article(s) for **{selected_category}**{fallback_msg}",
            color=EMBED_COLOR_PRIMARY if not fallback_msg else EMBED_COLOR_WARNING,
        )
        await interaction.followup.send(embed=header)
        
        display_count = min(len(batch), 5)
        for i in range(display_count):
            article = batch[i]
            is_last = (i == display_count - 1)
            has_more = (len(batch) > 5)

            if is_last and has_more:
                view = NewsPaginationView(
                    command_type="category",
                    offset=5,
                    category=selected_category,
                    search_date=target_date,
                    article_url=article.url,
                    start_index=i + 2,
                )
            else:
                view = _build_article_view(article)

            await interaction.followup.send(
                embed=_build_article_embed(article, index=i + 1),
                view=view,
            )


class CategoryView(discord.ui.View):
    """View wrapper for the category dropdown."""

    def __init__(self, categories: list):
        super().__init__(timeout=120)
        self.add_item(CategorySelect(categories))


# ── The Cog ───────────────────────────────────────────────────────────────────
class NewsCommands(commands.Cog):
    """All news-related slash commands."""

    def __init__(self, bot):
        self.bot = bot

    # ── /newsdaily ────────────────────────────────────────────────────────
    @app_commands.command(name="newsdaily", description="📰 Get today's top AI-analyzed news articles")
    async def newsdaily(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        target_date = date.today()
        test_articles = fetch_articles(target_date=target_date, limit=1)
        
        fallback_msg = ""
        if not test_articles:
            fallback = get_fallback_date()
            if fallback:
                target_date = fallback
                fallback_msg = f" (No news found for today. Showing the most recent articles from **{target_date.strftime('%B %d, %Y')}** instead)"
            else:
                target_date = None

        if not target_date:
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ No News Available",
                    description="The database has no articles yet. The pipeline may still be running.",
                    color=EMBED_COLOR_ERROR,
                )
            )
            return

        batch = fetch_articles(target_date=target_date, offset=0, limit=6)

        header_title = f"📰 Daily News Briefing — {target_date.strftime('%B %d, %Y')}"
        if fallback_msg:
            header_title = f"📰 Latest News Briefing"
            
        header = discord.Embed(
            title=header_title,
            description=f"Showing top {min(len(batch), 5)} AI-analyzed article(s){fallback_msg}",
            color=EMBED_COLOR_PRIMARY if not fallback_msg else EMBED_COLOR_WARNING,
        )

        await interaction.followup.send(embed=header)
        
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
                    start_index=i + 2,
                )
            else:
                view = _build_article_view(article)

            await interaction.followup.send(
                embed=_build_article_embed(article, index=i + 1),
                view=view,
            )

    # ── /news [date] [tag] ────────────────────────────────────────────────
    @app_commands.command(name="news", description="🔎 Search news by date or tag/keyword")
    @app_commands.describe(
        date="Date in YYYY-MM-DD format (e.g. 2026-06-04)",
        tag="Search by keyword or tag (e.g. 'technology', 'Modi', 'AI')",
    )
    async def news(
        self,
        interaction: discord.Interaction,
        date: str = None,
        tag: str = None,
    ):
        await interaction.response.defer(thinking=True)

        # Validate: user must provide at least one argument
        if not date and not tag:
            await interaction.followup.send(
                embed=discord.Embed(
                    title="⚠️ Missing Argument",
                    description="Please provide a **date** (e.g. `2026-06-04`) or a **tag** (e.g. `technology`).\n\n"
                                "**Usage:**\n"
                                "`/news date:2026-06-04`\n"
                                "`/news tag:technology`\n"
                                "`/news date:2026-06-04 tag:AI`",
                    color=EMBED_COLOR_WARNING,
                )
            )
            return

        target_date = None
        target_date_str = ""
        if date:
            try:
                target_date = datetime.strptime(date.strip(), "%Y-%m-%d").date()
                target_date_str = target_date.strftime("%B %d, %Y")
            except ValueError:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ Invalid Date Format",
                        description=f"Could not parse `{date}`. Please use the format **YYYY-MM-DD** (e.g. `2026-06-04`).",
                        color=EMBED_COLOR_ERROR,
                    )
                )
                return

        # Fetch page 1 (limit 6)
        batch = fetch_articles(target_date=target_date, tag=tag, offset=0, limit=6)

        if not batch:
            desc_parts = []
            if date:
                desc_parts.append(f"date **{date}**")
            if tag:
                desc_parts.append(f"tag **{tag}**")
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ No News Found",
                    description=f"No articles found for {' and '.join(desc_parts)}.\n\n"
                                "This could mean:\n"
                                "• The date format is incorrect\n"
                                "• No news was ingested for that date\n"
                                "• The tag doesn't match any articles",
                    color=EMBED_COLOR_ERROR,
                )
            )
            return

        # Build header
        title_parts = []
        if target_date_str:
            title_parts.append(target_date_str)
        if tag:
            title_parts.append(f"#{tag}")
        header_title = f"🔎 News Results — {' | '.join(title_parts)}"

        header = discord.Embed(
            title=header_title,
            description=f"Found {min(len(batch), 5)} article(s) shown",
            color=EMBED_COLOR_PRIMARY,
        )
        await interaction.followup.send(embed=header)
        
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
                    start_index=i + 2,
                )
            else:
                view = _build_article_view(article)

            await interaction.followup.send(
                embed=_build_article_embed(article, index=i + 1),
                view=view,
            )

    # ── /newscategories ───────────────────────────────────────────────────
    @app_commands.command(name="newscategories", description="📂 Browse news by category with an interactive dropdown")
    async def newscategories(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        session = _get_session()
        try:
            # Get all distinct categories from the database
            raw_categories = (
                session.query(Article.category)
                .filter(Article.category.isnot(None))
                .distinct()
                .all()
            )
            categories = sorted([cat[0] for cat in raw_categories if cat[0]])

            if not categories:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ No Categories Found",
                        description="The database has no categorized articles yet. Run the intelligence pipeline first.",
                        color=EMBED_COLOR_ERROR,
                    )
                )
                return

            # Build category list for the embed
            cat_list = "\n".join([f"**{i+1}.** {cat}" for i, cat in enumerate(categories)])

            embed = discord.Embed(
                title="📂 News Categories",
                description=f"Select a category from the dropdown below to view today's news.\n\n{cat_list}",
                color=EMBED_COLOR_PRIMARY,
            )
            embed.set_footer(text="Dropdown expires in 2 minutes")

            await interaction.followup.send(embed=embed, view=CategoryView(categories))

        finally:
            session.close()

    # ── /setup_daily (Admin only) ─────────────────────────────────────────
    @app_commands.command(name="setup_daily", description="⚙️ [Admin] Register this channel for the daily news drop")
    @app_commands.describe(category="Optional: only receive news from this category (e.g. 'Sci/Tech')")
    @app_commands.checks.has_permissions(administrator=True)
    async def setup_daily(self, interaction: discord.Interaction, category: str = None):
        await interaction.response.defer(thinking=True)

        server_id = str(interaction.guild.id)
        channel_id = str(interaction.channel.id)

        session = _get_session()
        try:
            # Check if this channel is already subscribed
            existing = session.query(DiscordSubscription).filter(
                DiscordSubscription.channel_id == channel_id,
            ).first()

            if existing:
                # Update the category preference
                existing.category = category
                session.commit()
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="✅ Daily Drop Updated",
                        description=f"This channel is already subscribed. Category updated to **{category or 'All News'}**.",
                        color=EMBED_COLOR_SUCCESS,
                    )
                )
                return

            # Create new subscription
            subscription = DiscordSubscription(
                server_id=server_id,
                channel_id=channel_id,
                category=category,
            )
            session.add(subscription)
            session.commit()

            await interaction.followup.send(
                embed=discord.Embed(
                    title="✅ Daily Drop Enabled!",
                    description=f"This channel will now receive the daily news briefing at **3:00 PM UTC** every day.\n\n"
                                f"**Category filter:** {category or 'All News'}\n"
                                f"**Channel:** <#{channel_id}>",
                    color=EMBED_COLOR_SUCCESS,
                )
            )

        except Exception as e:
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ Error",
                    description=f"Failed to set up daily drop: {str(e)}",
                    color=EMBED_COLOR_ERROR,
                )
            )
        finally:
            session.close()

    # Error handler for missing admin permissions
    @setup_daily.error
    async def setup_daily_error(self, interaction: discord.Interaction, error):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="🔒 Permission Denied",
                    description="Only server administrators can set up the daily news drop.",
                    color=EMBED_COLOR_ERROR,
                ),
                ephemeral=True,
            )

    # ── /remove_daily (Admin only) ────────────────────────────────────────
    @app_commands.command(name="remove_daily", description="⚙️ [Admin] Unregister this channel from the daily news drop")
    @app_commands.checks.has_permissions(administrator=True)
    async def remove_daily(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)

        channel_id = str(interaction.channel.id)

        session = _get_session()
        try:
            existing = session.query(DiscordSubscription).filter(
                DiscordSubscription.channel_id == channel_id,
            ).first()

            if not existing:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="⚠️ Not Subscribed",
                        description="This channel is not currently subscribed to the daily news drop.",
                        color=EMBED_COLOR_WARNING,
                    )
                )
                return

            session.delete(existing)
            session.commit()

            await interaction.followup.send(
                embed=discord.Embed(
                    title="✅ Daily Drop Removed",
                    description="This channel will no longer receive the daily news briefing.",
                    color=EMBED_COLOR_SUCCESS,
                )
            )
        finally:
            session.close()

    @remove_daily.error
    async def remove_daily_error(self, interaction: discord.Interaction, error):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="🔒 Permission Denied",
                    description="Only server administrators can remove the daily news drop.",
                    color=EMBED_COLOR_ERROR,
                ),
                ephemeral=True,
            )


# ── Cog Setup ─────────────────────────────────────────────────────────────────
async def setup(bot):
    await bot.add_cog(NewsCommands(bot))
