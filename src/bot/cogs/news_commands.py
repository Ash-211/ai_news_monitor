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
def _get_session():
    """Create a fresh database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


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
    # Discord embed title limit is 256 characters
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


def _build_news_embeds(articles: list) -> list:
    """Build a list of embeds for multiple articles."""
    embeds = []
    for i, article in enumerate(articles, 1):
        embeds.append(_build_article_embed(article, index=i))
    return embeds


# ── Category Select Menu ─────────────────────────────────────────────────────
class CategorySelect(discord.ui.Select):
    """Dropdown menu that lists all available news categories."""

    def __init__(self, categories: list):
        options = []
        for i, cat in enumerate(categories):
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

        session = _get_session()
        try:
            today = date.today()
            tomorrow = today + timedelta(days=1)

            articles = (
                session.query(Article)
                .filter(
                    Article.category == selected_category,
                    Article.published_at >= datetime.combine(today, datetime.min.time()),
                    Article.published_at < datetime.combine(tomorrow, datetime.min.time()),
                )
                .order_by(Article.credibility_score.desc())
                .limit(MAX_ARTICLES_PER_EMBED)
                .all()
            )

            if not articles:
                # If no articles today, get the latest ones for that category
                articles = (
                    session.query(Article)
                    .filter(Article.category == selected_category)
                    .order_by(Article.published_at.desc())
                    .limit(MAX_ARTICLES_PER_EMBED)
                    .all()
                )

            if not articles:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ No News Found",
                        description=f"No articles found for category **{selected_category}**.",
                        color=EMBED_COLOR_ERROR,
                    )
                )
                return

            header = discord.Embed(
                title=f"📂 {selected_category} News",
                description=f"Showing top {len(articles)} article(s) for **{selected_category}**",
                color=EMBED_COLOR_PRIMARY,
            )
            embeds = [header] + _build_news_embeds(articles)

            # Send each embed with its button separately (Discord limits 10 embeds per message)
            await interaction.followup.send(embed=embeds[0])
            for i, article in enumerate(articles):
                await interaction.followup.send(
                    embed=embeds[i + 1],
                    view=_build_article_view(article),
                )

        finally:
            session.close()


class CategoryView(discord.ui.View):
    """View wrapper for the category dropdown."""

    def __init__(self, categories: list):
        super().__init__(timeout=120)  # 2 minute timeout
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

        session = _get_session()
        try:
            today = date.today()
            tomorrow = today + timedelta(days=1)

            articles = (
                session.query(Article)
                .filter(
                    Article.published_at >= datetime.combine(today, datetime.min.time()),
                    Article.published_at < datetime.combine(tomorrow, datetime.min.time()),
                )
                .order_by(Article.credibility_score.desc())
                .limit(MAX_ARTICLES_PER_EMBED)
                .all()
            )

            if not articles:
                # Fallback: show most recent articles if nothing from today
                articles = (
                    session.query(Article)
                    .order_by(Article.published_at.desc())
                    .limit(MAX_ARTICLES_PER_EMBED)
                    .all()
                )
                if not articles:
                    await interaction.followup.send(
                        embed=discord.Embed(
                            title="❌ No News Available",
                            description="The database has no articles yet. The pipeline may still be running.",
                            color=EMBED_COLOR_ERROR,
                        )
                    )
                    return

                header = discord.Embed(
                    title=f"📰 Latest News Briefing",
                    description=f"No news found for today ({today.strftime('%B %d, %Y')}). Showing the most recent articles instead.",
                    color=EMBED_COLOR_WARNING,
                )
            else:
                header = discord.Embed(
                    title=f"📰 Daily News Briefing — {today.strftime('%B %d, %Y')}",
                    description=f"Top {len(articles)} AI-analyzed article(s) for today",
                    color=EMBED_COLOR_PRIMARY,
                )

            await interaction.followup.send(embed=header)
            for i, article in enumerate(articles, 1):
                await interaction.followup.send(
                    embed=_build_article_embed(article, index=i),
                    view=_build_article_view(article),
                )

        finally:
            session.close()

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

        session = _get_session()
        try:
            query = session.query(Article)

            # Filter by date if provided
            target_date_str = ""
            if date:
                try:
                    parsed_date = datetime.strptime(date.strip(), "%Y-%m-%d").date()
                    next_day = parsed_date + timedelta(days=1)
                    query = query.filter(
                        Article.published_at >= datetime.combine(parsed_date, datetime.min.time()),
                        Article.published_at < datetime.combine(next_day, datetime.min.time()),
                    )
                    target_date_str = parsed_date.strftime("%B %d, %Y")
                except ValueError:
                    await interaction.followup.send(
                        embed=discord.Embed(
                            title="❌ Invalid Date Format",
                            description=f"Could not parse `{date}`. Please use the format **YYYY-MM-DD** (e.g. `2026-06-04`).",
                            color=EMBED_COLOR_ERROR,
                        )
                    )
                    return

            # Filter by tag if provided (search in title, keywords, and category)
            if tag:
                tag_term = tag.strip()
                query = query.filter(
                    (Article.title.ilike(f"%{tag_term}%")) |
                    (Article.keywords.ilike(f"%{tag_term}%")) |
                    (Article.category.ilike(f"%{tag_term}%"))
                )

            articles = (
                query.order_by(Article.published_at.desc())
                .limit(MAX_ARTICLES_PER_EMBED)
                .all()
            )

            if not articles:
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
                description=f"Found {len(articles)} article(s)",
                color=EMBED_COLOR_PRIMARY,
            )
            await interaction.followup.send(embed=header)
            for i, article in enumerate(articles, 1):
                await interaction.followup.send(
                    embed=_build_article_embed(article, index=i),
                    view=_build_article_view(article),
                )

        finally:
            session.close()

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
