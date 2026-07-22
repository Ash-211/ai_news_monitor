"""
verify_commands.py — Slash-command cog for on-demand URL verification.

Commands:
    /verify url:<URL>  – Scrape, analyse, and fact-check a news article URL.
"""

from __future__ import annotations

import logging
from datetime import datetime

import discord
from discord import app_commands
from discord.ext import commands

from src.intelligence.url_verifier import verify_url

logger = logging.getLogger("ai_news_bot.verify_commands")

# ── Colour palette (mirrors news_commands.py) ──────────────────────────────
COLOR_CREDIBLE  = discord.Colour(0x2ECC71)   # Green  – score >= 0.6
COLOR_UNCERTAIN = discord.Colour(0xF39C12)   # Orange – 0.4 <= score < 0.6
COLOR_FLAGGED   = discord.Colour(0xE74C3C)   # Red    – score < 0.4
COLOR_DEFAULT   = discord.Colour(0x3498DB)   # Blue   – score is None
COLOR_ERROR     = discord.Colour(0x95A5A6)   # Grey   – error state


def _credibility_colour(score: float | None) -> discord.Colour:
    """Return an embed colour based on the credibility score."""
    if score is None:
        return COLOR_DEFAULT
    if score >= 0.6:
        return COLOR_CREDIBLE
    if score >= 0.4:
        return COLOR_UNCERTAIN
    return COLOR_FLAGGED


def _credibility_badge(score: float | None) -> str:
    """Human-friendly label for the credibility score."""
    if score is None:
        return "N/A"
    if score >= 0.6:
        return f"✅ Credible ({score:.0%})"
    if score >= 0.4:
        return f"⚠️ Uncertain ({score:.0%})"
    return f"🚩 Flagged ({score:.0%})"


def _verdict_label(is_fake: bool | None) -> str:
    """Turn the boolean is_fake into a readable verdict."""
    if is_fake is None:
        return "⬜ Unknown"
    return "🚩 Potentially Misleading" if is_fake else "✅ Authentic"


def _build_verify_embed(data: dict) -> discord.Embed:
    """
    Build a rich Discord embed from the verify_url() result dict.
    """
    title = data.get("title") or "Untitled Article"
    url = data.get("url") or ""
    credibility = data.get("credibility_score")

    embed = discord.Embed(
        title=f"🔍 {title}",
        url=url,
        colour=_credibility_colour(credibility),
        timestamp=datetime.utcnow(),
    )

    # ── Verdict & Credibility (top row) ───────────────────────────────
    embed.add_field(
        name="🛡️ Credibility",
        value=_credibility_badge(credibility),
        inline=True,
    )
    embed.add_field(
        name="📋 Verdict",
        value=_verdict_label(data.get("is_fake")),
        inline=True,
    )

    # ── Category ──────────────────────────────────────────────────────
    category = data.get("category") or "N/A"
    confidence = data.get("category_confidence")
    cat_display = category
    if confidence is not None:
        cat_display += f" ({confidence:.0%})"
    embed.add_field(name="📂 Category", value=cat_display, inline=True)

    # ── Keywords ──────────────────────────────────────────────────────
    keywords = data.get("keywords") or []
    if keywords:
        kw_text = ", ".join(keywords[:8])
        if len(kw_text) > 200:
            kw_text = kw_text[:200] + "…"
        embed.add_field(name="🔑 Keywords", value=kw_text, inline=False)

    # ── AI Explanation ────────────────────────────────────────────────
    explanation = data.get("explanation") or ""
    if explanation:
        if len(explanation) > 300:
            explanation = explanation[:300] + "…"
        embed.add_field(name="🤖 AI Analysis", value=explanation, inline=False)

    # ── Fact-Check ────────────────────────────────────────────────────
    fact_check = data.get("fact_check")
    if fact_check and isinstance(fact_check, dict):
        v_score = fact_check.get("verification_score", 0.5)
        cross_ref = fact_check.get("cross_reference", {})
        google_fc = fact_check.get("fact_check", {})

        fc_parts = [f"**Verification Score:** {v_score:.0%}"]

        # NewsAPI cross-reference
        total = cross_ref.get("total_results", 0)
        sources = cross_ref.get("matching_sources", [])
        if cross_ref.get("status") == "ok":
            fc_parts.append(f"**Other Outlets Reporting:** {total}")
            if sources:
                fc_parts.append(f"**Sources:** {', '.join(sources[:3])}")

        # Google Fact Check
        claims = google_fc.get("claims_found", 0)
        ratings = google_fc.get("ratings", [])
        if claims > 0:
            fc_parts.append(f"**Fact-Check Claims:** {claims}")
            if ratings:
                fc_parts.append(f"**Ratings:** {', '.join(ratings[:3])}")

        embed.add_field(
            name="🔎 External Fact-Check",
            value="\n".join(fc_parts),
            inline=False,
        )

    # ── Footer ────────────────────────────────────────────────────────
    source = data.get("source") or "Unknown"
    embed.set_footer(text=f"Source: {source} • AI News Pipeline — /verify")

    return embed


def _build_verify_view(url: str) -> discord.ui.View:
    """Return a View with a URL button to the original article."""
    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            label="Read Full Article",
            style=discord.ButtonStyle.link,
            url=url,
            emoji="📰",
        )
    )
    return view


# ════════════════════════════════════════════════════════════════════════════
#  Cog
# ════════════════════════════════════════════════════════════════════════════

class VerifyCommands(commands.Cog):
    """Slash command for on-demand URL credibility verification."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="verify",
        description="Verify a news article URL — check credibility, category, and fact-check results.",
    )
    @app_commands.describe(
        url="The full URL of the news article to verify.",
    )
    async def verify(self, interaction: discord.Interaction, url: str) -> None:
        """Scrape, analyse, and fact-check a news article from its URL."""

        # Defer immediately — the pipeline takes a few seconds
        await interaction.response.defer()

        try:
            # Run the full pipeline
            data = verify_url(url)
        except Exception as e:
            logger.exception("Unexpected error in /verify: %s", e)
            error_embed = discord.Embed(
                title="❌ Verification Failed",
                description=f"An unexpected error occurred:\n```{e}```",
                colour=COLOR_ERROR,
            )
            await interaction.followup.send(embed=error_embed)
            return

        # Handle pipeline-level errors (invalid URL, scrape failure, etc.)
        if data.get("error"):
            error_embed = discord.Embed(
                title="❌ Could Not Verify",
                description=data["error"],
                colour=COLOR_ERROR,
            )
            error_embed.set_footer(text="AI News Pipeline — /verify")
            await interaction.followup.send(embed=error_embed)
            return

        # Build and send the result embed
        embed = _build_verify_embed(data)
        view = _build_verify_view(url)
        await interaction.followup.send(embed=embed, view=view)


# ── Cog setup hook (required by discord.py) ─────────────────────────────────

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(VerifyCommands(bot))
    logger.info("VerifyCommands cog loaded.")
