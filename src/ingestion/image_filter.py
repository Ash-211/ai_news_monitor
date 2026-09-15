"""
Stage 3: Image Novelty Filter (Reverse Image Search Pre-filter)

Multi-layered heuristic approach to detect stock photos, placeholders,
and recycled images BEFORE they reach the Deepfake API.

Layers:
  1. URL-based heuristics  — instant, zero cost
  2. HTTP header analysis   — one HEAD request per image, zero cost
  3. Pluggable API stub     — ready for TinEye / SerpAPI / Google Vision

Usage:
    from src.ingestion.image_filter import check_image_novelty
    result = check_image_novelty("https://example.com/photo.jpg")
    # {"is_novel": True/False, "reason": "...", "filter_stage": "..."}
"""

import re
import logging
import requests

logger = logging.getLogger(__name__)

# ── Layer 1: Known Stock Photo / Placeholder Domains ─────────────────────────

STOCK_PHOTO_DOMAINS = {
    'gettyimages.com', 'istockphoto.com', 'shutterstock.com',
    'depositphotos.com', 'dreamstime.com', 'alamy.com',
    'stockphoto.com', '123rf.com', 'bigstockphoto.com',
    'adobestock.com', 'stock.adobe.com',
    # Free stock (still indicates non-novel / generic imagery)
    'unsplash.com', 'pexels.com', 'pixabay.com',
    'freepik.com', 'rawpixel.com',
}

# URL path fragments that indicate logos, icons, placeholders, or tracking pixels
PLACEHOLDER_PATTERNS = [
    r'placeholder', r'default[-_]?image', r'no[-_]?image',
    r'logo', r'avatar', r'icon', r'favicon',
    r'1x1', r'spacer', r'blank', r'pixel\.gif',
    r'transparent\.png', r'loading',
    r'wp-content/themes/.*/images/',   # WordPress theme assets
    r'gravatar\.com',                  # Gravatar profile pics
]

_placeholder_re = re.compile('|'.join(PLACEHOLDER_PATTERNS), re.IGNORECASE)


def _check_url_heuristics(image_url: str) -> dict | None:
    """
    Layer 1: Instant URL-based filtering. Zero network calls.
    Returns a rejection dict if the URL matches a stock/placeholder pattern,
    or None if the URL looks legitimate.
    """
    url_lower = image_url.lower()

    # Check stock photo domains
    for domain in STOCK_PHOTO_DOMAINS:
        if domain in url_lower:
            return {
                "is_novel": False,
                "reason": f"Image hosted on known stock-photo domain: {domain}",
                "filter_stage": "url_heuristic",
            }

    # Check placeholder / logo patterns in URL path
    if _placeholder_re.search(url_lower):
        return {
            "is_novel": False,
            "reason": f"URL matches placeholder/logo pattern",
            "filter_stage": "url_heuristic",
        }

    # Reject data URIs (inline base64) — not a real hosted image
    if url_lower.startswith('data:'):
        return {
            "is_novel": False,
            "reason": "Data URI (inline image), not a hosted photo",
            "filter_stage": "url_heuristic",
        }

    return None  # Passed layer 1


# ── Layer 2: HTTP Header Analysis ────────────────────────────────────────────

# Minimum size in bytes to consider an image "real" editorial content.
# Logos and tracking pixels are typically < 10 KB.
MIN_IMAGE_SIZE_BYTES = 15_000  # 15 KB

def _check_http_headers(image_url: str) -> dict | None:
    """
    Layer 2: Lightweight HEAD request to inspect Content-Length
    and Content-Type without downloading the full image.
    Returns a rejection dict or None if the image passes.
    """
    try:
        resp = requests.head(image_url, timeout=4, allow_redirects=True,
                             headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

        # If the server blocks HEAD requests or returns an error, pass through
        if resp.status_code != 200:
            return None

        # Must be a valid image content type
        content_type = resp.headers.get('Content-Type', '')
        if content_type and 'image' not in content_type:
            return {
                "is_novel": False,
                "reason": f"Not an image (Content-Type: {content_type})",
                "filter_stage": "http_header",
            }

        # Reject tiny images (likely tracking pixels, icons, or logos)
        content_length = resp.headers.get('Content-Length')
        if content_length:
            try:
                content_length_bytes = int(content_length)
            except (TypeError, ValueError):
                logger.debug(f"Invalid Content-Length header for {image_url}: {content_length!r}")
            else:
                if content_length_bytes < MIN_IMAGE_SIZE_BYTES:
                    size_kb = content_length_bytes / 1024
                    return {
                        "is_novel": False,
                        "reason": f"Image too small ({size_kb:.1f} KB) — likely a logo or icon",
                        "filter_stage": "http_header",
                    }

    except requests.RequestException as e:
        # If we can't reach the image, don't block ingestion — just log it
        logger.debug(f"HTTP HEAD failed for {image_url}: {e}")
        return None

    return None  # Passed layer 2


# ── Layer 3: Pluggable Reverse Image Search API ─────────────────────────────

def _check_reverse_image_search(image_url: str) -> dict | None:
    """
    Layer 3: Pluggable reverse image search.

    Currently a stub. When a real provider is wired up (TinEye, SerpAPI
    Google Lens, Google Cloud Vision Web Detection), this function will
    query the API and return is_novel=False for images that pre-date today.

    To enable a provider, set the REVERSE_IMAGE_PROVIDER env var:
      - "tineye"  (requires TINEYE_API_KEY)
      - "serpapi"  (requires SERPAPI_KEY)

    For now, this always passes (returns None), so the pipeline relies
    on Layers 1 & 2 for free-tier filtering.
    """
    # ── Future implementation placeholder ──────────────────────────
    # import os
    # provider = os.getenv('REVERSE_IMAGE_PROVIDER', '').lower()
    #
    # if provider == 'tineye':
    #     return _query_tineye(image_url)
    # elif provider == 'serpapi':
    #     return _query_serpapi(image_url)

    logger.debug(f"Reverse image search: no provider configured — skipping for {image_url}")
    return None  # Pass through — no API configured


# ── Public Interface ─────────────────────────────────────────────────────────

def check_image_novelty(image_url: str) -> dict:
    """
    Run all novelty filter layers in order (cheapest first).
    Returns as soon as any layer rejects the image.

    Returns:
        {
            "is_novel": bool,
            "reason": str,
            "filter_stage": str   # "url_heuristic" | "http_header" | "reverse_search" | "passed"
        }
    """
    if not image_url:
        return {"is_novel": False, "reason": "No image URL provided", "filter_stage": "none"}

    # Layer 1: URL heuristics (instant, free)
    result = _check_url_heuristics(image_url)
    if result:
        logger.info(f"  [Filter L1] DISCARDED: {result['reason']}  ->  {image_url[:80]}")
        return result

    # Layer 2: HTTP header inspection (one HEAD request, free)
    result = _check_http_headers(image_url)
    if result:
        logger.info(f"  [Filter L2] DISCARDED: {result['reason']}  ->  {image_url[:80]}")
        return result

    # Layer 3: Reverse image search API (pluggable, currently stub)
    result = _check_reverse_image_search(image_url)
    if result:
        logger.info(f"  [Filter L3] DISCARDED: {result['reason']}  ->  {image_url[:80]}")
        return result

    # All layers passed — image is considered novel
    logger.info(f"  [Filter] PASSED all layers  ->  {image_url[:80]}")
    return {"is_novel": True, "reason": "Passed all novelty filters", "filter_stage": "passed"}


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRE-FILTERS (Run before Sightengine API to minimize paid calls)
# ══════════════════════════════════════════════════════════════════════════════

# ── Filter 5: Trusted News Source Bypass ─────────────────────────────────────

TRUSTED_NEWS_SOURCES = {
    # International wire services & broadcasters
    'bbc', 'bbc news', 'reuters', 'associated press', 'ap news',
    'cnn', 'al jazeera', 'abc news', 'nbc news', 'cbs news',
    'the guardian', 'new york times', 'washington post',
    'the economist', 'financial times', 'bloomberg',
    'wall street journal',
    # Indian outlets
    'ndtv', 'the hindu', 'times of india', 'indian express',
    'hindustan times', 'the wire', 'scroll.in',
    # Tech / Science
    'techcrunch', 'nature', 'science', 'wired', 'ars technica',
    'the verge', 'mit technology review',
}


def check_trusted_source(source: str) -> dict | None:
    """
    Filter 5: Trusted editorial source bypass.
    Tier-1 news outlets have editorial review processes and don't
    publish AI-generated images as news photos. Skip deepfake API.
    Returns a verdict dict if trusted, None otherwise.
    """
    if not source:
        return None
    source_lower = source.lower().strip()
    for trusted in TRUSTED_NEWS_SOURCES:
        if trusted in source_lower:
            logger.info(f"  [Filter 5] Trusted source bypass: {source}")
            return {
                "skip_api": True,
                "image_status": "real",
                "deepfake_score": 0.02,
                "reason": f"Trusted editorial source: {source}",
                "filter_name": "trusted_source",
            }
    return None


# ── Filter 6: EXIF Camera Metadata ──────────────────────────────────────────

def check_exif_authenticity(image_path: str) -> dict | None:
    """
    Filter 6: Genuine camera EXIF verification.
    Real photographs from physical cameras contain rich EXIF metadata
    (Make, Model, DateTimeOriginal, GPS coordinates). AI-generated images
    have zero EXIF data. If we find genuine camera signatures, skip API.
    Returns a verdict dict if genuine camera EXIF found, None otherwise.
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        with PILImage.open(image_path) as img:
            exif_data = img._getexif()

        if not exif_data:
            return None  # No EXIF — inconclusive, continue to next filter

        decoded = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            decoded[tag_name] = value

        has_camera_make = 'Make' in decoded
        has_camera_model = 'Model' in decoded
        has_date = any(f in decoded for f in ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime'])
        has_gps = 'GPSInfo' in decoded

        # Strong signal: at least 2 of 4 camera indicators = real camera
        trust_signals = sum([has_camera_make, has_camera_model, has_date, has_gps])

        if trust_signals >= 2:
            camera_info = f"{decoded.get('Make', '?')} {decoded.get('Model', '')}".strip()
            logger.info(f"  [Filter 6] Camera EXIF: {camera_info} ({trust_signals}/4 signals)")
            return {
                "skip_api": True,
                "image_status": "real",
                "deepfake_score": 0.03,
                "reason": f"Genuine camera EXIF: {camera_info} ({trust_signals}/4 trust signals)",
                "filter_name": "exif_camera",
            }

    except Exception as e:
        logger.debug(f"EXIF check failed: {e}")

    return None


# ── Filter 7: AI Dimension Fingerprinting ────────────────────────────────────

# Known AI generator output resolutions (width, height)
AI_DIMENSIONS = {
    # Square formats (common across all generators)
    (512, 512), (768, 768), (1024, 1024), (2048, 2048),
    # Stable Diffusion 1.5
    (768, 512), (512, 768),
    # SDXL
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1536, 640), (640, 1536),
    # DALL-E 3
    (1792, 1024), (1024, 1792),
    # Midjourney v5/v6
    (1024, 1536), (1536, 1024),
    (1456, 816), (816, 1456),
}

# Minimum megapixels for a real editorial photo (DSLRs are 12-50+ MP)
DSLR_MIN_MEGAPIXELS = 3.0


def check_ai_dimensions(image_path: str) -> dict | None:
    """
    Filter 7: AI dimension fingerprinting.
    AI generators produce images at exact, predictable resolutions.
    Real cameras produce high-megapixel images at non-standard sizes.
    - If dimensions match a known AI size: let it through to Sightengine (suspicious).
    - If image is high-res (>3 MP) and NOT an AI size: likely real, skip API.
    Returns a verdict dict if dimensions conclusively indicate a real photo, None otherwise.
    """
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            w, h = img.size

        megapixels = (w * h) / 1_000_000

        # Exact AI resolution match → suspicious, DO NOT skip API
        if (w, h) in AI_DIMENSIONS:
            logger.info(f"  [Filter 7] AI dimensions detected: {w}x{h} — will send to Sightengine")
            return None  # Continue to Sightengine

        # High-resolution + not AI size → very likely a real camera photo
        if megapixels >= DSLR_MIN_MEGAPIXELS:
            logger.info(f"  [Filter 7] High-res photo: {w}x{h} ({megapixels:.1f} MP) — skipping API")
            return {
                "skip_api": True,
                "image_status": "real",
                "deepfake_score": 0.05,
                "reason": f"High-resolution image ({w}x{h}, {megapixels:.1f} MP) — consistent with camera",
                "filter_name": "dimension_analysis",
            }

    except Exception as e:
        logger.debug(f"Dimension check failed: {e}")

    return None
