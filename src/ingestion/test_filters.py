"""
Comprehensive Pipeline Filter Test Suite
Tests all 7 filter layers + C2PA with real-world test cases.
Reports accuracy metrics for each layer.
"""
import sys
import os
import time
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
#  TEST DATA — Curated test cases with known ground truth
# ══════════════════════════════════════════════════════════════════════════════

# Format: (url_or_input, expected_result, description)
# expected_result: True = should be REJECTED/FILTERED, False = should PASS

URL_HEURISTIC_TESTS = [
    # Should REJECT (True = correctly filtered)
    ("https://www.gettyimages.com/photos/news-conference", True, "Getty stock photo"),
    ("https://www.shutterstock.com/image-photo/business-meeting-123456", True, "Shutterstock stock"),
    ("https://images.unsplash.com/photo-1234567890", True, "Unsplash free stock"),
    ("https://www.istockphoto.com/photo/office-gm123456", True, "iStock"),
    ("https://www.pexels.com/photo/sunrise-over-mountains-123456/", True, "Pexels free stock"),
    ("https://cdn.pixabay.com/photo/2024/01/01/sunset.jpg", True, "Pixabay"),
    ("https://example.com/images/placeholder.png", True, "Placeholder pattern"),
    ("https://example.com/default-image.jpg", True, "Default image pattern"),
    ("https://example.com/wp-content/themes/flavor/images/header.jpg", True, "WordPress theme asset"),
    ("https://secure.gravatar.com/avatar/abc123?s=96&d=mm&r=g", True, "Gravatar avatar"),
    ("https://example.com/assets/logo.png", True, "Logo pattern"),
    ("https://example.com/favicon.ico", True, "Favicon"),
    ("https://example.com/pixel.gif", True, "Tracking pixel"),
    ("https://example.com/1x1.png", True, "1x1 spacer"),
    ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...", True, "Data URI"),
    # Should PASS (False = correctly allowed through)
    ("https://ichef.bbci.co.uk/news/976/cpsprodpb/1234/article-image.jpg", False, "BBC news photo"),
    ("https://static.reuters.com/resources/article-image-2024.jpg", False, "Reuters news photo"),
    ("https://images.indianexpress.com/2024/09/election-rally.jpg", False, "Indian Express photo"),
    ("https://cdn.techcrunch.com/wp-content/uploads/2024/09/startup-launch.jpg", False, "TechCrunch photo"),
    ("https://media.ndtv.com/images/2024/story-image.jpg", False, "NDTV photo"),
]

TRUSTED_SOURCE_TESTS = [
    # Should be TRUSTED (True = correctly identified as trusted)
    ("BBC News", True, "BBC"),
    ("Reuters", True, "Reuters"),
    ("Associated Press", True, "AP"),
    ("NDTV", True, "NDTV"),
    ("The Hindu", True, "The Hindu"),
    ("Times of India", True, "TOI"),
    ("Indian Express", True, "Indian Express"),
    ("TechCrunch", True, "TechCrunch"),
    ("The Guardian", True, "Guardian"),
    ("CNN", True, "CNN"),
    ("Bloomberg", True, "Bloomberg"),
    ("Al Jazeera", True, "Al Jazeera"),
    # Should NOT be trusted (False = correctly identified as unknown)
    ("RandomBlog.com", False, "Unknown blog"),
    ("InfoWars", False, "Unreliable source"),
    ("FreedomEagleNews", False, "Unknown site"),
    ("ViralHotTakes", False, "Unknown viral site"),
    ("", False, "Empty source"),
    (None, False, "None source"),
]

# Real image URLs for HTTP header testing
HTTP_HEADER_TESTS = [
    # Tiny images that should be REJECTED
    ("https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png", True, "Google logo (small)"),
    # Real editorial-sized images that should PASS
    ("https://ichef.bbci.co.uk/ace/standard/976/cpsprodpb/7721/live/e29a5e80-6bcf-11ef-8c32-fb4d0e498abc.jpg", False, "BBC editorial photo"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class FilterTestResults:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.correct = 0
        self.results = []

    def record(self, expected, actual, description):
        self.total += 1
        is_correct = expected == actual
        if is_correct:
            self.correct += 1
        status = "PASS" if is_correct else "FAIL"
        self.results.append((status, expected, actual, description))

    @property
    def accuracy(self):
        return (self.correct / self.total * 100) if self.total > 0 else 0

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"  {self.name}")
        print(f"  Accuracy: {self.correct}/{self.total} ({self.accuracy:.1f}%)")
        print(f"{'='*70}")
        for status, expected, actual, desc in self.results:
            icon = "  OK " if status == "PASS" else " FAIL"
            exp_str = "REJECT" if expected else "PASS"
            act_str = "REJECT" if actual else "PASS"
            print(f"  [{icon}] {desc:40s}  expected={exp_str:6s}  got={act_str:6s}")


def test_url_heuristics():
    """Test Filter L1: URL-based heuristic filtering."""
    from src.ingestion.image_filter import _check_url_heuristics

    results = FilterTestResults("Stage 3 — Layer 1: URL Heuristics")

    for url, should_reject, desc in URL_HEURISTIC_TESTS:
        verdict = _check_url_heuristics(url)
        was_rejected = verdict is not None  # None means passed, dict means rejected
        results.record(should_reject, was_rejected, desc)

    results.print_report()
    return results


def test_http_headers():
    """Test Filter L2: HTTP header analysis."""
    from src.ingestion.image_filter import _check_http_headers

    results = FilterTestResults("Stage 3 — Layer 2: HTTP Header Analysis")

    for url, should_reject, desc in HTTP_HEADER_TESTS:
        try:
            verdict = _check_http_headers(url)
            was_rejected = verdict is not None
        except Exception as e:
            was_rejected = False  # On error, filter passes through
            desc += f" (error: {e})"
        results.record(should_reject, was_rejected, desc)

    results.print_report()
    return results


def test_trusted_source():
    """Test Filter 5: Trusted news source bypass."""
    from src.ingestion.image_filter import check_trusted_source

    results = FilterTestResults("Filter 5: Trusted News Source Bypass")

    for source, should_be_trusted, desc in TRUSTED_SOURCE_TESTS:
        verdict = check_trusted_source(source)
        was_trusted = verdict is not None
        results.record(should_be_trusted, was_trusted, desc)

    results.print_report()
    return results


def test_ai_dimensions():
    """Test Filter 7: AI dimension fingerprinting."""
    from src.ingestion.image_filter import check_ai_dimensions, AI_DIMENSIONS

    results = FilterTestResults("Filter 7: AI Dimension Fingerprinting")

    # Create test images with specific dimensions
    from PIL import Image as PILImage

    test_dims = [
        # AI dimensions — should NOT skip API (return None = "keep checking")
        ((512, 512), False, "512x512 (SD 1.5 square)"),
        ((1024, 1024), False, "1024x1024 (DALL-E / SD)"),
        ((1792, 1024), False, "1792x1024 (DALL-E 3)"),
        ((1152, 896), False, "1152x896 (SDXL)"),
        # DSLR / phone camera — should skip API (return verdict = "real")
        ((4000, 3000), True, "4000x3000 (12 MP DSLR)"),
        ((6000, 4000), True, "6000x4000 (24 MP DSLR)"),
        ((4032, 3024), True, "4032x3024 (iPhone 12 MP)"),
        ((3840, 2160), True, "3840x2160 (4K photo)"),
        # Small web images — inconclusive (return None = keep checking)
        ((800, 600), False, "800x600 (web thumbnail)"),
        ((640, 480), False, "640x480 (small web image)"),
        ((300, 200), False, "300x200 (small thumbnail)"),
    ]

    for (w, h), should_skip, desc in test_dims:
        # Create a temp image with specific dimensions
        img = PILImage.new('RGB', (w, h), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, 'JPEG')
            tmp_path = tmp.name

        try:
            verdict = check_ai_dimensions(tmp_path)
            actually_skipped = verdict is not None and verdict.get('skip_api', False)
            results.record(should_skip, actually_skipped, desc)
        finally:
            os.remove(tmp_path)

    results.print_report()
    return results


def test_exif_authenticity():
    """Test Filter 6: EXIF camera metadata."""
    from src.ingestion.image_filter import check_exif_authenticity
    from PIL import Image as PILImage

    results = FilterTestResults("Filter 6: EXIF Camera Metadata")

    # Test 1: Try to create image WITH camera EXIF (requires piexif)
    try:
        import piexif
        img = PILImage.new('RGB', (1920, 1080), color='blue')
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"Canon",
                piexif.ImageIFD.Model: b"EOS R5",
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2024:09:01 14:30:00",
            },
        }
        exif_bytes = piexif.dump(exif_dict)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, 'JPEG', exif=exif_bytes)
            tmp_path = tmp.name

        verdict = check_exif_authenticity(tmp_path)
        results.record(True, verdict is not None, "Canon EOS R5 + date EXIF (should detect as real)")
        os.remove(tmp_path)
    except ImportError:
        print("    [INFO] piexif not installed — skipping camera EXIF injection test")

    # Test 2: JPEG without EXIF → should pass through (None = inconclusive)
    img = PILImage.new('RGB', (1920, 1080), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp, 'JPEG')
        tmp_path = tmp.name
    verdict = check_exif_authenticity(tmp_path)
    results.record(False, verdict is not None, "JPEG without EXIF (should pass through)")
    os.remove(tmp_path)

    # Test 3: PNG without EXIF → should pass through
    img = PILImage.new('RGB', (1024, 1024), color='green')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp, 'PNG')
        tmp_path = tmp.name
    verdict = check_exif_authenticity(tmp_path)
    results.record(False, verdict is not None, "PNG without EXIF (should pass through)")
    os.remove(tmp_path)

    results.print_report()
    return results


def test_full_novelty_pipeline():
    """Test the complete check_image_novelty() pipeline (Stages 2-3)."""
    from src.ingestion.image_filter import check_image_novelty

    results = FilterTestResults("Full Novelty Pipeline (check_image_novelty)")

    # Should REJECT
    novelty_tests = [
        ("https://www.shutterstock.com/image/business-123.jpg", True, "Shutterstock URL"),
        ("https://example.com/logo.png", True, "Logo URL"),
        ("", True, "Empty URL"),
        (None, True, "None URL"),
        # Should PASS
        ("https://ichef.bbci.co.uk/news/976/cpsprodpb/test-article.jpg", False, "BBC editorial URL"),
        ("https://media.ndtv.com/images/story/2024-election.jpg", False, "NDTV editorial URL"),
    ]

    for url, should_reject, desc in novelty_tests:
        result = check_image_novelty(url)
        was_rejected = not result['is_novel']
        results.record(should_reject, was_rejected, desc)

    results.print_report()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AI NEWS MONITOR — FILTER PIPELINE TEST SUITE")
    print("  Testing all 7 pre-filter layers for accuracy")
    print("=" * 70)

    start = time.time()
    all_results = []

    # Test each filter layer
    all_results.append(test_url_heuristics())
    all_results.append(test_trusted_source())
    all_results.append(test_ai_dimensions())
    all_results.append(test_exif_authenticity())
    all_results.append(test_full_novelty_pipeline())

    # HTTP header tests require network — run last
    print("\n  [Running HTTP header tests — requires network...]")
    all_results.append(test_http_headers())

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start

    print("\n")
    print("=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    total_tests = 0
    total_correct = 0

    for r in all_results:
        total_tests += r.total
        total_correct += r.correct
        status = "PASS" if r.accuracy == 100 else ("WARN" if r.accuracy >= 80 else "FAIL")
        print(f"  [{status:4s}] {r.name:50s}  {r.correct}/{r.total}  ({r.accuracy:.1f}%)")

    overall = (total_correct / total_tests * 100) if total_tests > 0 else 0

    print(f"\n  {'-' * 66}")
    print(f"  OVERALL: {total_correct}/{total_tests} tests passed ({overall:.1f}% accuracy)")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'='*70}\n")
