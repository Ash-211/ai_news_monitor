import sys
import os

sys.path.append(os.path.abspath('.'))

from src.intelligence.fake_news import _check_text_quality, _apply_scoring_heuristics, load_fake_news_detector

title1 = "In Big Escalation, US Says Will Block Strait Of Hormuz After Iran Talks Fail"
content1 = "US President Donald Trump has ordered a naval blockade of the Strait of Hormuz in response to Iran's \"unyielding\" refusal to give up its nuclear"
title2 = "Dalit medical student's death rocks Kerala, family claims college faculty humiliated him over caste, colour"
content2 = "The death of a 22-year-old Dalit medical student has rocked Kerala, with allegations emerging that he had been repeatedly subjected to insults and"

print("--- NDTV Article ---")
print(_check_text_quality(content1))
res1 = _apply_scoring_heuristics(title1, content1, 0.54, source="NDTV News SEARCH RECORDS", corroboration_count=1)
print(res1)

print("\n--- Indian Express Article ---")
print(_check_text_quality(content2))
res2 = _apply_scoring_heuristics(title2, content2, 0.80, source="THE INDIAN EXPRESS | WORLD", corroboration_count=1)
print(res2)
