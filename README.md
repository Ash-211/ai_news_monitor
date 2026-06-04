---
title: AI News API
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI-Based News Monitoring and Automated Daily Briefing System

An end-to-end 5-layer pipeline that ingests, cleans, classifies, and summarizes news articles, served on a web dashboard.

## Overview
1. **Data Ingestion Layer**: Fetches articles from NewsAPI and RSS.
2. **NLP Preprocessing Layer**: Cleans text using NLTK and spaCy.
3. **Core Intelligence Layer**: Multi-class categorization, fake news detection, and topic modeling (LDA).
4. **Briefing Generation Layer**: Extractive and abstractive (BART) summarization.
5. **Presentation Layer**: Streamlit web dashboard.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Rename or update `.env` to include your valid API keys:
   ```
   NEWSAPI_KEY=your_newsapi_key_here
   ```

3. **Run Data Ingestion Scheduler**
   ```bash
   python -m src.ingestion.scheduler
   ```
   This will initialize the SQLite database at `data/database.sqlite` and begin fetching articles.
