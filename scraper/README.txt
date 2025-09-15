# Scraper Documentation

## Overview
Fetches news from region-specific RSS feeds, processes articles, and saves to `articles.json` (no DB pushing). Supports 5-7 regions (GLOBAL, EU, IN, US, AS, AF, LATAM).

## Setup
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

2. Install dependencies:
pip install -r requirements.txt

3. Update config.yaml with RSS feeds

4. Run src/scraper.py
