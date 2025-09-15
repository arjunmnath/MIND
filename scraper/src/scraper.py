import feedparser
import yaml
from article_processor import process_article
from utils import setup_logging
import json

logger = setup_logging()

def scrape_feed(url, region):
    """Scrapes a single RSS feed."""
    try:
        logger.info(f"Scraping {region} feed: {url}")
        feed = feedparser.parse(url)
        if feed.bozo:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

        processed_articles = []
        for entry in feed.entries:
            processed = process_article(entry, feed, region)
            if processed:
                processed_articles.append(processed)

        # Log or save to file instead of DB
        with open('articles.json', 'a') as f:
            json.dump(processed_articles, f, indent=2)
            f.write('\n')  # For readability
        logger.info(f"Scraped {len(processed_articles)} articles from {url}")
        return processed_articles
    except Exception as e:
        logger.error(f"Error scraping feed {url}: {e}")
        return []

def scrape_all_regions():
    """Scrapes all regions from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        region_feeds = config.get('region_feeds', {})
        all_articles = []

        for region, urls in region_feeds.items():
            for url in urls:
                articles = scrape_feed(url, region)
                all_articles.extend(articles)

        logger.info(f"Total articles scraped: {len(all_articles)}")
        # Return or save all_articles
    except Exception as e:
        logger.error(f"Error in scrape_all_regions: {e}")

if __name__ == "__main__":
    scrape_all_regions()
