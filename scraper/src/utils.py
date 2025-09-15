import logging
from datetime import datetime

def setup_logging():
    """Sets up logging for the scraper."""
    logging.basicConfig(
        level=logging.INFO,  # INFO level shows progress; change to DEBUG for more details
        format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamped logs
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler('scraper_log.txt')  # Save to file for review
        ]
    )
    return logging.getLogger(__name__)

def parse_date(date_str):
    """Parses RSS published date into a standard format (e.g., ISO)."""
    try:
        # Common RSS date format: 'Mon, 08 Sep 2025 12:00:00 GMT'
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z').isoformat()
    except ValueError:
        logger = setup_logging()
        logger.warning(f"Invalid date format: {date_str}. Using current time.")
        return datetime.now().isoformat()
