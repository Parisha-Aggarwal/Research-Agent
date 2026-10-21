"""
tools/scraper.py — Web page scraper using requests + BeautifulSoup.
 
Fetches a URL and extracts clean readable text.
Strips scripts, styles, nav, and other noise.
The Researcher agent calls scrape_page() after search_web().
"""

import logging
import requests
from bs4 import BeautifulSoup
from config import MAX_SCRAPE_CHARS

logger = logging.getLogger(__name__)

# Timeout so we don't hang on a slow page (seconds)
REQUEST_TIMEOUT = 8
 
# Browser-like header — some sites block requests with no User-Agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

#  Tags whose content we always discard — noise, not information
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]

"""
    Below function fetch a web page and return its clean text content.
 
    Args:
        url: The full URL to scrape.
 
    Returns:
        A plain-text string of the page's main content,
        truncated to MAX_SCRAPE_CHARS from config.
        Returns empty string on any failure — never raises.
 
    Why we truncate:
        A full news article can be 10,000+ characters.
        We only need the key facts, not the full page.
        Smaller text = faster LLM calls = less cost.
"""

def scrape_page(url: str) -> str:
    logger.info(f"Scrapping : {url}")

    try:
        
        # Fetch the page with a timeout
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
 
        # Raise an exception for bad status codes (4xx, 5xx)
        response.raise_for_status()
 
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")
 
        # Remove all noise tags in-place
        for tag in soup(NOISE_TAGS):
            tag.decompose()
 
        # Extract remaining text, collapsing whitespace
        raw_text = soup.get_text(separator=" ", strip=True)
 
        # Collapse multiple spaces/newlines into single spaces
        clean_text = " ".join(raw_text.split())
 
        # Truncate to keep memory usage low
        if len(clean_text) > MAX_SCRAPE_CHARS:
            clean_text = clean_text[:MAX_SCRAPE_CHARS] + "..."
 
        logger.info(f"Scraped {len(clean_text)} chars from {url}")
        return clean_text
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout scraping {url}")
        return ""
 
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error scraping {url}: {e}")
        return ""
 
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        return ""