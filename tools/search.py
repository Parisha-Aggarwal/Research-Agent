"""
tools/search.py — Web search utility using DuckDuckGo (ddgs library).

No API key required. Returns a clean list of results.
The Researcher agent calls search_web() directly.
"""

import logging
from typing import List, Dict
from ddgs import DDGS
from config import MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


#  The method below returns:
#         List of dicts with keys: title, url, snippet.
#         Returns empty list on any failure — never raises.

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
    logger.info(f"Searching web for: '{query}'")

    try:
        results = []

        with DDGS as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)

            for r in raw_results:
                results.append({
                    "title":   r.get("title", "No title"),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        logger.info(f"Found {len(results)} results for: '{query}'")
        return results
    
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return []