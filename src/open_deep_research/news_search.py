"""
Google Programmable Search Engine integration for time-filtered news search.

This module is for news/sentiment search ONLY. Never use for financial data
(use SEC EDGAR for that).

Why Google PSE instead of Tavily:
- Tavily ignores strict date filters
- Google PSE respects dateRestrict parameter
- Critical for "Time Machine Mode" (no look-ahead bias)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urlparse

from googleapiclient.discovery import build

from open_deep_research.models import (
    Fact,
    Location,
    NewsResult,
    get_domain_tier,
)


logger = logging.getLogger(__name__)


# Tier 1 domains - highest credibility financial news
TIER1_DOMAINS = ["wsj.com", "reuters.com", "bloomberg.com", "ft.com"]


# =============================================================================
# News Article Model (extended)
# =============================================================================


class NewsArticle(NewsResult):
    """Extended NewsResult with full article content for extraction."""
    
    content: Optional[str] = None  # Full article text if fetched


def get_google_pse_credentials() -> tuple[str, str]:
    """Load Google PSE credentials from environment variables.
    
    Returns:
        Tuple of (api_key, engine_id)
        
    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.environ.get("GOOGLE_PSE_API_KEY")
    engine_id = os.environ.get("GOOGLE_PSE_ENGINE_ID")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_PSE_API_KEY environment variable required.\n"
            "Get your API key from: https://console.cloud.google.com/\n"
            "Enable the Custom Search API first."
        )
    if not engine_id:
        raise ValueError(
            "GOOGLE_PSE_ENGINE_ID environment variable required.\n"
            "Create a search engine at: https://programmablesearchengine.google.com/"
        )
    
    return api_key, engine_id


def extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "")


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse various date formats to datetime."""
    if not date_str:
        return None
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def filter_by_date(
    results: list[NewsResult],
    after_date: str | None = None,
    before_date: str | None = None
) -> list[NewsResult]:
    """Filter results by publication date.
    
    Args:
        results: List of NewsResult objects
        after_date: Only include results published after this date (YYYY-MM-DD)
        before_date: Only include results published before this date (YYYY-MM-DD)
        
    Returns:
        Filtered list of NewsResult objects
    """
    if not after_date and not before_date:
        return results
    
    after_dt = datetime.strptime(after_date, "%Y-%m-%d") if after_date else None
    before_dt = datetime.strptime(before_date, "%Y-%m-%d") if before_date else None
    
    filtered = []
    for result in results:
        if not result.published_date:
            # Keep results without dates - can't verify them
            # Alternatively, could exclude them with strict mode
            continue
            
        pub_dt = _parse_date(result.published_date)
        if not pub_dt:
            continue
            
        if after_dt and pub_dt < after_dt:
            continue
        if before_dt and pub_dt >= before_dt:
            continue
            
        filtered.append(result)
    
    return filtered


def search_news(
    query: str,
    after_date: str | None = None,
    before_date: str | None = None,
    domains: list[str] | None = None,
    num_results: int = 10
) -> list[NewsResult]:
    """Search for news using Google Programmable Search Engine.
    
    Args:
        query: Search query string
        after_date: Only return results published after this date (YYYY-MM-DD)
        before_date: Only return results published before this date (YYYY-MM-DD)
        domains: List of domains to restrict search to (e.g., ["wsj.com", "reuters.com"])
        num_results: Maximum number of results to return (max 10 per API call)
        
    Returns:
        List of NewsResult objects
    """
    api_key, engine_id = get_google_pse_credentials()
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Build search query
    search_query = query
    
    # Add domain restriction if specified
    if domains:
        site_filter = " OR ".join([f"site:{d}" for d in domains])
        search_query = f"{query} ({site_filter})"
    
    # Build request parameters
    request_params = {
        "q": search_query,
        "cx": engine_id,
        "num": min(num_results, 10),  # API max is 10 per request
    }
    
    # Add date sorting for better date-filtered results
    if before_date or after_date:
        request_params["sort"] = "date"
    
    # Execute search
    result = service.cse().list(**request_params).execute()
    
    # Parse results
    items = result.get("items", [])
    news_results = []
    
    for item in items:
        url = item.get("link", "")
        
        # Try to extract publish date from metadata
        published_date = None
        if "pagemap" in item:
            metatags = item["pagemap"].get("metatags", [{}])
            if metatags:
                meta = metatags[0]
                published_date = (
                    meta.get("article:published_time") or
                    meta.get("og:published_time") or
                    meta.get("date") or
                    meta.get("pubdate")
                )
        
        news_results.append(NewsResult(
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", ""),
            published_date=published_date,
            domain=extract_domain(url),
            tier=get_domain_tier(url),
            retrieved_at=datetime.utcnow()
        ))
    
    # Filter by date if specified
    if after_date or before_date:
        news_results = filter_by_date(news_results, after_date, before_date)
    
    return news_results


def search_tier1_news(
    query: str,
    after_date: str | None = None,
    before_date: str | None = None,
    num_results: int = 10
) -> list[NewsResult]:
    """Search for news restricted to Tier 1 sources only.
    
    Tier 1 domains: wsj.com, reuters.com, bloomberg.com, ft.com
    
    Args:
        query: Search query string
        after_date: Only return results published after this date (YYYY-MM-DD)
        before_date: Only return results published before this date (YYYY-MM-DD)
        num_results: Maximum number of results to return (max 10 per API call)
        
    Returns:
        List of NewsResult objects from Tier 1 sources only
    """
    return search_news(
        query=query,
        after_date=after_date,
        before_date=before_date,
        domains=TIER1_DOMAINS,
        num_results=num_results
    )


def search_news_point_in_time(
    query: str,
    as_of_date: str
) -> list[NewsResult]:
    """Search for news as of a specific point in time ("Time Machine Mode").
    
    This is critical for avoiding look-ahead bias in backtesting. Only returns
    results published BEFORE the specified date.
    
    Args:
        query: Search query string
        as_of_date: Return only results published before this date (YYYY-MM-DD)
        
    Returns:
        List of NewsResult objects published before as_of_date
    """
    return search_news(
        query=query,
        before_date=as_of_date
    )


# =============================================================================
# Pipeline Integration: search_news with ticker + days_back
# =============================================================================


def search_news_for_ticker(
    query: str,
    ticker: str,
    days_back: int = 7
) -> list[NewsArticle]:
    """Search for news about a ticker, restricted to Tier 1 domains.
    
    This is the main entry point for the pipeline integration.
    
    Args:
        query: Search query string (will be combined with ticker)
        ticker: Stock ticker symbol (e.g., "NVDA")
        days_back: Number of days to look back (default 7)
        
    Returns:
        List of NewsArticle objects from Tier 1 sources
    """
    # Calculate date range
    today = datetime.utcnow()
    after_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Combine query with ticker
    full_query = f"{ticker} {query}"
    
    # Search Tier 1 sources
    results = search_tier1_news(
        query=full_query,
        after_date=after_date,
        num_results=10
    )
    
    # Convert to NewsArticle (with content=None initially)
    # Content would be fetched separately if needed
    articles = []
    for result in results:
        article = NewsArticle(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            published_date=result.published_date,
            domain=result.domain,
            tier=result.tier,
            retrieved_at=result.retrieved_at,
            content=None  # Content not fetched from search
        )
        articles.append(article)
    
    return articles


# =============================================================================
# News Fact Extraction
# =============================================================================


NEWS_EXTRACTION_PROMPT = '''
Extract financial facts from the following news article about {ticker}.

ARTICLE TITLE: {title}
ARTICLE URL: {url}
ARTICLE DATE: {date}
ARTICLE CONTENT:
{content}

RULES:
1. Only extract facts that are EXPLICITLY stated in the article
2. Do NOT infer, calculate, or guess any values
3. Each fact must include sentence_string which is an EXACT QUOTE from the article
4. The sentence_string must be a complete sentence or phrase that contains the fact
5. If no financial facts are present, return an empty list []
6. Focus on: revenue, earnings, EPS, guidance, market cap, stock price changes
7. Exclude opinions, analyst estimates (unless clearly labeled), and speculative statements

OUTPUT FORMAT (JSON array):
[
  {{
    "metric": "string - name of the metric (e.g., 'revenue', 'net_income', 'eps')",
    "value": number or null,
    "unit": "string - USD, shares, percent, etc.",
    "period": "string - e.g., 'Q3 FY2025', 'FY2024', or 'N/A' if not specified",
    "period_end_date": "string - date if available, e.g., '2024-10-27', or empty",
    "sentence_string": "string - EXACT quote from article containing this fact"
  }}
]

Return ONLY valid JSON. No explanations.
'''


def _call_llm_for_news(prompt: str) -> str:
    """Call the LLM for news fact extraction.
    
    Uses the same client as extraction.py but with news-specific handling.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("Anthropic package not installed")
        raise
    
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _parse_llm_response(response: str) -> list[dict]:
    """Parse LLM JSON response for news extraction."""
    if not response or not response.strip():
        return []
    
    text = response.strip()
    
    # Extract from markdown code blocks if present
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        else:
            logger.warning(f"LLM response is not a list: {type(result)}")
            return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return []


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for substring comparison (whitespace-safe)."""
    if not text:
        return ""
    
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _create_news_fact(
    raw: dict,
    ticker: str,
    article: NewsArticle
) -> Fact:
    """Create a Fact object from raw LLM extraction for news.
    
    Args:
        raw: Raw extraction dict from LLM
        ticker: Company ticker
        article: The source NewsArticle
        
    Returns:
        Fact object with source_format="news" and trust_level="medium"
    """
    # Generate a unique snapshot_id based on article URL
    content_for_hash = (article.content or article.snippet or "")
    content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()
    snapshot_id = hashlib.sha256(article.url.encode()).hexdigest()[:16]
    
    return Fact(
        fact_id=str(uuid.uuid4()),
        entity=ticker,
        metric=raw.get("metric") or "",
        value=raw.get("value"),
        unit=raw.get("unit") or "USD",
        period=raw.get("period") or "",
        period_end_date=raw.get("period_end_date") or "",
        location=Location(
            cik="",  # News doesn't have CIK
            doc_date=article.published_date or datetime.utcnow().strftime("%Y-%m-%d"),
            doc_type="news",
            section_id="",
            paragraph_index=None,
            sentence_string=raw.get("sentence_string"),
            # Table fields are None for news
            table_index=None,
            row_index=None,
            column_index=None,
            row_label=None,
            column_label=None,
            # News-specific fields
            article_url=article.url,
            article_title=article.title,
            article_domain=article.domain,
            article_published_date=article.published_date,
        ),
        source_format="news",
        extracted_scale=None,
        doc_hash=content_hash,
        snapshot_id=snapshot_id,
        verification_status="unverified",
        trust_level="medium",  # News facts get medium trust (lower than XBRL)
    )


def verify_news_fact(fact: Fact, source_text: str) -> Fact:
    """Verify a news-based fact.
    
    For source_format == "news" only.
    Uses the text verification path (sentence_string must exist in source).
    
    Args:
        fact: The Fact to verify (must have source_format == "news")
        source_text: The article text the fact was extracted from
        
    Returns:
        The fact with updated verification_status
    """
    if fact.source_format != "news":
        raise ValueError("verify_news_fact only handles news facts")
    
    # Text verification: sentence_string must exist in source
    if not fact.location.sentence_string:
        fact.verification_status = "mismatch"
        return fact
    
    normalized_sentence = _normalize_for_comparison(fact.location.sentence_string)
    normalized_source = _normalize_for_comparison(source_text)
    
    if normalized_sentence not in normalized_source:
        fact.verification_status = "mismatch"
        return fact
    
    # Sentence exists in source - mark as verified
    fact.verification_status = "exact_match"
    return fact


def extract_facts_from_news(
    articles: list[NewsArticle],
    ticker: str
) -> list[Fact]:
    """Extract verified facts from news articles.
    
    This is the main pipeline entry point for news fact extraction.
    
    Steps:
    1. For each article, use LLM to extract facts
    2. Verify each fact (sentence_string exists in source)
    3. Only return verified facts
    
    Args:
        articles: List of NewsArticle objects (must have content or snippet)
        ticker: Stock ticker symbol
        
    Returns:
        List of verified Fact objects with trust_level="medium"
    """
    verified_facts = []
    rejected_count = 0
    
    for article in articles:
        # Use content if available, fall back to snippet
        source_text = article.content or article.snippet
        if not source_text or not source_text.strip():
            logger.warning(f"Skipping article with no content: {article.url}")
            continue
        
        # Create extraction prompt
        prompt = NEWS_EXTRACTION_PROMPT.format(
            ticker=ticker,
            title=article.title,
            url=article.url,
            date=article.published_date or "Unknown",
            content=source_text
        )
        
        try:
            response = _call_llm_for_news(prompt)
        except Exception as e:
            logger.error(f"LLM extraction failed for {article.url}: {e}")
            continue
        
        raw_facts = _parse_llm_response(response)
        
        if not raw_facts:
            continue
        
        # Create and verify facts
        for raw in raw_facts:
            try:
                fact = _create_news_fact(raw, ticker, article)
                
                # Verify: sentence_string must exist in source
                verified_fact = verify_news_fact(fact, source_text)
                
                if verified_fact.verification_status in ("exact_match", "approximate_match"):
                    verified_facts.append(verified_fact)
                    logger.info(
                        f"Verified news fact: {fact.metric} = {fact.value} "
                        f"from {article.domain}"
                    )
                else:
                    rejected_count += 1
                    logger.warning(
                        f"Rejected news fact: {fact.metric} - "
                        f"sentence not found in source"
                    )
            except Exception as e:
                logger.warning(f"Failed to create news fact: {e}")
                continue
    
    logger.info(
        f"News extraction complete: {len(verified_facts)} verified, "
        f"{rejected_count} rejected from {len(articles)} articles"
    )
    
    return verified_facts


# =============================================================================
# Convenience: Full news pipeline (search + extract)
# =============================================================================


def search_and_extract_news_facts(
    query: str,
    ticker: str,
    days_back: int = 7
) -> list[Fact]:
    """Search for news and extract verified facts.
    
    Convenience function combining search_news_for_ticker and extract_facts_from_news.
    Note: This only uses snippets from search results, not full article content.
    For better extraction, fetch full article content separately.
    
    Args:
        query: Search query
        ticker: Stock ticker
        days_back: Days to look back
        
    Returns:
        List of verified facts from news articles
    """
    articles = search_news_for_ticker(query, ticker, days_back)
    
    if not articles:
        logger.info(f"No news articles found for {ticker}")
        return []
    
    return extract_facts_from_news(articles, ticker)

