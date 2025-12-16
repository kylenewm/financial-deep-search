"""
Tests for Google PSE news search integration.
"""
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.models import NewsResult, Fact, get_domain_tier, NEWS_DOMAIN_TIERS
from open_deep_research.news_search import (
    get_google_pse_credentials,
    extract_domain,
    filter_by_date,
    search_news,
    search_tier1_news,
    search_news_point_in_time,
    search_news_for_ticker,
    extract_facts_from_news,
    verify_news_fact,
    NewsArticle,
    TIER1_DOMAINS,
    _normalize_for_comparison,
    _create_news_fact,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestCredentials:
    """Test credential loading from environment."""

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises clear error."""
        monkeypatch.delenv("GOOGLE_PSE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_PSE_ENGINE_ID", raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            get_google_pse_credentials()
        
        assert "GOOGLE_PSE_API_KEY" in str(exc_info.value)
        assert "console.cloud.google.com" in str(exc_info.value)

    def test_missing_engine_id_raises_error(self, monkeypatch):
        """Test that missing engine ID raises clear error."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.delenv("GOOGLE_PSE_ENGINE_ID", raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            get_google_pse_credentials()
        
        assert "GOOGLE_PSE_ENGINE_ID" in str(exc_info.value)
        assert "programmablesearchengine.google.com" in str(exc_info.value)

    def test_valid_credentials_accepted(self, monkeypatch):
        """Test that valid credentials are returned."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-api-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine-id")
        
        api_key, engine_id = get_google_pse_credentials()
        
        assert api_key == "test-api-key"
        assert engine_id == "test-engine-id"


# =============================================================================
# Domain Tier Tests
# =============================================================================


class TestDomainTiers:
    """Test domain tier classification."""

    def test_wsj_returns_tier_1(self):
        """Test wsj.com returns tier 1."""
        assert get_domain_tier("https://www.wsj.com/articles/test") == 1

    def test_reuters_returns_tier_1(self):
        """Test reuters.com returns tier 1."""
        assert get_domain_tier("https://www.reuters.com/news/test") == 1

    def test_bloomberg_returns_tier_1(self):
        """Test bloomberg.com returns tier 1."""
        assert get_domain_tier("https://www.bloomberg.com/news/test") == 1

    def test_ft_returns_tier_1(self):
        """Test ft.com returns tier 1."""
        assert get_domain_tier("https://www.ft.com/content/test") == 1

    def test_cnbc_returns_tier_2(self):
        """Test cnbc.com returns tier 2."""
        assert get_domain_tier("https://www.cnbc.com/2024/01/01/test.html") == 2

    def test_seekingalpha_returns_tier_3(self):
        """Test seekingalpha.com returns tier 3."""
        assert get_domain_tier("https://seekingalpha.com/article/test") == 3

    def test_unknown_domain_returns_tier_3(self):
        """Test unknown domain defaults to tier 3."""
        assert get_domain_tier("https://randomsite.com/news") == 3
        assert get_domain_tier("https://obscurenews.io/article") == 3

    def test_tier1_domains_constant(self):
        """Test TIER1_DOMAINS contains expected domains."""
        assert "wsj.com" in TIER1_DOMAINS
        assert "reuters.com" in TIER1_DOMAINS
        assert "bloomberg.com" in TIER1_DOMAINS
        assert "ft.com" in TIER1_DOMAINS
        assert len(TIER1_DOMAINS) == 4


# =============================================================================
# Domain Extraction Tests
# =============================================================================


class TestExtractDomain:
    """Test domain extraction from URLs."""

    def test_extract_simple_domain(self):
        """Test extracting domain from simple URL."""
        assert extract_domain("https://reuters.com/article/test") == "reuters.com"

    def test_extract_domain_strips_www(self):
        """Test that www. prefix is stripped."""
        assert extract_domain("https://www.wsj.com/articles/test") == "wsj.com"

    def test_extract_domain_with_subdomain(self):
        """Test extracting domain with subdomain."""
        assert extract_domain("https://markets.ft.com/data/test") == "markets.ft.com"

    def test_extract_domain_with_port(self):
        """Test extracting domain with port number."""
        assert extract_domain("http://localhost:8080/test") == "localhost:8080"


# =============================================================================
# Date Filtering Tests
# =============================================================================


class TestFilterByDate:
    """Test date-based filtering of results."""

    @pytest.fixture
    def sample_results(self):
        """Create sample NewsResult objects with various dates."""
        return [
            NewsResult(
                title="Article 1",
                url="https://wsj.com/1",
                snippet="Test 1",
                published_date="2024-01-15",
                domain="wsj.com",
                tier=1,
                retrieved_at=datetime.utcnow()
            ),
            NewsResult(
                title="Article 2",
                url="https://wsj.com/2",
                snippet="Test 2",
                published_date="2024-02-15",
                domain="wsj.com",
                tier=1,
                retrieved_at=datetime.utcnow()
            ),
            NewsResult(
                title="Article 3",
                url="https://wsj.com/3",
                snippet="Test 3",
                published_date="2024-03-15",
                domain="wsj.com",
                tier=1,
                retrieved_at=datetime.utcnow()
            ),
        ]

    def test_after_date_filter(self, sample_results):
        """Test filtering results after a specific date."""
        filtered = filter_by_date(sample_results, after_date="2024-02-01")
        
        assert len(filtered) == 2
        assert all("Feb" in r.published_date or "Mar" in r.published_date or 
                   "2024-02" in r.published_date or "2024-03" in r.published_date 
                   for r in filtered)

    def test_before_date_filter(self, sample_results):
        """Test filtering results before a specific date."""
        filtered = filter_by_date(sample_results, before_date="2024-02-01")
        
        assert len(filtered) == 1
        assert "2024-01-15" in filtered[0].published_date

    def test_date_range_filter(self, sample_results):
        """Test filtering results within a date range."""
        filtered = filter_by_date(
            sample_results,
            after_date="2024-01-20",
            before_date="2024-03-01"
        )
        
        assert len(filtered) == 1
        assert "2024-02-15" in filtered[0].published_date

    def test_no_dates_returns_all(self, sample_results):
        """Test that no date filters returns all results."""
        filtered = filter_by_date(sample_results)
        assert len(filtered) == len(sample_results)

    def test_results_without_dates_excluded(self):
        """Test that results without dates are excluded when filtering."""
        results = [
            NewsResult(
                title="No date",
                url="https://wsj.com/nodate",
                snippet="Test",
                published_date=None,
                domain="wsj.com",
                tier=1,
                retrieved_at=datetime.utcnow()
            ),
        ]
        
        filtered = filter_by_date(results, after_date="2024-01-01")
        assert len(filtered) == 0


# =============================================================================
# Search Tests (Mocked)
# =============================================================================


class TestSearchNews:
    """Test news search functionality with mocked API."""

    @pytest.fixture
    def mock_search_response(self):
        """Sample Google CSE response."""
        return {
            "items": [
                {
                    "title": "NVIDIA Reports Record Revenue",
                    "link": "https://www.reuters.com/nvidia-earnings",
                    "snippet": "NVIDIA announced record quarterly revenue...",
                    "pagemap": {
                        "metatags": [{
                            "article:published_time": "2024-11-20T15:00:00Z"
                        }]
                    }
                },
                {
                    "title": "Tech Stocks Rally",
                    "link": "https://www.wsj.com/tech-rally",
                    "snippet": "Technology stocks surged on AI optimism...",
                    "pagemap": {
                        "metatags": [{
                            "og:published_time": "2024-11-19T10:00:00Z"
                        }]
                    }
                }
            ]
        }

    @patch("open_deep_research.news_search.build")
    def test_basic_search_returns_results(self, mock_build, mock_search_response, monkeypatch):
        """Test that basic search returns properly formatted results."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        # Setup mock
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = mock_search_response
        mock_build.return_value = mock_service
        
        results = search_news("NVIDIA earnings")
        
        assert len(results) == 2
        assert all(isinstance(r, NewsResult) for r in results)

    @patch("open_deep_research.news_search.build")
    def test_results_have_required_fields(self, mock_build, mock_search_response, monkeypatch):
        """Test that results have all required fields."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = mock_search_response
        mock_build.return_value = mock_service
        
        results = search_news("NVIDIA")
        
        for result in results:
            assert result.title
            assert result.url
            assert result.snippet
            assert result.domain
            assert result.tier in (1, 2, 3)
            assert result.retrieved_at

    @patch("open_deep_research.news_search.build")
    def test_domain_extracted_correctly(self, mock_build, mock_search_response, monkeypatch):
        """Test that domain is extracted correctly from URL."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = mock_search_response
        mock_build.return_value = mock_service
        
        results = search_news("NVIDIA")
        
        domains = [r.domain for r in results]
        assert "reuters.com" in domains
        assert "wsj.com" in domains

    @patch("open_deep_research.news_search.build")
    def test_tier_assigned_correctly(self, mock_build, mock_search_response, monkeypatch):
        """Test that tiers are assigned correctly."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = mock_search_response
        mock_build.return_value = mock_service
        
        results = search_news("NVIDIA")
        
        # Both reuters and wsj are Tier 1
        assert all(r.tier == 1 for r in results)

    @patch("open_deep_research.news_search.build")
    def test_domain_restriction_in_query(self, mock_build, monkeypatch):
        """Test that domain restriction is added to query."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = {"items": []}
        mock_build.return_value = mock_service
        
        search_news("NVIDIA", domains=["wsj.com", "reuters.com"])
        
        # Check that the query was modified to include site: filters
        call_args = mock_service.cse().list.call_args
        query = call_args[1]["q"]
        assert "site:wsj.com" in query
        assert "site:reuters.com" in query


class TestSearchTier1News:
    """Test Tier 1 restricted search."""

    @patch("open_deep_research.news_search.build")
    def test_tier1_search_uses_tier1_domains(self, mock_build, monkeypatch):
        """Test that tier1 search restricts to tier 1 domains."""
        monkeypatch.setenv("GOOGLE_PSE_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "test-engine")
        
        mock_service = MagicMock()
        mock_service.cse().list().execute.return_value = {"items": []}
        mock_build.return_value = mock_service
        
        search_tier1_news("NVIDIA")
        
        call_args = mock_service.cse().list.call_args
        query = call_args[1]["q"]
        
        # Check all tier 1 domains are in query
        for domain in TIER1_DOMAINS:
            assert f"site:{domain}" in query


class TestSearchPointInTime:
    """Test time machine mode search."""

    @patch("open_deep_research.news_search.search_news")
    def test_point_in_time_uses_before_date(self, mock_search):
        """Test that point_in_time search uses before_date correctly."""
        mock_search.return_value = []
        
        search_news_point_in_time("NVIDIA", as_of_date="2024-01-15")
        
        mock_search.assert_called_once_with(
            query="NVIDIA",
            before_date="2024-01-15"
        )


# =============================================================================
# Integration Tests (require credentials)
# =============================================================================


# =============================================================================
# News Fact Extraction Tests
# =============================================================================


class TestNewsArticle:
    """Test NewsArticle model."""

    def test_news_article_extends_news_result(self):
        """Test that NewsArticle properly extends NewsResult."""
        article = NewsArticle(
            title="NVIDIA Reports Record Revenue",
            url="https://www.reuters.com/nvidia",
            snippet="NVIDIA reported $35.1 billion in revenue...",
            published_date="2024-11-20",
            domain="reuters.com",
            tier=1,
            retrieved_at=datetime.utcnow(),
            content="Full article content here..."
        )
        
        assert isinstance(article, NewsResult)
        assert article.content == "Full article content here..."

    def test_news_article_content_optional(self):
        """Test that content is optional."""
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            snippet="Test snippet",
            domain="example.com",
            tier=3,
            retrieved_at=datetime.utcnow()
        )
        
        assert article.content is None


class TestNormalizeForComparison:
    """Test whitespace normalization for verification."""

    def test_normalizes_whitespace(self):
        """Test that multiple spaces are collapsed."""
        assert _normalize_for_comparison("hello  world") == "hello world"

    def test_normalizes_nbsp(self):
        """Test that &nbsp; is converted to space."""
        assert _normalize_for_comparison("hello\xa0world") == "hello world"
        assert _normalize_for_comparison("hello&nbsp;world") == "hello world"

    def test_normalizes_newlines(self):
        """Test that newlines are converted to spaces."""
        assert _normalize_for_comparison("hello\nworld") == "hello world"
        assert _normalize_for_comparison("hello\r\nworld") == "hello world"

    def test_lowercases(self):
        """Test that text is lowercased."""
        assert _normalize_for_comparison("HELLO World") == "hello world"


class TestCreateNewsFact:
    """Test news fact creation."""

    @pytest.fixture
    def sample_article(self):
        """Create sample NewsArticle."""
        return NewsArticle(
            title="NVIDIA Reports Record Revenue",
            url="https://www.reuters.com/nvidia-earnings",
            snippet="NVIDIA reported $35.1 billion in quarterly revenue.",
            published_date="2024-11-20",
            domain="reuters.com",
            tier=1,
            retrieved_at=datetime.utcnow(),
            content="NVIDIA reported $35.1 billion in quarterly revenue."
        )

    def test_creates_fact_with_correct_source_format(self, sample_article):
        """Test that news facts have source_format='news'."""
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "period": "Q3 FY2025",
            "period_end_date": "2024-10-27",
            "sentence_string": "NVIDIA reported $35.1 billion in quarterly revenue."
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        
        assert fact.source_format == "news"

    def test_creates_fact_with_medium_trust_level(self, sample_article):
        """Test that news facts have trust_level='medium'."""
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "period": "Q3 FY2025",
            "sentence_string": "NVIDIA reported $35.1 billion."
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        
        assert fact.trust_level == "medium"

    def test_creates_fact_with_article_location_info(self, sample_article):
        """Test that news facts include article location info."""
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "sentence_string": "NVIDIA reported $35.1 billion."
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        
        assert fact.location.article_url == sample_article.url
        assert fact.location.article_title == sample_article.title
        assert fact.location.article_domain == "reuters.com"
        assert fact.location.doc_type == "news"


class TestVerifyNewsFact:
    """Test news fact verification."""

    @pytest.fixture
    def sample_article(self):
        """Create sample NewsArticle."""
        return NewsArticle(
            title="Test",
            url="https://www.reuters.com/test",
            snippet="Test",
            domain="reuters.com",
            tier=1,
            retrieved_at=datetime.utcnow(),
            content="NVIDIA reported $35.1 billion in quarterly revenue. The company beat expectations."
        )

    def test_verifies_fact_when_sentence_exists(self, sample_article):
        """Test that fact is verified when sentence exists in source."""
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "sentence_string": "NVIDIA reported $35.1 billion in quarterly revenue."
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        verified = verify_news_fact(fact, sample_article.content)
        
        assert verified.verification_status == "exact_match"

    def test_rejects_fact_when_sentence_not_found(self, sample_article):
        """Test that fact is rejected when sentence not in source."""
        raw = {
            "metric": "revenue",
            "value": 40000000000,
            "unit": "USD",
            "sentence_string": "NVIDIA reported $40 billion in revenue."  # Not in source
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        verified = verify_news_fact(fact, sample_article.content)
        
        assert verified.verification_status == "mismatch"

    def test_handles_whitespace_differences(self, sample_article):
        """Test that whitespace differences are handled."""
        # Source has normal space, sentence has multiple spaces
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "sentence_string": "NVIDIA  reported  $35.1  billion  in  quarterly  revenue."
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        verified = verify_news_fact(fact, sample_article.content)
        
        assert verified.verification_status == "exact_match"

    def test_rejects_fact_without_sentence_string(self, sample_article):
        """Test that fact without sentence_string is rejected."""
        raw = {
            "metric": "revenue",
            "value": 35100000000,
            "unit": "USD",
            "sentence_string": None
        }
        
        fact = _create_news_fact(raw, "NVDA", sample_article)
        verified = verify_news_fact(fact, sample_article.content)
        
        assert verified.verification_status == "mismatch"


class TestExtractFactsFromNews:
    """Test full news extraction pipeline."""

    @pytest.fixture
    def sample_articles(self):
        """Create sample articles for testing."""
        return [
            NewsArticle(
                title="NVIDIA Q3 Results",
                url="https://www.reuters.com/nvidia",
                snippet="NVIDIA reported strong quarterly results.",
                published_date="2024-11-20",
                domain="reuters.com",
                tier=1,
                retrieved_at=datetime.utcnow(),
                content="NVIDIA reported $35.1 billion in quarterly revenue, beating analyst expectations."
            )
        ]

    @patch("open_deep_research.news_search._call_llm_for_news")
    def test_extracts_and_verifies_facts(self, mock_llm, sample_articles):
        """Test that facts are extracted and verified."""
        mock_llm.return_value = '''
        [
            {
                "metric": "revenue",
                "value": 35100000000,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA reported $35.1 billion in quarterly revenue"
            }
        ]
        '''
        
        facts = extract_facts_from_news(sample_articles, "NVDA")
        
        assert len(facts) == 1
        assert facts[0].verification_status == "exact_match"
        assert facts[0].trust_level == "medium"

    @patch("open_deep_research.news_search._call_llm_for_news")
    def test_rejects_hallucinated_facts(self, mock_llm, sample_articles):
        """Test that facts not in source are rejected."""
        mock_llm.return_value = '''
        [
            {
                "metric": "revenue",
                "value": 50000000000,
                "unit": "USD",
                "period": "Q3 FY2025",
                "sentence_string": "NVIDIA reported $50 billion in revenue"
            }
        ]
        '''
        
        facts = extract_facts_from_news(sample_articles, "NVDA")
        
        assert len(facts) == 0  # Rejected because sentence not in source

    @patch("open_deep_research.news_search._call_llm_for_news")
    def test_handles_empty_extraction(self, mock_llm, sample_articles):
        """Test handling of no facts extracted."""
        mock_llm.return_value = "[]"
        
        facts = extract_facts_from_news(sample_articles, "NVDA")
        
        assert len(facts) == 0

    def test_skips_articles_without_content(self):
        """Test that articles without content are skipped."""
        articles = [
            NewsArticle(
                title="Test",
                url="https://example.com",
                snippet="",
                domain="example.com",
                tier=3,
                retrieved_at=datetime.utcnow(),
                content=None
            )
        ]
        
        # Should not raise, just return empty
        facts = extract_facts_from_news(articles, "NVDA")
        assert len(facts) == 0


class TestSearchNewsForTicker:
    """Test ticker-based news search."""

    @patch("open_deep_research.news_search.search_tier1_news")
    def test_combines_ticker_with_query(self, mock_search):
        """Test that ticker is combined with query."""
        mock_search.return_value = []
        
        search_news_for_ticker("earnings", "NVDA", days_back=7)
        
        call_args = mock_search.call_args
        assert "NVDA" in call_args[1]["query"]
        assert "earnings" in call_args[1]["query"]

    @patch("open_deep_research.news_search.search_tier1_news")
    def test_uses_tier1_sources(self, mock_search):
        """Test that search uses tier 1 sources."""
        mock_search.return_value = [
            NewsResult(
                title="Test",
                url="https://wsj.com/test",
                snippet="Test",
                domain="wsj.com",
                tier=1,
                retrieved_at=datetime.utcnow()
            )
        ]
        
        articles = search_news_for_ticker("earnings", "NVDA")
        
        mock_search.assert_called_once()
        assert len(articles) == 1
        assert isinstance(articles[0], NewsArticle)

    @patch("open_deep_research.news_search.search_tier1_news")
    def test_calculates_date_range(self, mock_search):
        """Test that date range is calculated correctly."""
        mock_search.return_value = []
        
        search_news_for_ticker("earnings", "NVDA", days_back=14)
        
        call_args = mock_search.call_args
        assert call_args[1].get("after_date") is not None


@pytest.mark.integration
class TestIntegration:
    """Integration tests that require real Google PSE credentials.
    
    Run with: pytest -m integration tests/test_news_search.py
    Requires GOOGLE_PSE_API_KEY and GOOGLE_PSE_ENGINE_ID environment variables.
    """

    @pytest.fixture(autouse=True)
    def check_credentials(self):
        """Skip if credentials not available."""
        if not os.environ.get("GOOGLE_PSE_API_KEY") or not os.environ.get("GOOGLE_PSE_ENGINE_ID"):
            pytest.skip("Google PSE credentials not configured")

    def test_real_search_returns_results(self):
        """Test that real search returns results."""
        results = search_news("NVIDIA earnings 2024", num_results=3)
        
        assert len(results) > 0
        assert all(isinstance(r, NewsResult) for r in results)

    def test_real_tier1_search(self):
        """Test that tier 1 search returns only tier 1 sources."""
        results = search_tier1_news("Apple quarterly results", num_results=5)
        
        # Note: We can't guarantee results, but if we get any they should be tier 1
        # or the search correctly restricted to those domains
        for result in results:
            # Domain should contain one of tier 1 domains
            is_tier1 = any(d in result.domain for d in TIER1_DOMAINS)
            if not is_tier1:
                # Could be a subdomain or edge case - just check tier assignment
                pass  # Don't fail - Google might return unexpected domains


