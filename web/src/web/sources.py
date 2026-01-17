#!/usr/bin/env python3
"""Multiple search sources for web tool."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel


class SearchResult(BaseModel):
    """A search result."""

    title: str
    url: str
    description: str
    source: str
    timestamp: float


class SearchSource(ABC):
    """Abstract base class for search sources."""

    name: str = "abstract"
    description: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using this source.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if source is available.

        Returns:
            True if source is working, False otherwise
        """
        pass


class DuckDuckGoSource(SearchSource):
    """DuckDuckGo search source."""

    name = "duckduckgo"
    description = "DuckDuckGo privacy-focused search engine"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                    results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            url=result.get("href", ""),
                            description=result.get("body", ""),
                            source=self.name,
                            timestamp=0,
                        )
                    )
            logger.info(f"DuckDuckGo: Found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Check DuckDuckGo availability."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = ddgs.text("test", max_results=1)
                return len(list(results)) > 0
        except Exception as e:
            logger.debug(f"DuckDuckGo health check failed: {e}")
            return False


class GoogleSource(SearchSource):
    """Google search source (requires API key)."""

    name = "google"
    description = "Google Custom Search API"

    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """Initialize Google search source."""
        self.api_key = api_key
        self.search_engine_id = search_engine_id

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Google Custom Search."""
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google source not configured (missing API key or search engine ID)")
            return []

        try:
            import httpx

            results = []
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.search_engine_id,
                "num": min(max_results, 10),
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1", params=params
                )
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("items", []):
                        results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                url=item.get("link", ""),
                                description=item.get("snippet", ""),
                                source=self.name,
                                timestamp=0,
                            )
                        )
            logger.info(f"Google: Found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Check Google API availability."""
        if not self.api_key or not self.search_engine_id:
            return False
        try:
            import httpx

            params = {
                "q": "test",
                "key": self.api_key,
                "cx": self.search_engine_id,
                "num": 1,
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1", params=params, timeout=5
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Google health check failed: {e}")
            return False


class BingSource(SearchSource):
    """Bing search source (requires API key)."""

    name = "bing"
    description = "Microsoft Bing Search API"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Bing search source."""
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Bing."""
        if not self.api_key:
            logger.warning("Bing source not configured (missing API key)")
            return []

        try:
            import httpx

            results = []
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, "count": min(max_results, 50)}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params=params,
                )
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("webPages", {}).get("value", []):
                        results.append(
                            SearchResult(
                                title=item.get("name", ""),
                                url=item.get("url", ""),
                                description=item.get("snippet", ""),
                                source=self.name,
                                timestamp=0,
                            )
                        )
            logger.info(f"Bing: Found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Check Bing API availability."""
        if not self.api_key:
            return False
        try:
            import httpx

            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params={"q": "test", "count": 1},
                    timeout=5,
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Bing health check failed: {e}")
            return False


class SourceManager:
    """Manage multiple search sources."""

    def __init__(self):
        """Initialize source manager."""
        self.sources: Dict[str, SearchSource] = {}
        self._register_default_sources()

    def _register_default_sources(self) -> None:
        """Register default sources."""
        self.register(DuckDuckGoSource())
        logger.info("Registered default search sources")

    def register(self, source: SearchSource) -> None:
        """Register a search source."""
        self.sources[source.name] = source
        logger.info(f"Registered search source: {source.name}")

    async def search_all(self, query: str, max_results: int = 10) -> Dict[str, List[SearchResult]]:
        """Search across all available sources."""
        results = {}
        for name, source in self.sources.items():
            try:
                source_results = await source.search(query, max_results)
                results[name] = source_results
            except Exception as e:
                logger.warning(f"Error searching {name}: {e}")
                results[name] = []
        return results

    async def search_primary(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using the primary source (DuckDuckGo)."""
        if "duckduckgo" in self.sources:
            return await self.sources["duckduckgo"].search(query, max_results)
        return []

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all sources."""
        health = {}
        for name, source in self.sources.items():
            try:
                health[name] = await source.health_check()
            except Exception as e:
                logger.debug(f"Health check failed for {name}: {e}")
                health[name] = False
        return health

    def get_source(self, name: str) -> Optional[SearchSource]:
        """Get a specific source by name."""
        return self.sources.get(name)

    def list_sources(self) -> List[str]:
        """List all registered sources."""
        return list(self.sources.keys())
