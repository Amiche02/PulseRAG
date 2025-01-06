import requests
import re
from abc import ABC, abstractmethod
from typing import List, Dict

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

try:
    from langchain.utilities import DuckDuckGoSearchAPIWrapper
    from langchain.document_loaders import UnstructuredURLLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class BaseWebSearchService(ABC):
    """
    Common interface for a web search + scraping service.
    Subclasses must implement `search_and_scrape(query)`.
    """
    @abstractmethod
    def search_and_scrape(self, query: str) -> List[Dict]:
        """
        Execute a search + retrieve textual content.
        Returns a list of dicts, e.g.:
          [
            {
              "title": str,
              "url": str,
              "body_snippet": str,
              "raw_text": str
            }, ...
          ]
        """
        pass


class DuckDuckGoSearchService(BaseWebSearchService):
    """
    A service to perform web search (with DuckDuckGo) and scrape results
    *without* using LangChain.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query,
                                    region='wt-wt',
                                    safesearch='moderate',
                                    timelimit=None,
                                    max_results=self.max_results):
                results.append(result)
        return results

    def _scrape_webpage(self, url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/58.0.3029.110 Safari/537.3"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

        soup = BeautifulSoup(resp.content, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # reduce multiple newlines
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return text

    def search_and_scrape(self, query: str) -> List[Dict]:
        raw_results = self._search_duckduckgo(query)
        output = []
        for item in raw_results:
            url = item.get("href", "")
            if not url:
                continue
            page_text = self._scrape_webpage(url)
            output.append({
                "title": item.get("title", ""),
                "url": url,
                "body_snippet": item.get("body", ""),
                "raw_text": page_text
            })
        return output


class LangChainWebSearchService(BaseWebSearchService):
    """
    A service to perform web search + scraping using LangChain utilities.
    Only works if LangChain is installed. Check `LANGCHAIN_AVAILABLE`.
    """

    def __init__(self, k: int = 5):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Cannot use LangChainWebSearchService.")
        self.k = k
        # Using a built-in DuckDuckGoSearchAPIWrapper
        self.searcher = DuckDuckGoSearchAPIWrapper()

    def search_and_scrape(self, query: str) -> List[Dict]:
        # The wrapper typically has a .run() method. 
        # For older langchain versions, check usage or do searcher._ddgr.run(...)
        # We'll assume .run(query) returns a list of {title, link, snippet}
        raw_results = self.searcher._ddgr.run(query, max_results=self.k)

        output = []
        for res in raw_results:
            url = res.get('link') or ''
            if not url.startswith('http'):
                continue
            # Load the webpage
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            combined_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)

            output.append({
                "title": res.get("title", ""),
                "url": url,
                "body_snippet": res.get("snippet", ""),
                "raw_text": combined_text
            })
        return output
