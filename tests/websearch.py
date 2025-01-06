import asyncio
import logging
import sys
import os

# Adjust paths if needed
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from ragutils.services.web_search import (
    DuckDuckGoSearchService,
    LangChainWebSearchService
)
from workflow.web_search_indexing import WebSearchIndexingWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    query = "Hello from web search"

    # 1) Using the manual DuckDuckGo approach
    ddg_service = DuckDuckGoSearchService(max_results=3)
    ddg_workflow = WebSearchIndexingWorkflow(search_service=ddg_service)
    ddg_indexed_data = await ddg_workflow.search_and_index(query)
    logger.info(f"DuckDuckGo-based indexing complete. Indexed Data:\n{ddg_indexed_data}")

    # 2) Using the LangChain approach (only if installed)
    try:
        lc_service = LangChainWebSearchService(k=3)
        lc_workflow = WebSearchIndexingWorkflow(search_service=lc_service)
        lc_indexed_data = await lc_workflow.search_and_index(query)
        logger.info(f"LangChain-based indexing complete. Indexed Data:\n{lc_indexed_data}")
    except ImportError:
        logger.warning("LangChain is not installed, skipping the LangChain web search test.")

if __name__ == "__main__":
    asyncio.run(main())
