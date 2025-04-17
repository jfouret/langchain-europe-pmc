from typing import Type

from langchain_europe_pmc.retrievers import EuropePMCRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestEuropePMCRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[EuropePMCRetriever]:
        """Get an empty vectorstore for unit tests."""
        return EuropePMCRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"max_k": 2, "page_size": 1}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "malaria"
