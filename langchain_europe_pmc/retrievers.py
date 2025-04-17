"""EuropePMC retrievers."""

import requests
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field


class EuropePMCAPIWrapper(BaseModel):
    """Wrapper around Europe PMC API.

    This wrapper will use the Europe PMC API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    Parameters:
        max_k: maximum number of results to return in total.
          Default is 3.
        page_size: number of results to return per page.
          Default is 25. Maximum is 300.
        max_query_length: maximum length of the query.
          Default is 300 characters.
        doc_content_chars_max: maximum length of the document content.
          Content will be truncated if it exceeds this length.
          Default is 2000 characters.
        result_type: type of results to return (core, lite, oa, etc.)
          Default is "core".
        sort_criteria: field to sort results by.
          Default is None (sort by relevance).
          Options include: P_PDATE_D, AUTH_FIRST, FIRST_IDATE_D, CITED, etc.
        sort_direction: direction to sort results.
          Default is "desc".
          Options are "asc" or "desc".
    """

    base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    max_k: int = 3
    page_size: int = 25
    max_query_length: int = 300
    doc_content_chars_max: int = 2000
    result_type: str = "core"
    sort_criteria: Optional[str] = None
    sort_direction: str = "desc"
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )

    def run(self, query: str) -> str:
        """
        Run Europe PMC search and get the article meta information.
        It uses only the most informative fields of article meta information.
        """
        try:
            # Retrieve the top-k results for the query
            docs = [
                f"Title: {result.get('title', 'No title available')}\n"
                f"Authors: {result.get('authorString', 'Unknown authors')}\n"
                f"Journal: {result.get('journalTitle', 'Unknown journal')}, "
                f"{result.get('pubYear', 'Unknown year')}\n"
                f"PMID: {result.get('pmid', '')}\n"
                f"DOI: {result.get('doi', '')}\n"
                f"Abstract: {result.get('abstractText', 'No abstract available')}"
                for result in self.load(query[: self.max_query_length])
            ]

            # Join the results and limit the character count
            return (
                "\n\n".join(docs)[: self.doc_content_chars_max]
                if docs
                else "No good Europe PMC Result was found"
            )
        except Exception as ex:
            return f"Europe PMC exception: {ex}"

    def load(self, query: str) -> List[Dict[str, Any]]:
        """
        Search Europe PMC for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        """
        all_results = []
        current_url = self.base_url
        
        # Ensure page_size is within limits
        page_size = min(min(300, self.max_k), max(1, self.page_size))
        
        # Prepare initial parameters
        params = {
            "query": query,
            "format": "json",
            "resultType": self.result_type,
            "pageSize": page_size,
        }
        
        # Add sort parameter if sort_criteria is specified
        if self.sort_criteria:
            params["sort"] = f"{self.sort_criteria} {self.sort_direction}"
        
        # Fetch results with pagination until we have enough or there are no more
        while current_url and len(all_results) < self.max_k:
            try:
                # If it's the initial URL, use params; otherwise, the URL already has params
                if current_url == self.base_url:
                    response = requests.get(
                        current_url, params=params, headers=self.headers
                    )
                else:
                    response = requests.get(current_url, headers=self.headers)
                
                response.raise_for_status()
                result_json = response.json()
                
                # Extract results from this page
                if "resultList" in result_json and "result" in result_json["resultList"]:
                    page_results = result_json["resultList"]["result"]
                    all_results.extend(page_results)
                
                # Check if there's a next page
                next_page_url = result_json.get("nextPageUrl")
                if next_page_url:
                    current_url = next_page_url
                else:
                    break  # No more pages
                    
            except Exception as e:
                print(f"Error fetching results from Europe PMC: {str(e)}")
                break
        
        # Limit to max_k results
        return all_results[:self.max_k]

    def _dict2document(self, article: Dict[str, Any]) -> Document:
        """
        Convert an article dictionary to a Document object.
        """
        title = article.get("title", "No title available")
        authors = article.get("authorString", "Unknown authors")
        journal = article.get("journalTitle", "Unknown journal")
        year = article.get("pubYear", "Unknown year")
        pmid = article.get("pmid", "")
        doi = article.get("doi", "")
        abstract = article.get("abstractText", "No abstract available")
        
        # Create the page content
        page_content = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal}, {year}\n"
            f"PMID: {pmid}\n"
            f"DOI: {doi}\n"
            f"Abstract: {abstract}"
        )
        
        # Create metadata
        metadata = {
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "source": f"EuropePMC:{pmid}",
            "url": f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
        }
        
        return Document(page_content=page_content, metadata=metadata)

    def load_docs(self, query: str) -> List[Document]:
        """
        Search Europe PMC for documents matching the query.
        Return a list of Document objects.
        """
        articles = self.load(query)
        return [self._dict2document(article) for article in articles]


class EuropePMCRetriever(BaseRetriever, EuropePMCAPIWrapper):
    k: Optional[int] = None  # Exact number of documents to return
    """EuropePMC retriever.

    This retriever uses the Europe PMC API to search for scientific articles
    and returns them as Document objects.

    Setup:
        Install ``langchain-europe-pmc``.

        .. code-block:: bash

            pip install -U langchain-europe-pmc

    Key init args:
        max_k: int
            Maximum number of results to return in total. Default is 3.
        page_size: int
            Number of results to return per page. Default is 25. Maximum is 300.
        max_query_length: int
            Maximum length of the query. Default is 300 characters.
        doc_content_chars_max: int
            Maximum length of the document content. Default is 2000 characters.
        result_type: str
            Type of results to return (core, lite, oa, etc.). Default is "core".
        sort_criteria: Optional[str]
            Field to sort results by. Default is None (sort by relevance).
            Options include: P_PDATE_D, AUTH_FIRST, FIRST_IDATE_D, CITED, etc.
        sort_direction: str
            Direction to sort results. Default is "desc".
            Options are "asc" or "desc".

    Instantiate:
        .. code-block:: python

            from langchain_europe_pmc import EuropePMCRetriever

            retriever = EuropePMCRetriever(max_k=5)

    Usage:
        .. code-block:: python

            query = "CRISPR gene editing for cystic fibrosis"
            docs = retriever.invoke(query)
            
            for doc in docs:
                print(doc.page_content)
                print(doc.metadata)
                print("---")

        .. code-block:: none

            Title: CRISPR/Cas9-mediated gene editing in human zygotes using Cas9 protein
            Authors: Tang L, Zeng Y, Du H, Gong M, Peng J, Zhang B, Lei M, Zhao F, Wang W, Li X, Liu J
            Journal: Molecular Genetics and Genomics, 2017
            PMID: 28251317
            DOI: 10.1007/s00438-017-1299-z
            Abstract: Previous works using human tripronuclear zygotes suggested that the...
            {'title': 'CRISPR/Cas9-mediated gene editing in human zygotes using Cas9 protein', 'authors': 'Tang L, Zeng Y, Du H, Gong M, Peng J, Zhang B, Lei M, Zhao F, Wang W, Li X, Liu J', 'journal': 'Molecular Genetics and Genomics', 'year': '2017', 'pmid': '28251317', 'doi': '10.1007/s00438-017-1299-z', 'source': 'EuropePMC:28251317', 'url': 'https://europepmc.org/article/MED/28251317'}
            ---

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What are the latest advances in CRISPR gene editing for cystic fibrosis?")

        .. code-block:: none

            Based on the context provided, there have been several advances in CRISPR gene editing for cystic fibrosis. 
            Researchers have successfully used CRISPR/Cas9 technology to edit genes in human zygotes, 
            demonstrating the potential for genetic modification at the earliest stages of development. 
            This technology has shown promise for treating genetic disorders like cystic fibrosis by 
            potentially correcting the CFTR gene mutations responsible for the disease.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant to the query."""
        # Get the k parameter from kwargs or use the class attribute
        k = kwargs.get("k", self.k)
        
        # Get documents from the API
        docs = self.load_docs(query=query)
        
        # If k is specified, ensure we return exactly k documents
        if k is not None:
            # If we have more documents than k, truncate
            if len(docs) > k:
                docs = docs[:k]
            # If we have fewer documents than k, add empty documents
            elif len(docs) < k:
                empty_doc = Document(
                    page_content="",
                    metadata={
                        "title": "",
                        "authors": "",
                        "journal": "",
                        "year": "",
                        "pmid": "",
                        "doi": "",
                        "source": "",
                        "url": "",
                    }
                )
                docs.extend([empty_doc] * (k - len(docs)))
        
        return docs
