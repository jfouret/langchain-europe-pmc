# langchain-europe-pmc

This package contains the LangChain integration with EuropePMC, providing a retriever for searching and retrieving scientific articles from Europe PMC.

## Installation

```bash
pip install -U langchain-europe-pmc
```

## Retrievers

`EuropePMCRetriever` class allows you to search and retrieve scientific articles from Europe PMC.

```python
from langchain_europe_pmc import EuropePMCRetriever

# Initialize the retriever with default parameters
retriever = EuropePMCRetriever()

# Or customize the retriever
retriever = EuropePMCRetriever(
    top_k_results=5,  # Return 5 results instead of the default 3
    result_type="core"  # Use the core result type
)

# Search for articles
docs = retriever.invoke("CRISPR gene editing for cystic fibrosis")

# Print the results
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
    print("---")
```

## Using the Retriever in a Chain

You can use the retriever in a LangChain chain to answer questions based on the retrieved documents:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Initialize the retriever
retriever = EuropePMCRetriever(top_k_results=3)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
response = chain.invoke("What are the latest advances in CRISPR gene editing for cystic fibrosis?")
print(response)
```

## Document Metadata

Each document returned by the retriever includes metadata such as the title, authors, journal, year, PMID, DOI, and URL:

```python
# Initialize the retriever
retriever = EuropePMCRetriever(top_k_results=1)

# Search for articles
docs = retriever.invoke("Alzheimer's disease")

# Print the metadata of the first document
if docs:
    doc = docs[0]
    print("Document Metadata:")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")
    
    # Print the URL to access the article
    print(f"\nAccess the article at: {doc.metadata.get('url', 'URL not available')}")

## Acknowledgments

This package is based on the work of the [LangChain](https://github.com/langchain-ai/langchain) project.

The repository skeleton was generated using `langchain-cli`.
