import httpx
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from sparql_llm import SparqlExamplesLoader, SparqlVoidShapesLoader

## general loader
# def load_resources_csv(url: str) -> list[Document]:
#   """Load resources from a CSV file and return a list of Document objects."""
#   resp = httpx.get(url, follow_redirects=True)
  
#   with open("tmp.csv", "w") as f:
#     f.write(resp.text)
  
#   docs = CSVLoader(file_path="tmp.csv", csv_args={"delimiter": ","}).load()
  
#   # Add page_content to metadata because we directly use the search engine lib instead of LangChain retriever abstraction
#   for doc in docs:
#     doc.metadata["description"] = doc.page_content
#   return docs

## custom loader for more accurate query matching
def load_resources_csv(url: str) -> list[Document]:
  """Load resources from a CSV file and return a list of Document objects."""
  df = pd.read_csv(url)
  docs: list[Document] = []
  for _, row in df.iterrows():
    description = f"[{row['title']}]({row['url']}) ({row['category']}): {row['description']}"
    page_content = f"{row['title']} {row['description']}"
    # Long description of the resource
    doc = Document(
      page_content=page_content,
      metadata={
        "iri": row["url"],
        "page_content": page_content,
        "description": description,
        "doc_type": "General information",
      }
    )
    docs.append(doc)
    # Ontology terms
    if row.get("ontology_terms"):
      page_content = f"{row['ontology_terms']}"
      doc = Document(
        page_content=page_content,
        metadata={
          "iri": row["url"],
          "page_content": page_content,
          "description": description,
          "doc_type": "General information",
        }
      )
      docs.append(doc)
  print(f"✅ {len(docs)} documents indexed from {url}")
  return docs

def load_sparql_endpoints() -> list[Document]:
  endpoints: list[str] = [
    "https://sparql.uniprot.org/sparql/",
    "https://www.bgee.org/sparql/",
    "https://sparql.omabrowser.org/sparql/",
    "https://sparql.rhea-db.org/sparql/",
  ]
  docs: list[Document] = []
  for endpoint in endpoints:
    print(f"\n  🔎 Getting metadata for {endpoint}")
    docs += SparqlExamplesLoader(endpoint).load()
    docs += SparqlVoidShapesLoader(endpoint).load()
  print(f"✅ {len(docs)} documents indexed from {len(endpoints)} endpoints")
  return docs

embedding_model = TextEmbedding(
  "BAAI/bge-small-en-v1.5",
  # providers=["CUDAExecutionProvider"], # To use GPUs, replace the fastembed dependency with fastembed-gpu
)
embedding_dimensions = 384 # Check the list of models to find a model dimensions
collection_name = "sib-biodata"
vectordb = QdrantClient(path="data/vectordb")

if __name__ == "__main__":
  docs = load_resources_csv("https://github.com/sib-swiss/sparql-llm/raw/refs/heads/main/src/expasy-agent/expasy_resources_metadata.csv")
  docs += load_sparql_endpoints()
  print(docs[0])

  if vectordb.collection_exists(collection_name):
    vectordb.delete_collection(collection_name)
  
  # Create the collection of embeddings
  vectordb.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
  )
  
  # Generate embeddings for each document
  embeddings = embedding_model.embed([q.page_content for q in docs])
  # Upload the embeddings in the collection
  vectordb.upload_collection(
    collection_name=collection_name,
    vectors=[embed.tolist() for embed in embeddings],
    payload=[doc.metadata for doc in docs],
  )
