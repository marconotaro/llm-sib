import argparse
from langchain_core.language_models import BaseChatModel
from index import vectordb, embedding_model, collection_name

parser = argparse.ArgumentParser(
  description="""
  - run: uv run --env-file <llm-api> appNN.py -p <provider>
  - help: uv run --env-file <llm-api> appNN.py --help
  """,
  formatter_class = argparse.RawDescriptionHelpFormatter
)

parser.add_argument("-p", "--provider", required = True, 
                    help = "provider to use. It can be mistral or google or the pulled ollama model")

args = parser.parse_args()

def load_chat_model(model: str) -> BaseChatModel:
  provider, model_name = model.split("/", maxsplit=1)
  if provider == "mistral":
    # https://python.langchain.com/docs/integrations/chat/mistralai/
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
      model=model_name,
      temperature=0,
      max_retries=2,
      random_seed=42,
    )
  elif provider == "google":
    # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
      model=model_name,
      temperature=0,
      max_retries=2,
      model_kwargs={"random_seed": 42},
    )
  elif provider == "ollama":
    # https://python.langchain.com/docs/integrations/chat/ollama/
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_name, temperature=0)

SYSTEM_PROMPT = """You are an assistant that helps users to navigate the resources and databases from the SIB Swiss Institute of Bioinformatics.

Depending on the user question and provided context, you may provide general information about the resources available at the SIB, or help the user to formulate a query to run on a SPARQL endpoint.

If answering with a SPARQL query:
Put the query inside a markdown codeblock with the `sparql` language tag, and always add the URL of the endpoint on which the query should be executed in a comment at the start of the query inside the codeblocks starting with "#+ endpoint: " (always only 1 endpoint).
Always answer with one query, if the answer lies in different endpoints, provide a federated query.
And briefly explain the query.

Here is a list of documents relevant to the user question that will help you answer the user question accurately:
{context}"""

def ask(question: str):
  # Generate embeddings for the user question
  question_embeddings = next(iter(embedding_model.embed([question])))
  
  # Find similar embeddings in the vector database
  retrieved_docs = vectordb.query_points(
    collection_name=collection_name,
    query=question_embeddings,
    limit=10,
  )
  print(f"📚️ Retrieved {len(retrieved_docs.points)} documents")
  formatted_docs = ""
  for doc in retrieved_docs.points:
    if doc.payload.get("description"):
      formatted_docs += f"\n{doc.payload['description']}"
    else:
      formatted_docs += f"\n{doc.payload.get('question')}:\n\n```sparql\n#+ endpoint: {doc.payload.get('endpoint_url')}\n{doc.payload.get('answer')}\n```\n"
  messages = [
    ("system", SYSTEM_PROMPT.format(context=formatted_docs)),
    ("human", question),
  ]
  for resp in llm.stream(messages):
    print(resp.content, end="")
    if resp.usage_metadata:
      print(f"\n\n{resp.usage_metadata}")

## call
if args.provider == "mistral":
  llm = load_chat_model("mistral/mistral-large-latest")
elif args.provider == "google":
  llm = load_chat_model("google/gemini-2.0-flash")
elif args.provider == "olama":
  llm = load_chat_model("ollama/mistral")
else:
  raise ValueError(f"Unknown provider: {args.provider}")

## call
ask("What is the HGNC symbol for the protein P68871?")
print("\n------\n")
ask("Where is the ACE2 gene expressed in humans?")
print("\n------\n")
ask("What are the rat orthologs of the human TP53 gene?")

