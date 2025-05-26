import time
import os
import chainlit as cl
from langchain_core.language_models import BaseChatModel
from index import vectordb, embedding_model, collection_name
from typing import Annotated, TypedDict, Literal
from qdrant_client.models import FieldCondition, Filter, MatchValue


class ExtractedQuestion(TypedDict):
  intent: Annotated[Literal["general_information", "sparql_query"], "Intent extracted from the user question"]
  reformulated: Annotated[str, "Reformulated question adapted to semantic similarity search"]


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


@cl.on_chat_start
async def on_chat_start():
  """Initializes the chat session and LLM based on environment variable."""
  # Get provider from environment variable
  provider = os.environ.get("LLM_PROVIDER")
  if not provider:
    raise ValueError("LLM_PROVIDER environment variable must be set")
  
  global llm, structured_llm
  
  if provider == "mistral":
    llm = load_chat_model("mistral/mistral-large-latest")
  elif provider == "google":
    llm = load_chat_model("google/gemini-2.0-flash")
  elif provider == "ollama":
    llm = load_chat_model("ollama/mistral")
  else:
    raise ValueError(f"Unknown provider: {provider}")
      
  structured_llm = llm.with_structured_output(ExtractedQuestion)
  
  # Display initial message with example questions
  await cl.Message(content=f"Chat initialized with {provider} provider. Below are some example questions you can ask:\n"
    "- **Which tools can I use for comparative genomics?**\n"
    "- **Which is the best SIB tool for comparative genomics?**\n"
    "- **Which resources should I use to study the evolution of a protein?**\n"
    "- **What is the HGNC symbol for the P68871 protein?**\n"
    "- **Where is the ACE2 gene expressed in humans?**\n"
    "- **What are the rat orthologs of the human TP53 gene?**"
  ).send()


SYSTEM_PROMPT = """You are an assistant that helps users to navigate the resources and databases from the SIB Swiss Institute of Bioinformatics.

Depending on the user question and provided context, you may provide general information about the resources available at the SIB, or help the user to formulate a query to run on a SPARQL endpoint.

If answering with a SPARQL query:
Put the query inside a markdown codeblock with the `sparql` language tag, and always add the URL of the endpoint on which the query should be executed in a comment at the start of the query inside the codeblocks starting with "#+ endpoint: " (always only 1 endpoint).
Always answer with one query, if the answer lies in different endpoints, provide a federated query.
And briefly explain the query.

Here is a list of documents relevant to the user question that will help you answer the user question accurately:
{context}"""

EXTRACT_PROMPT = """Given a user question:
- Extract the intent of the question: either "sparql_query" (query available resources to answer biomedical questions), or "general_informations" (tools available, infos about the resources)
- Reformulate the question to make it more straigthforward and adapted to running a semantic similarity search"""

@cl.on_message
async def on_message(msg: cl.Message):
  """Main function to handle when user send a message to the assistant."""
  extracted: ExtractedQuestion = structured_llm.invoke([
    ("system", EXTRACT_PROMPT),
    *cl.chat_context.to_openai(), # Pass the whole chat history
  ])
  time.sleep(1) # To avoid quota limitations
  
  # Show extraction results
  async with cl.Step(name="extracted ⚗️") as step:
    step.output = extracted

  # Filter based on intent
  if extracted["intent"] == "general_information":
    query_filter = Filter(
      must=[FieldCondition(
        key="doc_type",
        match=MatchValue(value="General information"),
      )]
    )
  else:
    query_filter = Filter(
      must_not=[FieldCondition(
        key="doc_type",
        match=MatchValue(value="General information"),
      )]
    )

  # Get embeddings and query vectordb
  question_embeddings = next(iter(embedding_model.embed([extracted["reformulated"]])))
  retrieved_docs = vectordb.query_points(
    collection_name=collection_name,
    query=question_embeddings,
    query_filter=query_filter,
    limit=10,
  )
  
  # Format retrieved documents
  formatted_docs = ""
  for doc in retrieved_docs.points:
    if doc.payload.get("description"):
      formatted_docs += f"\n{doc.payload['description']}"
    else:
      formatted_docs += f"\n{doc.payload.get('question')}:\n\n```sparql\n#+ endpoint: {doc.payload.get('endpoint_url')}\n{doc.payload.get('answer')}\n```\n"

  # Show retrieved documents
  async with cl.Step(name=f"{len(retrieved_docs.points)} relevant documents 📚️") as step:
    step.output = formatted_docs

  answer = cl.Message(content="")
  for resp in llm.stream([
    ("system", SYSTEM_PROMPT.format(context=formatted_docs)),
    *cl.chat_context.to_openai(),
  ]):
    await answer.stream_token(resp.content)
    if resp.usage_metadata:
      print(resp.usage_metadata)
  await answer.send()


