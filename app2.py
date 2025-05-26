import argparse
from langchain_core.language_models import BaseChatModel

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

def ask(question: str) -> None:
  for resp in llm.stream(question):
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

ask("Which tools can I use  for comparative genomics?")