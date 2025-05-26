import argparse

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

if args.provider == "mistral":
  from langchain_mistralai import ChatMistralAI
  llm = ChatMistralAI(
    model="mistral-large-latest",  # switch to small in case of quota limitations
    # model="mistral-medium-2505", # smaller model mainly for development
    temperature=0,                 # 0 for deterministic output
    max_retries=2,                 # number of retries in case of error
    random_seed=42,                # random seed for reproducibilit
  )
  resp = llm.invoke("Which tools can I use for comparative genomics?")
  print(resp)
elif args.provider == "google":
  from langchain_google_genai import ChatGoogleGenerativeAI
  llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",          # switch to small in case of quota limitations
    temperature=0,                     # 0 for deterministic output
    max_retries=2,                     # number of retries in case of error
    model_kwargs={"random_seed": 42},  # Pass random_seed
  )
  resp = llm.invoke("Which tools can I use for comparative genomics?")
  print(resp)
elif args.provider == "ollama":
  # https://python.langchain.com/docs/integrations/chat/ollama/
  from langchain_ollama import ChatOllama
  llm = ChatOllama(model="ollama/mistral", temperature=0)
  resp = llm.invoke("Which tools can I use for comparative genomics?")
  print(resp)
else:
  raise ValueError(f"Unknown provider: {args.provider}")
