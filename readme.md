My solutions to the SIB course [Using Large Language Models for Biodata Exploration](https://github.com/sib-swiss/llm-biodata-training) - built upon the provided materials.

## Activate a conda env

> [!NOTE]
>
> ### Prerequisites
>
> For installation details see
> [llm-biodata-training](https://github.com/sib-swiss/llm-biodata-training).

First we should creare a conda env where to build a LLM-powered app
(optional):

``` {bash}
conda create -n <env> python=3.12
conda activate <env>
```

## Launch a LLM-powered app

For each example covered in the course, I developed a dedicated app. To
run an app:

``` {bash}
uv run --env-file <llm-api> appN.py -p <provider>
```

where:

- `llm-api` is the file containing the API key(s) for the LLM provider
  you want to use
- `N` is the app number (from 0 to 6)
- `p` is the provider to use: **mistral**, **google** or the pulled
  **ollama** model

> [!IMPORTANT]
>
> ### Search Index
>
> Before running `app{4-6}.py`, a semantic search index needs to be
> built:
>
> ``` {bash}
> uv run index.py
> ```

## Build a LLM-powered app with Chainlit

Finally I built an app using the Chainlit web UI that wraps-up the
examples of `app{0-6}.py`. To initialize the chat session with
**mistral**:

``` {bash}
LLM_PROVIDER=mistral uv run --env-file <llm-api> chainlit run app7.py
```

> [!IMPORTANT]
>
> ### Search Index
>
> A semantic search index must be built before launching `app7.py`:
>
> ``` {bash}
> uv run index.py
> ```

> [!TIP]
>
> ### Other Providers
>
> To initialize the chat session with other providers:
>
> ``` {bash}
> LLM_PROVIDER=google uv run --env-file <llm-api> chainlit run app7.py
> LLM_PROVIDER=ollama uv run --env-file <llm-api> chainlit run app7.py
> ```
