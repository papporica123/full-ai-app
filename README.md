# Research Assistant

A Streamlit AI research assistant backed by Groq. It optionally searches the
web (no search API key needed) and asks the model to ground its answer in
those results, citing sources.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
streamlit run main.py
```

Get a free Groq API key at https://console.groq.com/keys.

## Project layout

- `main.py` — Streamlit UI and chat loop
- `groq_client.py` — thin wrapper around the Groq chat completions API
- `search.py` — free web search used to ground answers
- `config.py` — environment/config loading
