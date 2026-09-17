from groq import Groq

import config

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or a .env file."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate_response(messages, model: str = None) -> str:
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}"""
    client = _get_client()
    completion = client.chat.completions.create(
        model=model or config.GROQ_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content
