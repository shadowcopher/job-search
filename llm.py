import os, time, json, urllib.request
from google import genai
from google.genai import errors, types

MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.8-flash-lite").split(",") if m.strip()]
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=45000),  # 45s per request
)


def _groq(prompt):
    # optional fallback; needs GROQ_API_KEY (free at console.groq.com)
    body = json.dumps({
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", body,
        {"Content-Type": "application/json",
         "Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def llm(prompt: str) -> str:
    for rnd in range(3):
        for m in MODELS:
            try:
                return client.models.generate_content(model=m, contents=prompt).text or ""
            except errors.APIError as e:
                print(f"  [llm] {m}: error {getattr(e, 'code', '?')}, trying next")
            except Exception as e:
                print(f"  [llm] {m}: {type(e).__name__}, trying next")
        if os.environ.get("GROQ_API_KEY"):
            try:
                return _groq(prompt)
            except Exception as e:
                print(f"  [llm] groq: {type(e).__name__}")
        time.sleep(4 * (rnd + 1))
    raise RuntimeError("All LLM providers failed")


if __name__ == "__main__":
    # run `python llm.py` to list the model names your key can use
    for m in client.models.list():
        print(m.name)