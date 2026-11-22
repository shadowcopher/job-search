import json, os, re
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter
from search import search, df
from llm import llm

PROFILE = "profile.json"
STATS = {"steps": 0, "invalid": 0}  # for your README: how often the LLM broke the format

SKILLS = ["python", "java", "javascript", "typescript", "react", "node", "sql", "mongodb",
          "postgresql", "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
          "machine learning", "deep learning", "nlp", "llm", "pytorch", "tensorflow",
          "pandas", "spark", "tableau", "power bi", "excel", "django", "flask", "fastapi",
          "spring", "golang", "c++", "rest", "api", "ci/cd", "terraform", "android", "kotlin"]


# ---- structured actions the LLM may produce (validated by Pydantic) ----
class SearchJobs(BaseModel):
    action: Literal["search_jobs"]
    query: str
    k: int = Field(5, ge=1, le=10)

class SkillGap(BaseModel):
    action: Literal["skill_gap"]
    job_id: int

class UpdateProfile(BaseModel):
    action: Literal["update_profile"]
    skills: list[str] = []
    target_role: str = ""

class Final(BaseModel):
    action: Literal["final"]
    answer: str

Step = Annotated[Union[SearchJobs, SkillGap, UpdateProfile, Final], Field(discriminator="action")]
adapter = TypeAdapter(Step)

SYSTEM = """You are a career assistant that helps job seekers find jobs.
Reply with exactly ONE JSON object and nothing else. Allowed objects:
{"action":"search_jobs","query":"...","k":5}
{"action":"skill_gap","job_id":123}
{"action":"update_profile","skills":["python"],"target_role":"backend developer"}
{"action":"final","answer":"message to the user"}
Rules: use search_jobs before recommending jobs. Use skill_gap when the user asks what
they are missing for a job. Save skills or target role the user mentions with update_profile.
Only mention jobs that appeared in tool results. Finish with "final"."""


# ---- memory ----
def load_profile():
    if os.path.exists(PROFILE):
        return json.load(open(PROFILE))
    return {"skills": [], "target_role": ""}

def save_profile(p):
    json.dump(p, open(PROFILE, "w"), indent=2)


# ---- tools ----
def skills_in(text):
    t = text.lower()
    return {s for s in SKILLS if re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", t)}

def run_tool(step, profile):
    if isinstance(step, SearchJobs):
        res = search(step.query, "hybrid", step.k)
        return json.dumps([{k: r[k] for k in ("id", "title", "company", "location")} for r in res])
    if isinstance(step, SkillGap):
        rows = df[df["id"] == step.job_id]
        if rows.empty:
            return "error: no job with that id"
        need = skills_in(rows.iloc[0]["text"])
        have = {s.lower() for s in profile["skills"]}
        return json.dumps({"job": rows.iloc[0]["title"], "matching": sorted(need & have),
                           "missing": sorted(need - have)})
    if isinstance(step, UpdateProfile):
        profile["skills"] = sorted(set(profile["skills"]) | {s.lower() for s in step.skills})
        if step.target_role:
            profile["target_role"] = step.target_role
        save_profile(profile)
        return "profile updated: " + json.dumps(profile)


# ---- agent loop ----
def extract_json(raw):
    s, e = raw.find("{"), raw.rfind("}")
    return json.loads(raw[s:e + 1])

def get_step(prompt):
    raw = llm(prompt)
    for attempt in range(2):
        STATS["steps"] += 1
        try:
            return adapter.validate_python(extract_json(raw))
        except ValueError as e:  # covers bad JSON and Pydantic validation errors
            STATS["invalid"] += 1
            if attempt == 0:
                raw = llm(prompt + f"\nYour last reply was invalid ({str(e)[:200]}). "
                                   "Reply with ONE valid JSON object only.")
    return None

def turn(user_msg, history):
    transcript = f"{history}USER: {user_msg}\n"
    for _ in range(6):  # max tool calls per turn
        profile = load_profile()
        prompt = f"{SYSTEM}\n\nUSER PROFILE: {json.dumps(profile)}\n\n{transcript}"
        step = get_step(prompt)
        if step is None:
            return "Sorry, I couldn't produce a valid action. Try rephrasing."
        if isinstance(step, Final):
            return step.answer
        result = run_tool(step, profile)
        print(f"  [tool] {step.model_dump_json()}")
        transcript += f"ASSISTANT ACTION: {step.model_dump_json()}\nTOOL RESULT: {result}\n"
    return "I couldn't finish in time. Try a more specific question."


if __name__ == "__main__":
    history = ""
    print("Career agent. Type 'quit' to exit.")
    while True:
        msg = input("\nyou> ").strip()
        if msg.lower() in ("", "quit", "exit"):
            break
        try:
            ans = turn(msg, history)
        except Exception as e:
            ans = f"The AI service is busy right now ({type(e).__name__}). Please try again."
        print("\nagent>", ans)
        history += f"USER: {msg}\nASSISTANT: {ans}\n"
    print(f"\nLLM calls: {STATS['steps']}, invalid replies: {STATS['invalid']}")