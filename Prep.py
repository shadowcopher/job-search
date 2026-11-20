import sqlite3
import pandas as pd

df = pd.read_csv("data/postings.csv")
print(df.columns.tolist())  # check these match what's below

wanted = ["title", "company_name", "location", "description",
          "formatted_experience_level", "work_type"]
cols = [c for c in wanted if c in df.columns]
df = df[cols].dropna(subset=["title", "description"])
df = df.drop_duplicates(subset=["title", "company_name"])
df = df.sample(n=min(4000, len(df)), random_state=42).reset_index(drop=True)

df["id"] = df.index
df["text"] = df["title"] + ". " + df["description"].str[:1500]

con = sqlite3.connect("data/jobs.db")
df.to_sql("jobs", con, if_exists="replace", index=False)
print(f"Saved {len(df)} jobs")