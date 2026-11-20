import json
d = json.load(open("labels.json"))
d.pop("work with large language models", None)
json.dump(d, open("labels.json", "w"), indent=2)
print("removed; remaining:", len(d))