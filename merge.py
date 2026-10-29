"""Merge results/<ds>_part*.json into results/<ds>.json."""
import glob, json, sys
for ds in sys.argv[1:]:
    parts = [json.load(open(f)) for f in sorted(glob.glob(f"results/{ds}_part*.json"))]
    out = parts[0]
    for p in parts[1:]:
        out["runs"] += p["runs"]; out["noise"] += p["noise"]
        if "dual" in p: out["dual"] = p["dual"]
    out["runs"].sort(key=lambda r: r["seed"])
    json.dump(out, open(f"results/{ds}.json", "w")); print(ds, "seeds", [r["seed"] for r in out["runs"]])
