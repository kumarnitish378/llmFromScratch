#!/usr/bin/env python3
"""Explicit dispatcher. pipeline = crawl + process; it does not train the C++ Transformer."""
import argparse, importlib.util, sqlite3, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args):
    print("+"," ".join(map(str,args)),flush=True)
    return subprocess.run(args,cwd=ROOT).returncode
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke-test","crawl","process","pipeline","status"],required=True)
    p.add_argument("--max-hours",type=float); p.add_argument("--max-pages",type=int)
    p.add_argument("--max-disk-gb",type=float); p.add_argument("--max-ram-gb",type=float)
    p.add_argument("--seed-file"); p.add_argument("--resume",action="store_true"); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); py=sys.executable
    if a.mode=="smoke-test":
        for m in ("yaml","crawl4ai","psutil"):
            if importlib.util.find_spec(m) is None: raise SystemExit("Missing dependency: "+m)
        from data_pipeline.crawl import canonical_url,allowed
        assert canonical_url("https://Example.com/a#fragment")=="https://example.com/a"
        assert allowed("https://sub.example.com/a",{"example.com"})
        assert not allowed("https://example.org/a",{"example.com"})
        print("Smoke tests passed; no crawl or training started."); return 0
    if a.mode=="status":
        db=ROOT/"Data/crawl/queue.sqlite3"
        if not db.exists(): print("No crawl queue found."); return 0
        con=sqlite3.connect(db)
        for status,count in con.execute("SELECT status,COUNT(*) FROM urls GROUP BY status"): print(f"{status}: {count}")
        con.close(); raw=ROOT/"Data/crawl/raw_pages.jsonl"
        print("Raw records:",sum(1 for _ in raw.open(encoding="utf-8")) if raw.exists() else 0); return 0
    crawl=[py,"data_pipeline/crawl.py"]
    for key,val in [("--max-hours",a.max_hours),("--max-pages",a.max_pages),("--max-disk-gb",a.max_disk_gb),("--max-ram-gb",a.max_ram_gb),("--seed-file",a.seed_file)]:
        if val is not None: crawl += [key,str(val)]
    if a.resume: crawl += ["--resume"]
    if a.dry_run: crawl += ["--dry-run"]
    if a.mode=="crawl": return run(crawl)
    process=[py,"data_pipeline/process.py"]
    if a.mode=="process": return run(process)
    rc=run(crawl)
    return rc if rc else run(process)
if __name__=="__main__": raise SystemExit(main())
