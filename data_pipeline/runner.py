#!/usr/bin/env python3
"""Explicit dispatcher; train mode runs the existing C++ n-gram chat trainer, not Transformer training."""
import argparse, importlib.util, sqlite3, subprocess, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args):
    print("+"," ".join(map(str,args)),flush=True)
    return subprocess.run(args,cwd=ROOT).returncode
def train_chat():
    corpus=ROOT/"Data"/"clean_training_corpus.txt"
    if not corpus.exists() or corpus.stat().st_size==0:
        print("No clean_training_corpus.txt found. Run process first."); return 2
    build=shutil.which("make") or shutil.which("mingw32-make")
    if not build:
        print("make/mingw32-make not found. Install it or build the C++ project first."); return 2
    rc=run([build])
    if rc: return rc
    exe=ROOT/"build"/"app.exe"
    if not exe.exists():
        print("Build completed but build/app.exe was not found."); return 2
    print("Training the existing corpus n-gram chat baseline for 1 epoch over all lines; this is NOT Transformer training.")
    return subprocess.run([str(exe)],cwd=ROOT,input="4\n1\n0\n",text=True).returncode
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke-test","crawl","process","train","pipeline","status"],required=True)
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
    if a.mode=="train": return train_chat()
    crawl=[py,"data_pipeline/crawl.py"]
    for key,val in [("--max-hours",a.max_hours),("--max-pages",a.max_pages),("--max-disk-gb",a.max_disk_gb),("--max-ram-gb",a.max_ram_gb),("--seed-file",a.seed_file)]:
        if val is not None: crawl += [key,str(val)]
    if a.resume: crawl += ["--resume"]
    if a.dry_run: crawl += ["--dry-run"]
    if a.mode=="crawl": return run(crawl)
    process=[py,"data_pipeline/process.py"]
    if a.mode=="process": return run(process)
    rc=run(crawl)
    if rc: return rc
    rc=run(process)
    return rc if rc else train_chat()
if __name__=="__main__": raise SystemExit(main())
