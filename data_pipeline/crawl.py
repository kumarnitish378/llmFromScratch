#!/usr/bin/env python3
"""Bounded recursive Crawl4AI collector with persistent SQLite queue and JSONL output."""
from __future__ import annotations
import argparse, asyncio, json, os, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, parse_qsl, urlencode, urlunparse
from urllib.robotparser import RobotFileParser
import yaml
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/"data_pipeline"/"crawl_config.yaml"
def canonical_url(raw):
    raw=urldefrag((raw or "").strip())[0]; p=urlparse(raw)
    if p.scheme not in ("http","https") or not p.hostname or p.username or p.password: return None
    q=urlencode([(k,v) for k,v in parse_qsl(p.query,keep_blank_values=True) if not k.lower().startswith("utm_") and k.lower() not in {"fbclid","gclid"}])
    return urlunparse((p.scheme.lower(),p.netloc.lower(),p.path or "/", "",q,""))
def allowed(url,domains):
    host=(urlparse(url).hostname or "").lower().rstrip(".")
    return any(host==d or host.endswith("."+d) for d in domains)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(DEFAULT_CONFIG)); ap.add_argument("--max-pages",type=int)
    ap.add_argument("--max-hours",type=float); ap.add_argument("--max-disk-gb",type=float)
    ap.add_argument("--max-ram-gb",type=float); ap.add_argument("--seed-file"); ap.add_argument("--resume",action="store_true")
    ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    out=(ROOT/cfg.get("output_dir","Data/crawl")).resolve(); out.mkdir(parents=True,exist_ok=True)
    domains={str(x).lower().strip().rstrip(".") for x in cfg.get("allowed_domains",[]) if str(x).strip()}
    if not domains: raise SystemExit("ERROR: allowed_domains is empty. Edit crawl_config.yaml.")
    seeds=list(cfg.get("seeds",[]))
    if a.seed_file: seeds += [x.strip() for x in Path(a.seed_file).read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    seeds=[canonical_url(x) for x in seeds]; seeds=[x for x in seeds if x and allowed(x,domains)]
    if not seeds: raise SystemExit("ERROR: no valid seeds within allowed_domains.")
    if a.dry_run:
        print("Dry run only. Domains:",sorted(domains)); print("Seeds:",*seeds,sep="\n  "); print("Output:",out); return
    max_pages=max(1,a.max_pages or int(cfg.get("max_pages",1000))); max_hours=max(.05,a.max_hours or float(cfg.get("max_hours",8)))
    max_disk=max(.1,a.max_disk_gb or float(cfg.get("max_disk_gb",20))); max_ram=max(1,a.max_ram_gb or float(cfg.get("max_ram_gb",6)))
    delay=max(0,float(cfg.get("request_delay_seconds",1.5))); max_depth=int(cfg.get("max_depth",3))
    db=sqlite3.connect(out/"queue.sqlite3"); db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS urls(url TEXT PRIMARY KEY,depth INTEGER,status TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,last_error TEXT,updated_at TEXT)")
    db.commit()
    for u in seeds: db.execute("INSERT OR IGNORE INTO urls(url,depth,status,updated_at) VALUES(?,0,'pending',?)",(u,datetime.now(timezone.utc).isoformat()))
    # Recover a process interrupted while a URL was marked working.
    db.execute("UPDATE urls SET status='pending' WHERE status='working'"); db.commit()
    raw_path=out/"raw_pages.jsonl"; robots={}
    def robot_allowed(url):
        p=urlparse(url); origin=f"{p.scheme}://{p.netloc}"
        if origin not in robots:
            rp=RobotFileParser(); rp.set_url(origin+"/robots.txt")
            try: rp.read()
            except Exception: pass
            robots[origin]=rp
        return robots[origin].can_fetch(str(cfg.get("user_agent","llmFromScratchResearchBot/1.0")),url)
    started=time.monotonic(); pages=0
    run_cfg=CrawlerRunConfig(cache_mode=CacheMode.BYPASS,page_timeout=30000,word_count_threshold=30)
    browser_cfg=BrowserConfig(headless=True,verbose=False)
    async def loop():
        nonlocal pages
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            while pages<max_pages and time.monotonic()-started<max_hours*3600:
                raw_bytes=raw_path.stat().st_size if raw_path.exists() else 0
                if raw_bytes>max_disk*1024**3: print("Disk limit reached."); break
                try:
                    import psutil
                    if psutil.Process().memory_info().rss>max_ram*1024**3: print("RAM limit reached."); break
                except ImportError: pass
                row=db.execute("SELECT url,depth,attempts FROM urls WHERE status='pending' ORDER BY depth,url LIMIT 1").fetchone()
                if not row: break
                url,depth,attempts=row
                db.execute("UPDATE urls SET status='working',attempts=attempts+1 WHERE url=?",(url,)); db.commit()
                try:
                    if not robot_allowed(url):
                        db.execute("UPDATE urls SET status='skipped',last_error='robots.txt disallows' WHERE url=?",(url,)); db.commit(); continue
                    result=await crawler.arun(url=url,config=run_cfg)
                    if not getattr(result,"success",False): raise RuntimeError(str(getattr(result,"error_message","crawl failed"))[:300])
                    md=getattr(result,"markdown",""); markdown=getattr(md,"fit_markdown",None) or getattr(md,"raw_markdown",None) or (md if isinstance(md,str) else "")
                    links=[]
                    for group in ("internal","external"):
                        for item in (getattr(result,"links",{}) or {}).get(group,[]) or []:
                            nxt=canonical_url(urljoin(url,item.get("href","") if isinstance(item,dict) else ""))
                            if nxt and allowed(nxt,domains): links.append(nxt)
                    record={"url":url,"title":getattr(result,"title","") or "","markdown":markdown or "","links":sorted(set(links)),"crawled_at":datetime.now(timezone.utc).isoformat(),"license_status":cfg.get("default_license_status","unknown"),"source_domain":urlparse(url).hostname,"depth":depth}
                    with raw_path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
                    if depth<max_depth:
                        for nxt in record["links"]: db.execute("INSERT OR IGNORE INTO urls(url,depth,status,updated_at) VALUES(?,?,'pending',?)",(nxt,depth+1,datetime.now(timezone.utc).isoformat()))
                    db.execute("UPDATE urls SET status='done',last_error=NULL WHERE url=?",(url,)); db.commit(); pages+=1
                    print(f"[{pages}/{max_pages}] depth={depth} chars={len(markdown or '')} {url}")
                    if delay: await asyncio.sleep(delay)
                except Exception as exc:
                    status="pending" if attempts<2 else "failed"
                    db.execute("UPDATE urls SET status=?,last_error=? WHERE url=?",(status,str(exc)[:500],url)); db.commit()
                    print("[WARN]",url,str(exc)[:250])
    try: asyncio.run(loop())
    finally: db.close()
    print(f"Stopped after {pages} pages. Queue is persistent at {out/'queue.sqlite3'}")
if __name__=="__main__": main()
