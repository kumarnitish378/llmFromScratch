#!/usr/bin/env python3
"""Streaming cleaning and exact deduplication backed by SQLite."""
import argparse, hashlib, json, re, sqlite3
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
URL_RE=re.compile(r"https?://\S+|www\.\S+",re.I)
CONTROL_RE=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
def clean(text):
    text=CONTROL_RE.sub(" ",(text or "").replace("\r\n","\n").replace("\r","\n"))
    text=re.sub(r"\[\s*\d+\s*\]"," ",text); text=URL_RE.sub(" ",text)
    out=[]
    for line in text.splitlines():
        line=re.sub(r"[ \t]+"," ",line).strip()
        if len(line)<25 or line.lower() in {"home","menu","search","log in","sign up","cookie policy"}: continue
        out.append(line)
    return "\n".join(out).strip()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=str(ROOT/"data_pipeline"/"crawl_config.yaml")); ap.add_argument("--input"); ap.add_argument("--output"); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}; base=(ROOT/cfg.get("output_dir","Data/crawl")).resolve()
    inp=Path(a.input).resolve() if a.input else base/"raw_pages.jsonl"; out=Path(a.output).resolve() if a.output else base/"clean_pages.jsonl"
    out.parent.mkdir(parents=True,exist_ok=True); db=sqlite3.connect(base/"dedup.sqlite3"); db.execute("CREATE TABLE IF NOT EXISTS hashes(sha TEXT PRIMARY KEY)"); db.execute("DELETE FROM hashes"); db.commit()
    corpus_path=ROOT/"Data"/"clean_training_corpus.txt"; corpus_path.parent.mkdir(parents=True,exist_ok=True)
    kept=dupes=short=0
    with inp.open(encoding="utf-8") as src, out.open("w",encoding="utf-8") as dst, corpus_path.open("w",encoding="utf-8") as corpus:
        for line in src:
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            text=clean(row.get("markdown",""))
            if len(text)<200: short+=1; continue
            digest=hashlib.sha256(re.sub(r"\s+"," ",text.lower()).encode("utf-8")).hexdigest()
            try: db.execute("INSERT INTO hashes(sha) VALUES(?)",(digest,))
            except sqlite3.IntegrityError: dupes+=1; continue
            record={k:row.get(k) for k in ("url","title","crawled_at","license_status","source_domain","depth")}; record["text"]=text
            dst.write(json.dumps(record,ensure_ascii=False)+"\n"); corpus.write(text+"\n\n"); kept+=1
            if kept%100==0: db.commit(); print(f"kept={kept} duplicates={dupes} short={short}")
    db.commit(); db.close(); print(f"Done: kept={kept}, duplicates={dupes}, too_short={short}, output={out}")
    print(f"Plain-text training corpus: {corpus_path}")
if __name__=="__main__": main()
