# Data Download Tools

## OpenWebText2 Downloader

Script: `Tool/download_openwebtext2.py`

Source references:
- https://openwebtext2.readthedocs.io/en/latest/

### What it does
- Downloads OpenWebText2 data using official links from the docs.
- Supports variants:
  - `clean` -> `openwebtext2.jsonl.zst.tar`
  - `raw` -> `openwebtext2_raw.tar`
- Shows progress while downloading.
- Optionally extracts downloaded tar files.

### Usage

```bash
python Tool/download_openwebtext2.py --variant clean
```

```bash
python Tool/download_openwebtext2.py --variant raw --extract
```

Optional output folder:

```bash
python Tool/download_openwebtext2.py --variant clean --output-dir Data/openwebtext2 --extract
```
