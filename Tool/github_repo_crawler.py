#!/usr/bin/env python3
"""
GitHub repository crawler for building programming-language corpora.

Features:
- repo seed files and repo groups
- allow/deny repo filters
- recursive GitHub API traversal
- bounded corpus output files
- resume support via manifest
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


TOOL_DIR = Path("Tool")
DEFAULT_REPO_FILE = TOOL_DIR / "github_repos.txt"
DEFAULT_OUTPUT_DIR = Path("Data/github_crawled")
DEFAULT_MAX_FILES = 10
DEFAULT_MAX_FILE_SIZE_KB = 20 * 1024
DEFAULT_MAX_SOURCE_FILE_KB = 2048
DEFAULT_WORKERS = 8
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_TIMEOUT_SEC = 20
DEFAULT_MAX_DEPTH = 32
DEFAULT_MANIFEST_NAME = "github_repo_manifest.jsonl"
DEFAULT_MAX_CONSECUTIVE_AUTH_FAILURES = 3


def normalize_network_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path_part = urllib.parse.quote(parsed.path, safe="/%:@+~!$&'()*;,=-._")
    query_part = urllib.parse.quote(parsed.query, safe="=&%:@+~!$'()*;,/-._")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path_part, query_part, parsed.fragment))

GROUP_FILES = {
    "all": TOOL_DIR / "github_repos.txt",
    "python": TOOL_DIR / "github_repos_python.txt",
    "ml": TOOL_DIR / "github_repos_ml.txt",
    "systems": TOOL_DIR / "github_repos_systems.txt",
    "web": TOOL_DIR / "github_repos_web.txt",
    "data": TOOL_DIR / "github_repos_data.txt",
}

TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".py", ".pyi", ".java", ".kt", ".kts", ".scala", ".go", ".rs",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".php", ".rb",
    ".swift", ".m", ".mm", ".cs", ".fs", ".f90", ".jl", ".r", ".lua",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".html", ".htm", ".css", ".scss", ".less", ".xml", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".rst", ".tex", ".txt",
    ".proto", ".cmake", ".gradle",
}

IGNORE_DIRS = {
    ".git", ".github", ".idea", ".vscode", "node_modules", "vendor", "dist",
    "build", "target", "coverage", "__pycache__", "venv", ".venv", "third_party",
    "deps", "external", "generated", "out", "bin", "obj",
}


def load_dotenv_token() -> str:
    env_path = Path(".env")
    if not env_path.exists():
        return ""

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "GITHUB_TOKEN":
            return value.strip().strip(chr(34)).strip("'")
    return ""


@dataclasses.dataclass(frozen=True)
class RepoSpec:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclasses.dataclass(frozen=True)
class RepoFile:
    repo: RepoSpec
    branch: str
    path: str
    size: int
    download_url: str

    @property
    def key(self) -> str:
        return f"{self.repo.slug}:{self.branch}:{self.path}"


@dataclasses.dataclass
class CrawlerConfig:
    repo_files: List[Path]
    allow_repo_file: Optional[Path]
    deny_repo_file: Optional[Path]
    output_dir: Path
    manifest_path: Path
    max_files: int
    max_file_size_bytes: int
    max_source_file_bytes: int
    max_depth: int
    timeout_sec: int
    workers: int
    progress_every: int
    user_agent: str
    github_token: str
    resume: bool
    max_consecutive_auth_failures: int
    target_total_bytes: int


@dataclasses.dataclass
class CrawlStats:
    repos_seen: int = 0
    repos_completed: int = 0
    repos_skipped: int = 0
    directories_scanned: int = 0
    candidate_files: int = 0
    skipped_manifest: int = 0
    downloaded_files: int = 0
    saved_files: int = 0
    bytes_written: int = 0
    started_at: float = dataclasses.field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        return max(0.001, time.time() - self.started_at)


class GitHubClient:
    def __init__(self, timeout_sec: int, user_agent: str, github_token: str) -> None:
        self.timeout_sec = timeout_sec
        self.user_agent = user_agent
        self.github_token = github_token.strip()
        self.last_error = ""

    def _request(self, url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return urllib.request.Request(normalize_network_url(url), headers=headers)

    def fetch_json(self, url: str) -> Optional[object]:
        req = self._request(url)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                self.last_error = ""
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            self.last_error = f"HTTP {exc.code} for {url}"
            return None
        except urllib.error.URLError as exc:
            self.last_error = f"URL error for {url}: {exc.reason}"
            return None
        except TimeoutError:
            self.last_error = f"Timeout for {url}"
            return None
        except Exception as exc:
            self.last_error = f"Unexpected error for {url}: {exc}"
            return None
        except json.JSONDecodeError:
            self.last_error = f"Invalid JSON from {url}"
            return None

    def fetch_text(self, url: str) -> Optional[str]:
        req = self._request(url, accept="text/plain, application/octet-stream;q=0.1, */*;q=0.01")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                self.last_error = ""
                ctype = (resp.headers.get("Content-Type") or "").lower()
                data = resp.read()
                if not data:
                    return None
                charset = "utf-8"
                if "charset=" in ctype:
                    charset = ctype.split("charset=", 1)[1].split(";", 1)[0].strip()
                return data.decode(charset, errors="replace")
        except UnicodeDecodeError:
            self.last_error = f"Decode error for {url}"
            return None
        except urllib.error.HTTPError as exc:
            self.last_error = f"HTTP {exc.code} for {url}"
            return None
        except urllib.error.URLError as exc:
            self.last_error = f"URL error for {url}: {exc.reason}"
            return None
        except TimeoutError:
            self.last_error = f"Timeout for {url}"
            return None
        except Exception as exc:
            self.last_error = f"Unexpected error for {url}: {exc}"
            return None

    def get_default_branch(self, repo: RepoSpec) -> Optional[str]:
        payload = self.fetch_json(f"https://api.github.com/repos/{repo.slug}")
        if not isinstance(payload, dict):
            return None
        branch = payload.get("default_branch")
        return branch if isinstance(branch, str) and branch else None

    def list_directory(self, repo: RepoSpec, branch: str, path: str) -> List[dict]:
        encoded_path = urllib.parse.quote(path.strip("/"))
        url = f"https://api.github.com/repos/{repo.slug}/contents/{encoded_path}?ref={urllib.parse.quote(branch)}"
        payload = self.fetch_json(url)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []


class ResumeState:
    def __init__(self, manifest_path: Path, enabled: bool) -> None:
        self.manifest_path = manifest_path
        self.enabled = enabled
        self.entries: Set[str] = set()
        if self.enabled and self.manifest_path.exists():
            for raw in self.manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                if isinstance(key, str) and key:
                    self.entries.add(key)

    def contains(self, key: str) -> bool:
        return self.enabled and key in self.entries

    def add(self, key: str, repo_slug: str, branch: str, path: str) -> None:
        if not self.enabled or key in self.entries:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "key": key,
                "repo": repo_slug,
                "branch": branch,
                "path": path,
                "saved_at": int(time.time()),
            }, ensure_ascii=False) + "\n")
        self.entries.add(key)


class OutputPacker:
    def __init__(self, output_dir: Path, max_files: int, max_file_size_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_files = max_files
        self.max_file_size_bytes = max_file_size_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_parts: List[bytes] = []
        self.current_size = 0
        self.current_file_index = 0
        self.current_output_path: Optional[Path] = None
        self.saved_file_count = 0
        self.total_bytes_on_disk = 0
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        indexed_files = []
        for path in self.output_dir.glob("github_corpus_*.txt"):
            stem = path.stem
            try:
                index = int(stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            indexed_files.append((index, path, path.stat().st_size))

        indexed_files.sort(key=lambda item: item[0])
        self.saved_file_count = len(indexed_files)
        self.total_bytes_on_disk = sum(size for _, _, size in indexed_files)

        if not indexed_files:
            return

        last_index, last_path, last_size = indexed_files[-1]
        if self.saved_file_count <= self.max_files and last_size < self.max_file_size_bytes:
            self.current_file_index = last_index
            self.current_output_path = last_path
            self.current_size = last_size

    def _ensure_output_slot(self) -> bool:
        if self.current_output_path is not None:
            return True
        if self.saved_file_count >= self.max_files:
            return False
        self.current_file_index = self.saved_file_count + 1
        self.current_output_path = self.output_dir / f"github_corpus_{self.current_file_index:04d}.txt"
        self.current_size = 0
        return True

    def is_full(self) -> bool:
        if self.current_output_path is not None and self.current_size < self.max_file_size_bytes:
            return False
        return self.saved_file_count >= self.max_files

    def add_chunk(self, chunk_text: str) -> int:
        encoded = chunk_text.encode("utf-8", errors="replace")
        if not encoded:
            return 0

        written_total = 0
        offset = 0
        while offset < len(encoded):
            if self.current_size >= self.max_file_size_bytes:
                self.flush()

            if not self._ensure_output_slot():
                break

            remaining = self.max_file_size_bytes - self.current_size
            piece = encoded[offset: offset + remaining]
            while piece and (piece[-1] & 0b1100_0000) == 0b1000_0000:
                piece = piece[:-1]
            if not piece:
                break

            self.current_parts.append(piece)
            self.current_size += len(piece)
            written_total += len(piece)
            offset += len(piece)

            if self.current_size >= self.max_file_size_bytes:
                self.flush()

        return written_total

    def flush(self) -> bool:
        if not self.current_parts or self.current_output_path is None:
            return False

        write_mode = "ab" if self.current_output_path.exists() else "wb"
        written_now = sum(len(part) for part in self.current_parts)
        with self.current_output_path.open(write_mode) as handle:
            for part in self.current_parts:
                handle.write(part)

        if self.current_file_index > self.saved_file_count:
            self.saved_file_count = self.current_file_index

        self.total_bytes_on_disk += written_now
        self.current_parts = []

        if self.current_size >= self.max_file_size_bytes:
            self.current_output_path = None
            self.current_file_index = 0
            self.current_size = 0

        return True


def parse_repo_spec(value: str) -> Optional[RepoSpec]:
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("#"):
        return None

    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urllib.parse.urlsplit(cleaned)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            return None
        owner, name = parts[0], parts[1]
    else:
        parts = [part for part in cleaned.split("/") if part]
        if len(parts) != 2:
            return None
        owner, name = parts

    if name.endswith(".git"):
        name = name[:-4]
    return RepoSpec(owner=owner, name=name) if owner and name else None


def load_repo_specs(paths: Sequence[Path]) -> List[RepoSpec]:
    repos: List[RepoSpec] = []
    seen: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for part in raw_line.split(","):
                repo = parse_repo_spec(part)
                if repo is None or repo.slug in seen:
                    continue
                repos.append(repo)
                seen.add(repo.slug)
    return repos


def load_repo_slug_set(path: Optional[Path]) -> Set[str]:
    if path is None or not path.exists():
        return set()
    slugs: Set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for part in raw_line.split(","):
            repo = parse_repo_spec(part)
            if repo is not None:
                slugs.add(repo.slug)
    return slugs


def apply_repo_filters(repos: Sequence[RepoSpec], allow: Set[str], deny: Set[str]) -> List[RepoSpec]:
    filtered: List[RepoSpec] = []
    for repo in repos:
        if allow and repo.slug not in allow:
            continue
        if repo.slug in deny:
            continue
        filtered.append(repo)
    return filtered


def should_keep_file(path: str, size: int, max_source_file_bytes: int) -> bool:
    if size <= 0 or size > max_source_file_bytes:
        return False
    file_path = Path(path)
    if any(segment.lower() in IGNORE_DIRS for segment in file_path.parts[:-1]):
        return False
    suffix = file_path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or file_path.name.lower() == "dockerfile"


def discover_repo_files(
    client: GitHubClient,
    repo: RepoSpec,
    branch: str,
    max_depth: int,
    max_source_file_bytes: int,
    resume_state: ResumeState,
    stats: CrawlStats,
) -> List[RepoFile]:
    discovered: List[RepoFile] = []
    queue: List[Tuple[str, int]] = [("", 0)]
    cursor = 0

    while cursor < len(queue):
        path, depth = queue[cursor]
        cursor += 1
        entries = client.list_directory(repo, branch, path)
        stats.directories_scanned += 1

        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not isinstance(entry_path, str):
                continue

            if entry_type == "dir":
                name = Path(entry_path).name.lower()
                if name in IGNORE_DIRS or depth >= max_depth:
                    continue
                queue.append((entry_path, depth + 1))
                continue

            if entry_type != "file":
                continue

            size = entry.get("size")
            download_url = entry.get("download_url")
            if not isinstance(size, int) or not isinstance(download_url, str):
                continue
            if not should_keep_file(entry_path, size, max_source_file_bytes):
                continue

            repo_file = RepoFile(repo=repo, branch=branch, path=entry_path, size=size, download_url=download_url)
            if resume_state.contains(repo_file.key):
                stats.skipped_manifest += 1
                continue

            discovered.append(repo_file)
            stats.candidate_files += 1

    return discovered


def format_repo_chunk(repo_file: RepoFile, text: str) -> str:
    return "\n".join([
        "=" * 80,
        f"repo: {repo_file.repo.slug}",
        f"branch: {repo_file.branch}",
        f"path: {repo_file.path}",
        "=" * 80,
        "",
        text.rstrip(),
        "",
    ])


def download_repo_file(client: GitHubClient, repo_file: RepoFile) -> Tuple[RepoFile, Optional[str]]:
    return repo_file, client.fetch_text(repo_file.download_url)


def print_progress(stats: CrawlStats, packer: OutputPacker, pending_files: int, target_bytes: int) -> None:
    percent = 0.0 if target_bytes <= 0 else min(100.0, (stats.bytes_written / target_bytes) * 100.0)
    rate = stats.downloaded_files / stats.elapsed_seconds()
    sys.stdout.write(
        "\r"
        f"[GitHub] {percent:6.2f}% | repos {stats.repos_completed}/{stats.repos_seen}"
        f" | dirs {stats.directories_scanned} | candidates {stats.candidate_files}"
        f" | skipped {stats.skipped_manifest} | downloaded {stats.downloaded_files}"
        f" | saved {stats.saved_files}/{packer.max_files} | bytes {stats.bytes_written}/{target_bytes}"
        f" | pending {pending_files} | rate {rate:5.2f} files/s"
    )
    sys.stdout.flush()


def crawl_repositories(config: CrawlerConfig, repos: Sequence[RepoSpec]) -> int:
    client = GitHubClient(config.timeout_sec, config.user_agent, config.github_token)
    resume_state = ResumeState(config.manifest_path, config.resume)
    packer = OutputPacker(config.output_dir, config.max_files, config.max_file_size_bytes)
    stats = CrawlStats(repos_seen=len(repos), saved_files=packer.saved_file_count, bytes_written=packer.total_bytes_on_disk)
    target_bytes = config.target_total_bytes

    consecutive_auth_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        for repo in repos:
            if packer.is_full() or stats.bytes_written >= target_bytes:
                break

            branch = client.get_default_branch(repo)
            if not branch:
                stats.repos_skipped += 1
                stats.repos_completed += 1
                if client.last_error:
                    print(f"[GitHub][Skip] {repo.slug}: {client.last_error}", file=sys.stderr)
                    if client.last_error.startswith("HTTP 403"):
                        consecutive_auth_failures += 1
                        if not config.github_token:
                            print("[GitHub] Missing GITHUB_TOKEN or --github-token. GitHub API is rejecting unauthenticated requests.", file=sys.stderr)
                        else:
                            print("[GitHub] GitHub API returned 403. Token may be invalid, rate-limited, or missing required access.", file=sys.stderr)
                        if consecutive_auth_failures >= config.max_consecutive_auth_failures:
                            print(f"[GitHub] Stopping early after {consecutive_auth_failures} consecutive HTTP 403 responses.", file=sys.stderr)
                            break
                    else:
                        consecutive_auth_failures = 0
                continue

            consecutive_auth_failures = 0

            repo_files = discover_repo_files(
                client,
                repo,
                branch,
                config.max_depth,
                config.max_source_file_bytes,
                resume_state,
                stats,
            )
            pending = [executor.submit(download_repo_file, client, repo_file) for repo_file in repo_files]
            completed_in_repo = 0

            for future in concurrent.futures.as_completed(pending):
                if packer.is_full() or stats.bytes_written >= target_bytes:
                    break
                try:
                    repo_file, text = future.result()
                except Exception as exc:
                    print(f"[GitHub][FileSkip] Worker failed: {exc}", file=sys.stderr)
                    continue
                completed_in_repo += 1
                if text:
                    written = packer.add_chunk(format_repo_chunk(repo_file, text))
                    if written > 0:
                        stats.bytes_written += written
                        stats.downloaded_files += 1
                        resume_state.add(repo_file.key, repo_file.repo.slug, repo_file.branch, repo_file.path)
                        stats.saved_files = packer.saved_file_count
                        if stats.bytes_written >= target_bytes:
                            break

                if config.progress_every > 0 and completed_in_repo % config.progress_every == 0:
                    stats.saved_files = packer.saved_file_count
                    print_progress(stats, packer, max(0, len(pending) - completed_in_repo), target_bytes)

            stats.repos_completed += 1
            stats.saved_files = packer.saved_file_count
            print_progress(stats, packer, 0, target_bytes)

    if packer.current_parts and packer.flush():
        stats.saved_files = packer.saved_file_count

    print_progress(stats, packer, 0, target_bytes)
    sys.stdout.write("\n")
    if stats.directories_scanned == 0 and stats.candidate_files == 0:
        print("[GitHub] No repository contents were fetched. Check GitHub API access or set GITHUB_TOKEN.", file=sys.stderr)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl open-source GitHub repositories into bounded text corpus files.")
    parser.add_argument("--repo-file", action="append", default=[], help="Additional repo seed file. Can be passed multiple times.")
    parser.add_argument("--repo-group", action="append", choices=sorted(GROUP_FILES.keys()), default=[], help="Built-in repo group to include.")
    parser.add_argument("--allow-repo-file", default="", help="Optional allowlist repo file. If set, only listed repos are crawled.")
    parser.add_argument("--deny-repo-file", default="", help="Optional denylist repo file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where packed text files are written.")
    parser.add_argument("--manifest-path", default="", help="Resume manifest path. Defaults to output-dir/github_repo_manifest.jsonl.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Maximum number of output files.")
    parser.add_argument("--max-file-size-kb", type=int, default=DEFAULT_MAX_FILE_SIZE_KB, help="Maximum size of each output file in KB.")
    parser.add_argument("--max-source-file-kb", type=int, default=DEFAULT_MAX_SOURCE_FILE_KB, help="Maximum size of each downloaded source file in KB.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Maximum repository directory depth to recurse.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="HTTP timeout in seconds.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel download workers.")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Refresh progress after every N downloaded files.")
    parser.add_argument("--user-agent", default="llmFromScratch-github-crawler/1.0", help="HTTP User-Agent header.")
    parser.add_argument("--github-token", default="", help="Optional GitHub token. Defaults to GITHUB_TOKEN env var or .env file.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume manifest checks.")
    parser.add_argument("--max-consecutive-auth-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_AUTH_FAILURES, help="Stop early after this many consecutive HTTP 403 repo failures.")
    parser.add_argument("--target-total-size-mb", type=int, default=0, help="Optional total corpus target size in MB. Overrides max-files*max-file-size-kb for stopping/progress if set.")
    return parser


def validate_config(args: argparse.Namespace) -> CrawlerConfig:
    repo_files: List[Path] = []
    if not args.repo_file and not args.repo_group:
        repo_files.append(DEFAULT_REPO_FILE)
    else:
        for group in args.repo_group:
            repo_files.append(GROUP_FILES[group])
        for path in args.repo_file:
            repo_files.append(Path(path))

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / DEFAULT_MANIFEST_NAME

    return CrawlerConfig(
        repo_files=repo_files,
        allow_repo_file=Path(args.allow_repo_file) if args.allow_repo_file else None,
        deny_repo_file=Path(args.deny_repo_file) if args.deny_repo_file else None,
        output_dir=output_dir,
        manifest_path=manifest_path,
        max_files=max(1, args.max_files),
        max_file_size_bytes=max(1, args.max_file_size_kb) * 1024,
        max_source_file_bytes=max(1, args.max_source_file_kb) * 1024,
        max_depth=max(0, args.max_depth),
        timeout_sec=max(1, args.timeout_sec),
        workers=max(1, args.workers),
        progress_every=max(1, args.progress_every),
        user_agent=args.user_agent,
        github_token=args.github_token or os.environ.get("GITHUB_TOKEN", "") or load_dotenv_token(),
        resume=not args.no_resume,
        max_consecutive_auth_failures=max(1, args.max_consecutive_auth_failures),
        target_total_bytes=(max(1, args.target_total_size_mb) * 1024 * 1024) if args.target_total_size_mb > 0 else max(1, args.max_files) * max(1, args.max_file_size_kb) * 1024,
    )


def main() -> int:
    config = validate_config(build_arg_parser().parse_args())
    repos = load_repo_specs(config.repo_files)
    repos = apply_repo_filters(
        repos,
        load_repo_slug_set(config.allow_repo_file),
        load_repo_slug_set(config.deny_repo_file),
    )
    if not repos:
        print("No repository seeds found after applying filters.", file=sys.stderr)
        return 1
    return crawl_repositories(config, repos)


if __name__ == "__main__":
    raise SystemExit(main())
