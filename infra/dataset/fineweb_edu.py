# FineWeb-Edu acquisition: pinned-revision download of the raw parquet
# shards from HuggingFace. Tokenization is a separate stage; this module
# only materialises source files on disk.

from pathlib import Path

REPO_ID = "HuggingFaceFW/fineweb-edu"

# main as of 2026-08-27. The repo keeps growing (new CC dumps), so main
# moves; pinning makes the shard list a pure function of (sample, count)
# and lets manifests record an exact source.
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

# Sizes at REVISION: 10BT = 14 shards / 28.5 GB, 100BT = 140 / 286 GB.
# Every shard is a self-contained parquet of ~2.15 GB holding ~0.71B
# GPT-2 tokens (the "BT" counts are GPT-2 counts; mistral32k yields
# ~10-15% more tokens on the same text).
SAMPLES = ("10BT", "100BT", "350BT")


def _default_dest() -> Path:
    # <repo-root>/data: git-ignored, a real directory (since 2026-08-29
    # data lives beside the repo, not behind a scratch-fs symlink — raven
    # /ptmp purges cost us nothing but raw parquet, which is disposable).
    # Valid under the editable install only, which is how this repo is
    # used.
    return Path(__file__).resolve().parents[2] / "data" / "fineweb-edu"


def download(
        sample: str = "10BT",
        file_count: int | None = None,
        dest: str | Path | None = None,
    ) -> list[Path]:
    '''
    Download the first `file_count` parquet shards (in sorted name order)
    of sample/<sample> at REVISION into `dest`, and return their local
    paths. The sample sets are random subsamples of the full corpus and
    each shard is self-contained, so a shard prefix is a valid smaller
    corpus. file_count=None means the whole sample set.

    Idempotent: complete shards are skipped, interrupted ones resume, so
    re-running after a crash is always safe. When `dest` already holds
    enough shards the call returns without touching the network — which
    relies on this function owning the directory (it only ever downloads
    name-sorted prefixes; don't drop other parquet files in it).
    '''
    assert sample in SAMPLES

    dest = Path(dest) if dest is not None else _default_dest()
    have = sorted((dest / "sample" / sample).glob("*.parquet"))
    if file_count is not None and len(have) >= file_count:
        return have[:file_count]

    # hub access is prep-only; keep the import lazy so infra.dataset stays
    # importable in the bare training env (MPCDF module: torch/numpy only)
    from huggingface_hub import HfApi, snapshot_download

    names = sorted(
        entry.path
        for entry in HfApi().list_repo_tree(
            REPO_ID, f"sample/{sample}", repo_type="dataset", revision=REVISION)
        if entry.path.endswith(".parquet")
    )
    if file_count is not None:
        if file_count > len(names):
            raise ValueError(f"sample/{sample} has only {len(names)} shards.")
        names = names[:file_count]

    snapshot_download(
        REPO_ID, repo_type="dataset", revision=REVISION,
        local_dir=dest, allow_patterns=names,
    )
    return [dest / name for name in names]
