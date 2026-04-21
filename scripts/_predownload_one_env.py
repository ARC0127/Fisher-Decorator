#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADER = ["env_name", "status", "frame_stack", "train_size", "val_size", "note"]

ADROIT_ENVS = {
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
}

ADROIT_BASE_URL = "https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1"


def normalize_tsv(tsv_path: Path):
    if not tsv_path.exists():
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
            writer.writeheader()
        return

    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({k: r.get(k, "") for k in HEADER})

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def upsert_status(tsv_path: Path, row: dict):
    normalize_tsv(tsv_path)

    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("env_name") != row["env_name"]:
                rows.append({k: r.get(k, "") for k in HEADER})

    rows.append({k: row.get(k, "") for k in HEADER})

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def download_stream(url: str, dst: Path, chunk_size: int = 8 * 1024 * 1024):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if dst.exists():
        return "exists"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)
    return "downloaded"


def is_ogbench_env(env_name: str) -> bool:
    return ("singletask" in env_name) or env_name.startswith("visual-")


def is_adroit_env(env_name: str) -> bool:
    return env_name in ADROIT_ENVS


def ogbench_base_name(env_name: str) -> str:
    return re.sub(r"-singletask-task\d+-v0$", "-v0", env_name)


def handle_ogbench(env_name: str):
    data_dir = Path(os.environ["OGBENCH_DATA_DIR"])
    base = ogbench_base_name(env_name)
    root = "https://rail.eecs.berkeley.edu/datasets/ogbench"

    train_name = f"{base}.npz"
    val_name = f"{base}-val.npz"

    train_path = data_dir / train_name
    val_path = data_dir / val_name

    s1 = download_stream(f"{root}/{train_name}", train_path)
    s2 = download_stream(f"{root}/{val_name}", val_path)

    note = f"ogbench_direct:{s1},{s2}"
    print(f"[OK] {env_name} -> {train_path.name}, {val_path.name}")
    return "", "", note


def handle_adroit_direct(env_name: str):
    data_dir = Path(os.environ["D4RL_DATASET_DIR"])
    filename = f"{env_name}.hdf5"
    dst = data_dir / filename
    url = f"{ADROIT_BASE_URL}/{filename}"
    status = download_stream(url, dst)
    note = f"adroit_direct:{status}:{filename}"
    print(f"[OK] {env_name} -> {filename}")
    return "", "", note


def handle_d4rl_non_adroit(env_name: str):
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    import gym
    import d4rl  # noqa: F401

    spec = gym.spec(env_name)

    kwargs = getattr(spec, "kwargs", None)
    if kwargs is None:
        kwargs = getattr(spec, "_kwargs", None)
    if kwargs is None:
        raise RuntimeError(f"Cannot access spec kwargs for {env_name}")

    dataset_url = kwargs.get("dataset_url", None)
    if dataset_url is None:
        raise RuntimeError(f"No dataset_url found in gym spec for {env_name}")

    data_dir = Path(os.environ["D4RL_DATASET_DIR"])
    filename = dataset_url.rstrip("/").split("/")[-1]
    dst = data_dir / filename

    status = download_stream(dataset_url, dst)
    note = f"d4rl_direct:{status}:{filename}"
    print(f"[OK] {env_name} -> {filename}")
    return "", "", note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--status_file", required=True)
    ap.add_argument("--frame_stack", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OGBENCH_DATA_DIR", "/root/autodl-tmp/rl_datasets/ogbench")
    os.environ.setdefault("D4RL_DATASET_DIR", "/root/autodl-tmp/rl_datasets/d4rl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

    status = "ok"
    note = ""
    train_size = ""
    val_size = ""

    try:
        if is_ogbench_env(args.env_name):
            train_size, val_size, note = handle_ogbench(args.env_name)
        elif is_adroit_env(args.env_name):
            train_size, val_size, note = handle_adroit_direct(args.env_name)
        else:
            train_size, val_size, note = handle_d4rl_non_adroit(args.env_name)
    except Exception as e:
        status = "fail"
        note = repr(e)
        print(f"[FAIL] {args.env_name}: {e}")
        traceback.print_exc()

    upsert_status(
        Path(args.status_file),
        {
            "env_name": args.env_name,
            "status": status,
            "frame_stack": args.frame_stack if args.frame_stack is not None else "",
            "train_size": train_size,
            "val_size": val_size,
            "note": note,
        },
    )

    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
