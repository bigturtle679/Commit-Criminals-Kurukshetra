#!/usr/bin/env python3
"""Download and verify NeuroScope 3D model checkpoints.

Cross-platform utility (Windows, Linux, macOS) for automated checkpoint verification
and retrieval from public research hubs (Hugging Face / MONAI).
"""
import sys
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    {
        "name": "SwinUNETR (Vision Transformer BraTS21)",
        "path": ROOT / "models/swinunetr_brats/model_weights_BRATS-finetune.pt",
        "url": "https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-finetune.pt",
        "expected_sha256": "6fd4ec5dc690f365a99c10b92f0d5519ca8a6b68feb844ff3424e9f6a01f6a39",
        "size_mb": 245,
    },
    {
        "name": "ResEnc-M (3D Residual Encoder UNet)",
        "path": ROOT / "models/resenc_brats/resenc_m_brats.pt",
        "expected_sha256": "9e65a271f99f81605bf09cd028382888a318f804a002b459779474698f80b5c6",
        "size_mb": 36,
    },
    {
        "name": "SegResNet (3D CNN Autoencoder)",
        "path": ROOT / "models/brats_mri_segmentation/models/model.pt",
        "expected_sha256": "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae",
        "size_mb": 18,
    }
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, name: str, expected_size: int = 0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".pt.download")
    print(f"\n[+] Downloading {name} ({expected_size} MB)...")
    print(f"    Source: {url}")
    print(f"    Destination: {dest}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeuroScope/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = int(downloaded / total_bytes * 100)
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_bytes / (1024 * 1024)
                    bar = "=" * (pct // 4) + "-" * (25 - (pct // 4))
                    sys.stdout.write(f"\r    [{bar}] {pct}% ({mb_cur:.1f}/{mb_tot:.1f} MB)")
                    sys.stdout.flush()

        print()
        temp_dest.rename(dest)
        print(f"[✓] Successfully downloaded {dest.name}")
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        print(f"\n[!] Failed to download {name}: {e}")
        raise


def main():
    print("=" * 60)
    print("NeuroScope 3D — Checkpoint Verification & Download Manager")
    print("=" * 60)

    for item in CHECKPOINTS:
        p = item["path"]
        name = item["name"]
        expected_hash = item.get("expected_sha256")
        url = item.get("url")
        size_mb = item.get("size_mb", 0)

        print(f"\nChecking {name}...")
        if p.is_file() and p.stat().st_size > 1000:
            actual_hash = sha256_file(p)
            if expected_hash and actual_hash == expected_hash:
                print(f"  [✓] Verified: {p.name} (SHA-256: {actual_hash[:16]}...)")
                continue
            elif not expected_hash:
                print(f"  [✓] Present: {p.name}")
                continue
            else:
                print(f"  [!] Hash mismatch for {p.name}! Re-downloading...")

        if url:
            download_file(url, p, name, size_mb)
            actual_hash = sha256_file(p)
            if expected_hash and actual_hash != expected_hash:
                print(f"  [WARNING] Hash mismatch after download! Expected {expected_hash}, got {actual_hash}")
            else:
                print(f"  [✓] Hash verified: {actual_hash[:16]}...")
        else:
            print(f"  [!] Local checkpoint {p.name} missing and has no public URL.")

    print("\n" + "=" * 60)
    print("All checkpoints ready for NeuroScope 3D.")
    print("=" * 60)


if __name__ == "__main__":
    main()
