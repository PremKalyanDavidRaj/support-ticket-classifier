"""Download and validate the original BANKING77 splits."""

import hashlib
import io
import json
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data"
)

EXPECTED_ROWS = {"train": 10003, "test": 3080}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for split, expected_count in EXPECTED_ROWS.items():
            url = f"{BASE_URL}/{split}.csv"
            response = client.get(url)
            response.raise_for_status()

            content = response.content
            frame = pd.read_csv(io.BytesIO(content))

            required = {"text", "category"}
            if not required.issubset(frame.columns):
                raise ValueError(
                    f"{split}: unexpected columns {list(frame.columns)}"
                )

            if len(frame) != expected_count:
                raise ValueError(f"{split}: unexpected number of rows")

            if frame[["text", "category"]].isna().any().any():
                raise ValueError(f"{split}: missing text or labels")

            if frame["text"].str.strip().eq("").any():
                raise ValueError(f"{split}: empty text")

            if frame["category"].nunique() != 77:
                raise ValueError(f"{split}: expected 77 categories")

            # Preserve the downloaded source bytes without modifications.
            path = DATA_DIR / f"{split}.csv"
            path.write_bytes(content)

            manifest[split] = {
                "url": url,
                "rows": len(frame),
                "categories": int(frame["category"].nunique()),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

            print(
                f"{split}: {len(frame):,} rows, "
                f"{frame['category'].nunique()} categories"
            )

    report_path = REPORT_DIR / "data_manifest.json"
    report_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Download complete. Source hashes saved in reports/data_manifest.json")


if __name__ == "__main__":
    main()
