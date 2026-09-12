from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)
DATA = Path("cotahist_2015_2025_clean.parquet")
URLS = [
    "https://media.githubusercontent.com/media/cockles98/mfg-for-financial-market/main/data/processed/cotahist_2015_2025_clean.parquet",
    "https://media.githubusercontent.com/media/cockles98/ibov-data-from-b3/main/data/processed/cotahist_2015_2025_clean.parquet",
    "https://github.com/cockles98/mfg-for-financial-market/raw/refs/heads/main/data/processed/cotahist_2015_2025_clean.parquet",
]


def download() -> tuple[str, str]:
    errors: list[str] = []
    for url in URLS:
        try:
            print(f"Tentando {url}")
            with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
                response.raise_for_status()
                h = hashlib.sha256()
                total = 0
                with DATA.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            h.update(chunk)
                            total += len(chunk)
            if total < 1_000_000:
                raise RuntimeError(f"arquivo pequeno demais: {total} bytes")
            print(f"Baixado: {total:,} bytes")
            return url, h.hexdigest()
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if DATA.exists():
                DATA.unlink()
    raise RuntimeError("Falha em todas as URLs: " + " | ".join(errors))


def scalar(v: Any) -> Any:
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def main() -> None:
    url, sha256 = download()
    df = pd.read_parquet(DATA)
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}

    date_col = next((lower[k] for k in ["dtpreg", "date", "data_pregao", "trading_date"] if k in lower), None)
    ticker_col = next((lower[k] for k in ["codneg", "ticker", "symbol", "cod_papel"] if k in lower), None)
    market_col = next((lower[k] for k in ["tpmerc", "market_type", "tp_merc"] if k in lower), None)

    audit: dict[str, Any] = {
        "source_url": url,
        "sha256": sha256,
        "file_bytes": DATA.stat().st_size,
        "rows": int(len(df)),
        "columns": cols,
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
    }

    if date_col is not None:
        s = pd.to_datetime(df[date_col], errors="coerce")
        audit["date_column"] = str(date_col)
        audit["date_min"] = scalar(s.min())
        audit["date_max"] = scalar(s.max())
        audit["date_non_null"] = int(s.notna().sum())

    if ticker_col is not None:
        audit["ticker_column"] = str(ticker_col)
        audit["unique_tickers"] = int(df[ticker_col].astype(str).nunique())
        audit["ticker_examples"] = sorted(df[ticker_col].dropna().astype(str).unique().tolist())[:50]

    if market_col is not None:
        audit["market_column"] = str(market_col)
        vc = df[market_col].value_counts(dropna=False).head(30)
        audit["market_counts"] = {str(k): int(v) for k, v in vc.items()}
        numeric = pd.to_numeric(df[market_col], errors="coerce")
        opts = df[numeric.isin([70, 80])].copy()
        audit["option_rows_market_70_80"] = int(len(opts))
        if ticker_col is not None and len(opts):
            audit["option_ticker_examples"] = sorted(opts[ticker_col].dropna().astype(str).unique().tolist())[:100]
            opts.head(1000).to_csv(OUT / "option_sample.csv", index=False)

    df.head(1000).to_csv(OUT / "head_sample.csv", index=False)
    with (OUT / "audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
