from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import atlas_backtest_v01 as atlas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_download_year(year: int) -> tuple[Path, dict[str, Any]]:
    path = atlas.CACHE / f"COTAHIST_A{year}.ZIP"
    partial = path.with_suffix(".partial")
    if path.exists() and zipfile.is_zipfile(path):
        return path, {
            "year": year,
            "source": "cache",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504, 520),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ATLAS-QP-research/0.1)",
        "Accept": "application/zip,application/octet-stream,*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
    errors: list[str] = []

    for attempt in range(1, 9):
        path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        try:
            request_url = f"{url}?atlas_attempt={attempt}"
            print(f"DOWNLOAD_ATTEMPT year={year} attempt={attempt}", flush=True)
            size = 0
            expected: int | None = None
            signature = b""
            with session.get(
                request_url,
                headers=headers,
                stream=True,
                timeout=(30, 600),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                raw_length = response.headers.get("Content-Length")
                if raw_length and raw_length.isdigit():
                    expected = int(raw_length)
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        if size == 0:
                            signature = chunk[:8]
                        handle.write(chunk)
                        size += len(chunk)
            if signature[:2] != b"PK":
                raise RuntimeError(f"assinatura nao ZIP: {signature!r}")
            if expected is not None and size != expected:
                raise RuntimeError(f"download incompleto: {size} de {expected} bytes")
            partial.replace(path)
            if not zipfile.is_zipfile(path):
                raise RuntimeError("arquivo nao passou na validacao ZIP")
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"CRC invalido no membro {bad_member}")
            return path, {
                "year": year,
                "source": url,
                "download_attempt": attempt,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        except Exception as exc:
            errors.append(f"attempt={attempt}: {type(exc).__name__}: {exc}")
            path.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            time.sleep(min(5 * attempt, 30))

    raise RuntimeError(f"Falha persistente no download de {year}: " + " | ".join(errors))


atlas.download_year = robust_download_year

if __name__ == "__main__":
    atlas.main()
