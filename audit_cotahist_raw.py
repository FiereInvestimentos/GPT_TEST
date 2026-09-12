from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

YEARS = [2023, 2024, 2025]
PREFIXES = ("PETR", "VALE", "BOVA", "ITUB", "BBDC", "BBAS", "ABEV", "B3SA", "WEGE")
OUT = Path("artifacts_raw")
OUT.mkdir(exist_ok=True)
TMP = Path("tmp_cotahist")
TMP.mkdir(exist_ok=True)


def n(s: str) -> int:
    s = s.strip()
    return int(s) if s else 0


def p(s: str) -> float:
    return n(s) / 100.0


def parse_record(line: str) -> dict[str, object]:
    return {
        "date": line[2:10],
        "bdi": line[10:12].strip(),
        "ticker": line[12:24].strip(),
        "market_type": n(line[24:27]),
        "short_name": line[27:39].strip(),
        "specification": line[39:49].strip(),
        "currency": line[52:56].strip(),
        "open": p(line[56:69]),
        "high": p(line[69:82]),
        "low": p(line[82:95]),
        "average": p(line[95:108]),
        "close": p(line[108:121]),
        "best_bid": p(line[121:134]),
        "best_ask": p(line[134:147]),
        "trades": n(line[147:152]),
        "quantity": n(line[152:170]),
        "financial_volume": p(line[170:188]),
        "strike": p(line[188:201]),
        "option_indicator": line[201:202].strip(),
        "expiration": line[202:210].strip(),
        "quote_factor": n(line[210:217]),
        "points_strike": p(line[217:230]),
        "isin": line[230:242].strip(),
        "distribution": line[242:245].strip(),
    }


def candidate_urls(year: int) -> list[str]:
    name = f"COTAHIST_A{year}.ZIP"
    return [
        f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{name}",
        f"http://bvmf.bmfbovespa.com.br/InstDados/SerHist/{name}",
    ]


def download(year: int) -> tuple[Path, dict[str, object]]:
    dest = TMP / f"COTAHIST_A{year}.ZIP"
    errors: list[str] = []
    for url in candidate_urls(year):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ATLAS-QP-research/0.1; +https://github.com/FiereInvestimentos/GPT_TEST)",
                "Accept": "application/zip,application/octet-stream,*/*",
            }
            with requests.get(url, headers=headers, timeout=180, stream=True, allow_redirects=True) as r:
                status = r.status_code
                r.raise_for_status()
                h = hashlib.sha256()
                total = 0
                first = b""
                with dest.open("wb") as fh:
                    for chunk in r.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if total == 0:
                            first = chunk[:8]
                        fh.write(chunk)
                        h.update(chunk)
                        total += len(chunk)
            if first[:2] != b"PK":
                raise RuntimeError(f"conteudo nao ZIP: assinatura={first!r}, bytes={total}")
            return dest, {
                "year": year,
                "url": url,
                "http_status": status,
                "bytes": total,
                "sha256": h.hexdigest(),
            }
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError(" | ".join(errors))


def audit_year(year: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    path, meta = download(year)
    market_counts: Counter[int] = Counter()
    option_prefix_counts: Counter[str] = Counter()
    option_dates: list[str] = []
    samples: list[dict[str, object]] = []
    stats = Counter()
    min_date: str | None = None
    max_date: str | None = None

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        txt_names = [x for x in names if x.upper().endswith(".TXT")]
        if not txt_names:
            raise RuntimeError(f"ZIP {year} sem TXT: {names}")
        inner = txt_names[0]
        meta["zip_members"] = names
        meta["data_member"] = inner
        meta["uncompressed_bytes"] = zf.getinfo(inner).file_size

        with zf.open(inner) as raw:
            wrapper = io.TextIOWrapper(raw, encoding="latin-1", errors="strict", newline="")
            for line_no, line in enumerate(wrapper, start=1):
                if not line.startswith("01"):
                    continue
                line = line.rstrip("\r\n")
                if len(line) < 245:
                    stats["short_lines"] += 1
                    continue
                rec = parse_record(line)
                mt = int(rec["market_type"])
                market_counts[mt] += 1
                date = str(rec["date"])
                min_date = date if min_date is None or date < min_date else min_date
                max_date = date if max_date is None or date > max_date else max_date
                stats["records"] += 1

                if mt in (70, 80):
                    stats["option_records"] += 1
                    ticker = str(rec["ticker"])
                    prefix = next((x for x in PREFIXES if ticker.startswith(x)), "OTHER")
                    option_prefix_counts[prefix] += 1
                    option_dates.append(date)
                    if float(rec["open"]) > 0:
                        stats["option_open_positive"] += 1
                    if int(rec["trades"]) > 0:
                        stats["option_trades_positive"] += 1
                    if int(rec["quantity"]) > 0:
                        stats["option_quantity_positive"] += 1
                    if str(rec["expiration"]) not in ("", "99991231", "00000000"):
                        stats["option_expiration_valid"] += 1
                    if float(rec["strike"]) > 0:
                        stats["option_strike_positive"] += 1
                    if prefix != "OTHER" and len(samples) < 3000:
                        samples.append(rec)

    meta.update(
        {
            "date_min": min_date,
            "date_max": max_date,
            "market_type_counts": {str(k): v for k, v in sorted(market_counts.items())},
            "option_prefix_counts": dict(option_prefix_counts),
            "stats": dict(stats),
            "option_date_min": min(option_dates) if option_dates else None,
            "option_date_max": max(option_dates) if option_dates else None,
        }
    )
    path.unlink(missing_ok=True)
    return meta, samples


def main() -> None:
    report: dict[str, object] = {"years": [], "errors": []}
    all_samples: list[dict[str, object]] = []
    for year in YEARS:
        print(f"AUDIT_YEAR_START={year}", flush=True)
        try:
            meta, samples = audit_year(year)
            report["years"].append(meta)
            all_samples.extend(samples)
            print("AUDIT_YEAR_RESULT=" + json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            err = {"year": year, "error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(err)
            print("AUDIT_YEAR_ERROR=" + json.dumps(err, ensure_ascii=False), flush=True)

    with (OUT / "cotahist_raw_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    if all_samples:
        fields = list(all_samples[0].keys())
        with (OUT / "option_samples.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_samples)

    print("FINAL_AUDIT=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if report["errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
