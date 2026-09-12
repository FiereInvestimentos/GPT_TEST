from __future__ import annotations

import hashlib
import io
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "atlas_config_v01.json"
OUT = ROOT / "atlas_results_v01"
CACHE = ROOT / ".cache" / "cotahist"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def as_int(s: str) -> int:
    s = s.strip()
    return int(s) if s else 0


def as_price(s: str) -> float:
    return as_int(s) / 100.0


def annual_urls(year: int) -> list[str]:
    name = f"COTAHIST_A{year}.ZIP"
    return [
        f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{name}",
        f"http://bvmf.bmfbovespa.com.br/InstDados/SerHist/{name}",
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_year(year: int) -> tuple[Path, dict[str, Any]]:
    path = CACHE / f"COTAHIST_A{year}.ZIP"
    if path.exists() and zipfile.is_zipfile(path):
        return path, {
            "year": year,
            "source": "cache",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    errors: list[str] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ATLAS-QP-research/0.1)",
        "Accept": "application/zip,application/octet-stream,*/*",
    }
    for url in annual_urls(year):
        try:
            tmp = path.with_suffix(".partial")
            h = hashlib.sha256()
            size = 0
            signature = b""
            with requests.get(url, headers=headers, stream=True, timeout=240, allow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if size == 0:
                            signature = chunk[:8]
                        fh.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
            if signature[:2] != b"PK":
                raise RuntimeError(f"resposta nao ZIP: assinatura={signature!r}, bytes={size}")
            tmp.replace(path)
            if not zipfile.is_zipfile(path):
                raise RuntimeError("arquivo baixado nao passou na validacao ZIP")
            return path, {"year": year, "source": url, "bytes": size, "sha256": h.hexdigest()}
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            path.unlink(missing_ok=True)
            path.with_suffix(".partial").unlink(missing_ok=True)
    raise RuntimeError(" | ".join(errors))


def root_for_ticker(ticker: str, roots: Iterable[str]) -> str | None:
    for root in roots:
        if ticker.startswith(root):
            return root
    return None


def extract_market_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    root_to_under = cfg["assets"]
    roots = tuple(root_to_under.keys())
    underlyings = set(root_to_under.values())
    option_rows: list[tuple[Any, ...]] = []
    spot_rows: list[tuple[Any, ...]] = []
    manifest: list[dict[str, Any]] = []

    for year in cfg["data_years"]:
        print(f"DATA_YEAR_START={year}", flush=True)
        path, meta = download_year(int(year))
        counts = Counter()
        with zipfile.ZipFile(path) as zf:
            txts = [n for n in zf.namelist() if n.upper().endswith(".TXT")]
            if not txts:
                raise RuntimeError(f"ZIP {year} sem arquivo TXT")
            inner = txts[0]
            meta["member"] = inner
            meta["uncompressed_bytes"] = zf.getinfo(inner).file_size
            with zf.open(inner) as raw:
                text = io.TextIOWrapper(raw, encoding="latin-1", errors="strict", newline="")
                for line in text:
                    if not line.startswith("01"):
                        continue
                    if len(line) < 245:
                        counts["short"] += 1
                        continue
                    ticker = line[12:24].strip()
                    mt = as_int(line[24:27])
                    if mt == 10 and ticker in underlyings:
                        spot_rows.append(
                            (
                                line[2:10], ticker, as_price(line[56:69]), as_price(line[108:121]),
                                as_int(line[210:217]), line[242:245].strip(),
                            )
                        )
                        counts["spot"] += 1
                        continue
                    if mt not in (70, 80):
                        continue
                    root = root_for_ticker(ticker, roots)
                    if root is None:
                        continue
                    expiration = line[202:210].strip()
                    if not expiration or expiration in ("00000000", "99991231"):
                        counts["bad_expiration"] += 1
                        continue
                    option_rows.append(
                        (
                            line[2:10], root, ticker, mt,
                            as_price(line[56:69]), as_price(line[69:82]), as_price(line[82:95]),
                            as_price(line[108:121]), as_price(line[121:134]), as_price(line[134:147]),
                            as_int(line[147:152]), as_int(line[152:170]), as_price(line[170:188]),
                            as_price(line[188:201]), expiration, as_int(line[210:217]),
                        )
                    )
                    counts["options"] += 1
        meta["filtered_counts"] = dict(counts)
        manifest.append(meta)
        print("DATA_YEAR_RESULT=" + json.dumps(meta, sort_keys=True), flush=True)

    spots = pd.DataFrame(
        spot_rows,
        columns=["date", "ticker", "open", "close", "quote_factor", "distribution"],
    )
    options = pd.DataFrame(
        option_rows,
        columns=[
            "date", "root", "ticker", "market_type", "open", "high", "low", "close",
            "bid", "ask", "trades", "quantity", "financial_volume", "strike", "expiration",
            "quote_factor",
        ],
    )
    if spots.empty or options.empty:
        raise RuntimeError(f"dados insuficientes: spots={len(spots)}, options={len(options)}")

    spots["date"] = pd.to_datetime(spots["date"], format="%Y%m%d", errors="coerce")
    options["date"] = pd.to_datetime(options["date"], format="%Y%m%d", errors="coerce")
    options["expiration"] = pd.to_datetime(options["expiration"], format="%Y%m%d", errors="coerce")
    spots = spots.dropna(subset=["date", "close"])
    options = options.dropna(subset=["date", "expiration", "strike"])
    spots = spots[(spots["close"] > 0) & (spots["quote_factor"].isin([0, 1]))]
    options = options[(options["quote_factor"].isin([0, 1])) & (options["strike"] > 0)]

    spots = spots.sort_values(["ticker", "date", "close"]).drop_duplicates(["ticker", "date"], keep="last")
    options = options.sort_values(["date", "ticker", "trades", "quantity"]).drop_duplicates(["date", "ticker"], keep="last")

    numeric_float = ["open", "high", "low", "close", "bid", "ask", "financial_volume", "strike"]
    for col in numeric_float:
        options[col] = pd.to_numeric(options[col], errors="coerce").astype("float32")
    for col in ["trades", "quantity", "market_type"]:
        options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0).astype("int32")
    for col in ["open", "close"]:
        spots[col] = pd.to_numeric(spots[col], errors="coerce").astype("float32")

    print(f"FILTERED_SPOT_ROWS={len(spots)} FILTERED_OPTION_ROWS={len(options)}", flush=True)
    return spots, options, manifest


def prepare_underlying(spots: pd.DataFrame, ticker: str) -> dict[str, Any] | None:
    d = spots[spots["ticker"] == ticker].sort_values("date").copy()
    if len(d) < 400:
        return None
    dates = pd.DatetimeIndex(d["date"])
    close = d["close"].astype(float).to_numpy()
    logp = np.log(close)
    ret = np.full(len(close), np.nan)
    ret[1:] = np.diff(logp)
    sret = pd.Series(ret)
    vol5 = sret.rolling(5).std(ddof=1).to_numpy()
    vol20 = sret.rolling(20).std(ddof=1).to_numpy()
    vol60 = sret.rolling(60).std(ddof=1).to_numpy()

    def momentum(h: int) -> np.ndarray:
        out = np.full(len(close), np.nan)
        out[h:] = (logp[h:] - logp[:-h]) / (np.maximum(vol20[h:], 1e-8) * math.sqrt(h))
        return out

    sq = ret * ret
    negsq = np.where(ret < 0, sq, 0.0)
    downside = pd.Series(negsq).rolling(20).sum().to_numpy() / np.maximum(
        pd.Series(sq).rolling(20).sum().to_numpy(), 1e-12
    )
    rollmax = pd.Series(logp).rolling(60).max().to_numpy()
    drawdown = logp - rollmax
    vol_ratio = vol5 / np.maximum(vol60, 1e-8)
    features = np.column_stack([momentum(5), momentum(20), momentum(60), vol_ratio, downside, drawdown])
    return {
        "dates": dates,
        "close": close,
        "logp": logp,
        "ret": ret,
        "vol20": vol20,
        "features": features,
        "date_to_index": {pd.Timestamp(x): i for i, x in enumerate(dates)},
    }


def weighted_stats(values: np.ndarray, weights: np.ndarray | None, horizon: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[finite]
    if len(values) < 10:
        return math.nan, math.nan, 0.0
    if weights is None:
        weights = np.full(len(values), 1.0 / len(values))
        raw_eff = float(len(values))
    else:
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        raw_eff = 1.0 / np.sum(weights * weights)
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    effective_n = max(8.0, raw_eff / max(1.0, float(horizon)))
    se = math.sqrt(max(var, 0.0) / effective_n)
    return mean, se, effective_n


def forecast_distributions(prep: dict[str, Any], idx: int, horizon: int, cfg: dict[str, Any]) -> list[dict[str, Any]] | None:
    lookback = int(cfg["rolling_lookback_days"])
    minimum_origins = int(cfg["minimum_training_origins"])
    minimum_samples = int(cfg["minimum_model_samples"])
    features = prep["features"]
    logp = prep["logp"]
    vol20 = prep["vol20"]
    start = max(60, idx - lookback)
    end = idx - horizon
    if end < start:
        return None
    origins = np.arange(start, end + 1, dtype=int)
    fwd = logp[origins + horizon] - logp[origins]
    valid = (
        np.isfinite(fwd) & np.isfinite(vol20[origins]) & (vol20[origins] > 1e-8)
        & np.all(np.isfinite(features[origins]), axis=1) & (np.abs(fwd) < 0.60)
    )
    origins = origins[valid]
    fwd = fwd[valid]
    if len(origins) < minimum_origins or not np.isfinite(vol20[idx]) or vol20[idx] <= 1e-8:
        return None

    center = float(np.median(fwd))
    scale_ratio = np.clip(vol20[idx] / vol20[origins], 0.50, 2.00)
    scaled = center + (fwd - center) * scale_ratio
    age = idx - origins
    recency_w = np.power(0.5, age / 504.0)

    current = features[idx]
    same_trend = np.sign(features[origins, 1]) == np.sign(current[1])
    vol_close = (scale_ratio >= 0.75) & (scale_ratio <= 1.33)
    downside_close = np.abs(features[origins, 4] - current[4]) <= 0.20
    regime_mask = same_trend & vol_close & downside_close
    regime_idx = np.flatnonzero(regime_mask)

    x = features[origins]
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    mad = np.where((mad > 1e-6) & np.isfinite(mad), mad, 1.0)
    dist = np.sum(((x - current) / mad) ** 2, axis=1)
    k = int(min(250, max(minimum_samples, round(math.sqrt(len(origins)) * 8))))
    nearest_idx = np.argsort(dist)[:k]
    if len(regime_idx) < minimum_samples:
        regime_idx = nearest_idx[: max(minimum_samples, min(200, len(nearest_idx)))]

    models = [
        {"name": "FHS_RECENCY", "returns": scaled, "weights": recency_w},
        {"name": "REGIME_FILTER", "returns": scaled[regime_idx], "weights": None},
        {"name": "KNN_CONDITIONAL", "returns": scaled[nearest_idx], "weights": None},
    ]
    if any(len(m["returns"]) < minimum_samples for m in models):
        return None
    return models


def vertical_payoff(st: np.ndarray, k1: float, k2: float, kind: str) -> np.ndarray:
    width = k2 - k1
    if kind == "call":
        return np.clip(st - k1, 0.0, width)
    return np.clip(k2 - st, 0.0, width)


def option_chain(chain: pd.DataFrame, spot: float, cfg: dict[str, Any]) -> pd.DataFrame:
    c = chain.copy()
    c = c[
        (c["close"] > 0) & (c["bid"] > 0) & (c["ask"] > 0) & (c["ask"] >= c["bid"])
        & (c["trades"] >= int(cfg["minimum_option_trades_signal_day"]))
        & (c["quantity"] >= int(cfg["minimum_option_quantity_signal_day"]))
    ]
    mid = (c["bid"].astype(float) + c["ask"].astype(float)) / 2.0
    rel_spread = (c["ask"].astype(float) - c["bid"].astype(float)) / np.maximum(mid, 0.01)
    c = c[rel_spread <= float(cfg["maximum_relative_bid_ask_spread"])]
    c = c[np.abs(c["strike"].astype(float) / spot - 1.0) <= float(cfg["maximum_absolute_moneyness"])]
    c = c.sort_values(["strike", "trades", "quantity"], ascending=[True, False, False]).drop_duplicates("strike")
    return c


def evaluate_verticals(
    chain: pd.DataFrame,
    models: list[dict[str, Any]],
    spot: float,
    horizon: int,
    kind: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    c = option_chain(chain, spot, cfg)
    if len(c) < 2:
        return []
    rows = list(c.itertuples(index=False))
    max_steps = int(cfg["maximum_strike_steps"])
    rf = float(cfg["risk_free_annual_conservative"])
    discount = 1.0 / ((1.0 + rf) ** (horizon / 252.0))
    cost_unit = float(cfg["signal_cost_per_contract_brl"]) / 100.0
    z = float(cfg["lcb_z"])
    results: list[dict[str, Any]] = []

    for a in range(len(rows) - 1):
        for b in range(a + 1, min(len(rows), a + 1 + max_steps)):
            lo, hi = rows[a], rows[b]
            k1, k2 = float(lo.strike), float(hi.strike)
            width = k2 - k1
            if width <= 0:
                continue
            if width < float(cfg["minimum_width_pct_spot"]) * spot:
                continue
            if width > min(float(cfg["maximum_width_pct_spot"]) * spot, float(cfg["maximum_width_brl"])):
                continue
            if width * 100.0 + float(cfg["cost_scenarios"]["severe"]["fixed_brl"]) > float(cfg["maximum_risk_per_trade_brl"]):
                continue

            stats: list[dict[str, float]] = []
            for model in models:
                st = spot * np.exp(np.asarray(model["returns"], dtype=float))
                payoff = vertical_payoff(st, k1, k2, kind)
                mean, se, eff_n = weighted_stats(payoff, model["weights"], horizon)
                stats.append({"mean": mean, "se": se, "eff_n": eff_n})
            if any(not np.isfinite(s["mean"]) or not np.isfinite(s["se"]) for s in stats):
                continue

            structures: list[tuple[str, float, str, str, str]]
            if kind == "call":
                structures = [
                    ("long", float(lo.ask) - float(hi.bid), str(lo.ticker), str(hi.ticker), "bullish"),
                    ("short", float(lo.bid) - float(hi.ask), str(hi.ticker), str(lo.ticker), "bearish"),
                ]
            else:
                structures = [
                    ("long", float(hi.ask) - float(lo.bid), str(hi.ticker), str(lo.ticker), "bearish"),
                    ("short", float(hi.bid) - float(lo.ask), str(lo.ticker), str(hi.ticker), "bullish"),
                ]

            for side, market_price, buy_ticker, sell_ticker, direction in structures:
                if market_price <= 0 or market_price >= width:
                    continue
                if side == "long":
                    model_edges = [discount * s["mean"] - market_price - cost_unit for s in stats]
                    model_lcbs = [discount * max(0.0, s["mean"] - z * s["se"]) - market_price - cost_unit for s in stats]
                    max_loss = market_price * 100.0 + float(cfg["signal_cost_per_contract_brl"])
                else:
                    margin = max(0.0, width - market_price)
                    carry = margin * ((1.0 + rf) ** (horizon / 252.0) - 1.0)
                    model_edges = [market_price - discount * s["mean"] - carry - cost_unit for s in stats]
                    model_lcbs = [market_price - discount * (s["mean"] + z * s["se"]) - carry - cost_unit for s in stats]
                    max_loss = margin * 100.0 + float(cfg["signal_cost_per_contract_brl"])
                if max_loss <= 0 or max_loss > float(cfg["maximum_risk_per_trade_brl"]):
                    continue
                if min(model_edges) <= 0:
                    continue
                worst_lcb = min(model_lcbs)
                lcb_contract = worst_lcb * 100.0
                if lcb_contract < float(cfg["minimum_lcb_per_contract_brl"]):
                    continue
                if lcb_contract / max_loss < float(cfg["minimum_lcb_to_max_loss"]):
                    continue
                min_trades = min(int(lo.trades), int(hi.trades))
                rel1 = (float(lo.ask) - float(lo.bid)) / max((float(lo.ask) + float(lo.bid)) / 2.0, 0.01)
                rel2 = (float(hi.ask) - float(hi.bid)) / max((float(hi.ask) + float(hi.bid)) / 2.0, 0.01)
                liquidity_factor = min(1.0, math.log1p(min_trades) / math.log1p(100.0)) * max(0.05, 1.0 - max(rel1, rel2))
                results.append(
                    {
                        "kind": kind,
                        "side": side,
                        "direction": direction,
                        "k1": k1,
                        "k2": k2,
                        "width": width,
                        "buy_ticker": buy_ticker,
                        "sell_ticker": sell_ticker,
                        "signal_market_price": market_price,
                        "model_edge_min_unit": min(model_edges),
                        "model_edge_median_unit": float(np.median(model_edges)),
                        "lcb_unit": worst_lcb,
                        "lcb_contract": lcb_contract,
                        "signal_max_loss": max_loss,
                        "score": (lcb_contract / max_loss) * liquidity_factor,
                        "model1_mean": stats[0]["mean"],
                        "model2_mean": stats[1]["mean"],
                        "model3_mean": stats[2]["mean"],
                        "model1_se": stats[0]["se"],
                        "model2_se": stats[1]["se"],
                        "model3_se": stats[2]["se"],
                        "effective_n_min": min(s["eff_n"] for s in stats),
                        "min_leg_trades": min_trades,
                    }
                )
    return results


def lookup_option(indexed: pd.DataFrame, date: pd.Timestamp, ticker: str) -> pd.Series | None:
    try:
        row = indexed.loc[(date, ticker)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.sort_values(["trades", "quantity"], ascending=False).iloc[0]
    return row


def adverse_slippage(row: pd.Series, scenario: str) -> float:
    op = float(row["open"])
    day_range = max(0.0, float(row["high"]) - float(row["low"]))
    if scenario == "raw_open":
        return 0.0
    if scenario == "base":
        return 0.01
    if scenario == "stress":
        return max(0.02, 0.05 * op, 0.10 * day_range)
    return max(0.03, 0.10 * op, 0.25 * day_range)


def scenario_entry_price(buy: pd.Series, sell: pd.Series, side: str, scenario: str) -> float:
    buy_px = float(buy["open"]) + adverse_slippage(buy, scenario)
    sell_px = max(0.0, float(sell["open"]) - adverse_slippage(sell, scenario))
    return buy_px - sell_px if side == "long" else sell_px - buy_px


def generate_candidates(
    spots: pd.DataFrame,
    options: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_to_under = cfg["assets"]
    prepared = {root: prepare_underlying(spots, ticker) for root, ticker in root_to_under.items()}
    prepared = {k: v for k, v in prepared.items() if v is not None}
    if not prepared:
        raise RuntimeError("nenhum ativo-objeto possui historico suficiente")

    options = options[options["root"].isin(prepared.keys())].copy()
    indexed = options.set_index(["date", "ticker"]).sort_index()
    forecast_cache: dict[tuple[str, int, int], list[dict[str, Any]] | None] = {}
    candidates: list[dict[str, Any]] = []
    counters = Counter()
    signal_start = pd.Timestamp(cfg["signal_start"])
    min_dte, max_dte = map(int, cfg["dte_business_days"])

    for date, day in options.groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date < signal_start:
            continue
        for root, root_day in day.groupby("root", sort=False):
            prep = prepared.get(root)
            if prep is None:
                continue
            idx = prep["date_to_index"].get(date)
            if idx is None or idx + 1 >= len(prep["dates"]):
                continue
            entry_date = pd.Timestamp(prep["dates"][idx + 1])
            spot = float(prep["close"][idx])
            if not np.isfinite(spot) or spot <= 0:
                continue
            best: dict[str, Any] | None = None
            counters["asset_days"] += 1

            for expiration, exp_day in root_day.groupby("expiration", sort=False):
                expiration = pd.Timestamp(expiration)
                exp_idx = prep["date_to_index"].get(expiration)
                if exp_idx is None:
                    counters["expiration_not_trading_day"] += 1
                    continue
                horizon = exp_idx - idx
                if horizon < min_dte or horizon > max_dte:
                    continue
                counters["eligible_expiries"] += 1
                key = (root, idx, horizon)
                if key not in forecast_cache:
                    forecast_cache[key] = forecast_distributions(prep, idx, horizon, cfg)
                models = forecast_cache[key]
                if models is None:
                    counters["forecast_unavailable"] += 1
                    continue

                all_verticals: list[dict[str, Any]] = []
                calls = exp_day[exp_day["market_type"] == 70]
                puts = exp_day[exp_day["market_type"] == 80]
                if len(calls) >= 2:
                    all_verticals.extend(evaluate_verticals(calls, models, spot, horizon, "call", cfg))
                if len(puts) >= 2:
                    all_verticals.extend(evaluate_verticals(puts, models, spot, horizon, "put", cfg))
                counters["qualified_verticals"] += len(all_verticals)

                for cand in all_verticals:
                    buy = lookup_option(indexed, entry_date, cand["buy_ticker"])
                    sell = lookup_option(indexed, entry_date, cand["sell_ticker"])
                    if buy is None or sell is None:
                        counters["missing_entry_leg"] += 1
                        continue
                    if float(buy["open"]) <= 0 or float(sell["open"]) <= 0:
                        counters["zero_entry_open"] += 1
                        continue
                    raw_entry = scenario_entry_price(buy, sell, cand["side"], "raw_open")
                    if raw_entry <= 0 or raw_entry >= cand["width"] * 1.05:
                        counters["invalid_raw_vertical_open"] += 1
                        continue
                    allowed = float(cfg["maximum_edge_consumption_at_open"]) * cand["lcb_unit"]
                    if cand["side"] == "long" and raw_entry > cand["signal_market_price"] + allowed:
                        counters["opening_limit_not_filled"] += 1
                        continue
                    if cand["side"] == "short" and raw_entry < cand["signal_market_price"] - allowed:
                        counters["opening_limit_not_filled"] += 1
                        continue

                    full = dict(cand)
                    full.update(
                        {
                            "signal_date": date,
                            "entry_date": entry_date,
                            "expiration": expiration,
                            "root": root,
                            "underlying": root_to_under[root],
                            "signal_spot": spot,
                            "horizon": horizon,
                            "buy_open": float(buy["open"]),
                            "buy_high": float(buy["high"]),
                            "buy_low": float(buy["low"]),
                            "sell_open": float(sell["open"]),
                            "sell_high": float(sell["high"]),
                            "sell_low": float(sell["low"]),
                            "raw_entry_price": raw_entry,
                        }
                    )
                    for scenario in cfg["cost_scenarios"]:
                        full[f"entry_{scenario}"] = scenario_entry_price(buy, sell, cand["side"], scenario)
                    if best is None or full["score"] > best["score"]:
                        best = full
            if best is not None:
                candidates.append(best)
                counters["best_asset_day_candidates"] += 1

    result = pd.DataFrame(candidates)
    if result.empty:
        return result, {"counters": dict(counters), "prepared_assets": list(prepared.keys())}
    return result.sort_values(["entry_date", "score"], ascending=[True, False]), {
        "counters": dict(counters),
        "prepared_assets": list(prepared.keys()),
    }


def payoff_at_expiry(row: pd.Series, prep: dict[str, Any]) -> tuple[float, float] | None:
    exp = pd.Timestamp(row["expiration"])
    idx = prep["date_to_index"].get(exp)
    if idx is None:
        return None
    st = float(prep["close"][idx])
    width = float(row["width"])
    if row["kind"] == "call":
        payoff = min(max(st - float(row["k1"]), 0.0), width)
    else:
        payoff = min(max(float(row["k2"]) - st, 0.0), width)
    return st, payoff


def assign_splits(year: int, cfg: dict[str, Any]) -> str:
    for name, years in cfg["splits"].items():
        if year in years:
            return name
    return "other"


def select_portfolio_and_score(
    candidates: pd.DataFrame,
    spots: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    prepared = {root: prepare_underlying(spots, ticker) for root, ticker in cfg["assets"].items()}
    accepted: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    max_positions = int(cfg["maximum_open_positions"])
    max_direction = int(cfg["maximum_same_direction_positions"])
    max_asset = int(cfg["maximum_positions_per_asset"])

    for entry_date, day in candidates.groupby("entry_date", sort=True):
        entry_date = pd.Timestamp(entry_date)
        open_positions = [p for p in open_positions if pd.Timestamp(p["expiration"]) >= entry_date]
        for row in day.sort_values("score", ascending=False).itertuples(index=False):
            r = row._asdict()
            if len(open_positions) >= max_positions:
                continue
            if sum(1 for p in open_positions if p["root"] == r["root"]) >= max_asset:
                continue
            if sum(1 for p in open_positions if p["direction"] == r["direction"]) >= max_direction:
                continue
            prep = prepared.get(r["root"])
            if prep is None:
                continue
            expiry = payoff_at_expiry(pd.Series(r), prep)
            if expiry is None:
                continue
            expiry_spot, payoff = expiry
            r["expiry_spot"] = expiry_spot
            r["intrinsic_payoff_unit"] = payoff
            r["split"] = assign_splits(pd.Timestamp(r["entry_date"]).year, cfg)
            for scenario, scenario_cfg in cfg["cost_scenarios"].items():
                entry_price = float(r[f"entry_{scenario}"])
                fixed = float(scenario_cfg["fixed_brl"])
                if r["side"] == "long":
                    pnl = (payoff - entry_price) * 100.0 - fixed
                else:
                    pnl = (entry_price - payoff) * 100.0 - fixed
                r[f"pnl_{scenario}"] = pnl
            accepted.append(r)
            open_positions.append(r)
    return pd.DataFrame(accepted)


def equity_drawdown(trades: pd.DataFrame, pnl_col: str, capital: float) -> tuple[float, pd.Series]:
    if trades.empty:
        return math.nan, pd.Series(dtype=float)
    daily = trades.groupby("expiration")[pnl_col].sum().sort_index()
    curve = capital + daily.cumsum()
    drawdown = curve - curve.cummax()
    return float(drawdown.min()), curve


def probabilistic_sharpe(daily_returns: np.ndarray, benchmark: float = 0.0) -> float:
    x = np.asarray(daily_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30 or np.std(x, ddof=1) <= 0:
        return math.nan
    sr = np.mean(x) / np.std(x, ddof=1) * math.sqrt(252.0)
    sk = float(skew(x, bias=False))
    ku = float(kurtosis(x, fisher=False, bias=False))
    denom = math.sqrt(max(1e-12, 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark) * math.sqrt(len(x) - 1.0) / denom
    return float(norm.cdf(z))


def bootstrap_cluster_stats(trades: pd.DataFrame, pnl_col: str, iterations: int = 4000) -> tuple[float, float, float]:
    if trades.empty:
        return math.nan, math.nan, math.nan
    clustered = trades.groupby("expiration")[pnl_col].agg(["sum", "count"])
    vals = clustered["sum"].to_numpy(dtype=float)
    counts = clustered["count"].to_numpy(dtype=float)
    if len(vals) < 5:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(20260912)
    means = np.empty(iterations)
    observed = float(np.sum(vals) / np.sum(counts))
    for i in range(iterations):
        pick = rng.integers(0, len(vals), len(vals))
        means[i] = np.sum(vals[pick]) / max(1.0, np.sum(counts[pick]))
    lo, hi = np.quantile(means, [0.025, 0.975])
    sign_sims = np.sum(vals * rng.choice([-1.0, 1.0], size=(iterations, len(vals))), axis=1)
    pvalue = float((1.0 + np.sum(sign_sims >= np.sum(vals))) / (iterations + 1.0))
    return float(lo), float(hi), pvalue


def metrics_for_subset(trades: pd.DataFrame, pnl_col: str, capital: float) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0}
    pnl = trades[pnl_col].astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    max_dd, _ = equity_drawdown(trades, pnl_col, capital)
    start = pd.Timestamp(trades["entry_date"].min())
    end = pd.Timestamp(trades["expiration"].max())
    business = pd.bdate_range(start, end)
    realized = trades.groupby("expiration")[pnl_col].sum().reindex(business, fill_value=0.0) / capital
    std = float(realized.std(ddof=1))
    sharpe = float(realized.mean() / std * math.sqrt(252.0)) if std > 0 else math.nan
    downside = realized[realized < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else math.nan
    sortino = float(realized.mean() / dstd * math.sqrt(252.0)) if np.isfinite(dstd) and dstd > 0 else math.nan
    lo, hi, pvalue = bootstrap_cluster_stats(trades, pnl_col)
    return {
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "return_on_50k_pct": float(pnl.sum() / capital * 100.0),
        "average_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "hit_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": float(gp / gl) if gl > 0 else math.inf,
        "max_realized_drawdown": max_dd,
        "sharpe_daily_realized": sharpe,
        "sortino_daily_realized": sortino,
        "probabilistic_sharpe_gt_zero": probabilistic_sharpe(realized.to_numpy()),
        "bootstrap_mean_trade_ci_low": lo,
        "bootstrap_mean_trade_ci_high": hi,
        "cluster_signflip_pvalue_one_sided": pvalue,
        "start": str(start.date()),
        "end": str(end.date()),
    }


def build_reports(trades: pd.DataFrame, candidates: pd.DataFrame, diagnostics: dict[str, Any], manifest: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    capital = float(cfg["capital_brl"])
    metric_rows: list[dict[str, Any]] = []
    for scenario in cfg["cost_scenarios"]:
        pnl_col = f"pnl_{scenario}"
        for split in ["all", "development", "validation", "holdout"]:
            subset = trades if split == "all" else trades[trades["split"] == split]
            row = {"scenario": scenario, "split": split}
            row.update(metrics_for_subset(subset, pnl_col, capital))
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    breakdown_rows: list[dict[str, Any]] = []
    if not trades.empty:
        for scenario in cfg["cost_scenarios"]:
            pnl_col = f"pnl_{scenario}"
            for (asset, year), g in trades.groupby(["underlying", trades["entry_date"].dt.year]):
                breakdown_rows.append(
                    {
                        "scenario": scenario,
                        "underlying": asset,
                        "year": int(year),
                        "trades": int(len(g)),
                        "net_pnl": float(g[pnl_col].sum()),
                        "average_pnl": float(g[pnl_col].mean()),
                        "hit_rate_pct": float((g[pnl_col] > 0).mean() * 100.0),
                    }
                )
    breakdown = pd.DataFrame(breakdown_rows)

    robustness: dict[str, Any] = {}
    if not trades.empty:
        base = "pnl_base"
        by_asset = trades.groupby("underlying")[base].sum().sort_values(ascending=False)
        by_year = trades.groupby(trades["entry_date"].dt.year)[base].sum().sort_values(ascending=False)
        best_asset = str(by_asset.index[0])
        best_year = int(by_year.index[0])
        robustness = {
            "base_net_all": float(trades[base].sum()),
            "best_asset": best_asset,
            "net_excluding_best_asset": float(trades.loc[trades["underlying"] != best_asset, base].sum()),
            "best_year": best_year,
            "net_excluding_best_year": float(trades.loc[trades["entry_date"].dt.year != best_year, base].sum()),
            "largest_asset_profit_share": float(by_asset.iloc[0] / trades[base].sum()) if trades[base].sum() > 0 else math.nan,
            "largest_year_profit_share": float(by_year.iloc[0] / trades[base].sum()) if trades[base].sum() > 0 else math.nan,
        }

    trades.to_csv(OUT / "accepted_trades.csv", index=False)
    candidates.to_csv(OUT / "all_open_qualified_candidates.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    breakdown.to_csv(OUT / "asset_year_breakdown.csv", index=False)
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "config_snapshot.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    if not trades.empty:
        for scenario in cfg["cost_scenarios"]:
            pnl_col = f"pnl_{scenario}"
            daily = trades.groupby("expiration")[pnl_col].sum().sort_index()
            curve = capital + daily.cumsum()
            plt.figure(figsize=(10, 5))
            plt.plot(curve.index, curve.values)
            plt.title(f"ATLAS Direct-QP v0.1 — Equity realizada — {scenario}")
            plt.xlabel("Data de vencimento")
            plt.ylabel("Capital (R$)")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(OUT / f"equity_{scenario}.png", dpi=160)
            plt.close()

    summary = {
        "system": cfg["system_name"],
        "generated_candidates": int(len(candidates)),
        "accepted_trades": int(len(trades)),
        "metrics": metric_rows,
        "robustness": robustness,
        "diagnostics": diagnostics,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# ATLAS Direct-QP v0.1 — primeiro backtest congelado",
        "",
        f"- Candidatos executáveis pela abertura: **{len(candidates)}**",
        f"- Operações aceitas após limites de portfólio: **{len(trades)}**",
        f"- Capital de referência: **R$ {capital:,.2f}**",
        "- Entrada: abertura da sessão seguinte ao sinal.",
        "- Saída: valor intrínseco no vencimento.",
        "- O preço de abertura de cada perna é assíncrono; por isso os cenários stress e severe são decisivos.",
        "",
        "## Métricas",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Robustez de concentração (cenário base)",
        "",
        "```json",
        json.dumps(robustness, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Diagnóstico da geração de sinais",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    (OUT / "ATLAS_v01_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    random.seed(20260912)
    np.random.seed(20260912)
    cfg = load_config()
    print("CONFIG_SHA256=" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), flush=True)
    spots, options, manifest = extract_market_data(cfg)
    candidates, diagnostics = generate_candidates(spots, options, cfg)
    print(f"CANDIDATES_AFTER_OPEN_FILTER={len(candidates)}", flush=True)
    trades = select_portfolio_and_score(candidates, spots, cfg) if not candidates.empty else pd.DataFrame()
    print(f"ACCEPTED_TRADES={len(trades)}", flush=True)
    summary = build_reports(trades, candidates, diagnostics, manifest, cfg)
    compact = {
        "accepted_trades": summary["accepted_trades"],
        "generated_candidates": summary["generated_candidates"],
        "metrics": summary["metrics"],
        "robustness": summary["robustness"],
    }
    print("ATLAS_FINAL_SUMMARY=" + json.dumps(compact, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
