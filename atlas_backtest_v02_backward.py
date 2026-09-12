from __future__ import annotations

from pathlib import Path

import atlas_backtest_v01_retry as retry

atlas = retry.atlas
atlas.CONFIG_PATH = atlas.ROOT / "atlas_config_v02_backward.json"
atlas.OUT = atlas.ROOT / "atlas_results_v02_backward"
atlas.OUT.mkdir(exist_ok=True)

_original_evaluate_verticals = atlas.evaluate_verticals


def evaluate_short_call_credit_only(chain, models, spot, horizon, kind, cfg):
    if kind != "call":
        return []
    results = _original_evaluate_verticals(chain, models, spot, horizon, kind, cfg)
    return [item for item in results if item["side"] == "short"]


atlas.evaluate_verticals = evaluate_short_call_credit_only

if __name__ == "__main__":
    atlas.main()
    old_report = atlas.OUT / "ATLAS_v01_report.md"
    new_report = atlas.OUT / "ATLAS_v02_backward_report.md"
    if old_report.exists():
        old_report.replace(new_report)
