# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
prepare_data_for_train_production.py (optional)
A thin wrapper to expose both "resample→then cut" and "cut→then resample".
Use ONLY resample→cut when you need exact parity with the simulator.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

class PREPARE_DATA_FOR_TRAIN_PRODUCTION(PREPARE_DATA_FOR_TRAIN):
    def __init__(
        self,
        filepaths: Optional[Dict[str, str]] = None,
        main_timeframe: str = "30T",
        verbose: bool = False,
        fast_mode: bool = False,
        strict_disk_feed: bool = False,
    ):
        super().__init__(filepaths=filepaths, main_timeframe=main_timeframe,
                         verbose=verbose, fast_mode=fast_mode, strict_disk_feed=strict_disk_feed)

    def load_data_resample_first_then_cut(self, ts_now) -> pd.DataFrame:
        """EXACT simulator parity. Needs full raw CSVs."""
        merged = self.load_data()
        tcol = f"{self.main_timeframe}_time"
        merged[tcol] = pd.to_datetime(merged[tcol])
        merged = merged[merged[tcol] <= pd.to_datetime(ts_now)].copy()
        merged.sort_values(tcol, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def load_data_cut_then_resample(self, ts_now) -> pd.DataFrame:
        """When you only have tail slices and must cut raw before resample."""
        # NOTE: Implement by copying your earlier production logic if ever needed.
        raise NotImplementedError("Cut→then resample mode intentionally omitted for parity.")
