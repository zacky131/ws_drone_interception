"""
src/control/capture_time_mpc/horizon_config.py

Phase 2: Common Horizon Configuration Interface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any

@dataclass
class HorizonSpecification:
    name: str
    grid_type: str  # "fixed" or "nonuniform"
    node_dts: List[float] = field(default_factory=list)
    
    @property
    def N(self) -> int:
        return len(self.node_dts)
        
    @property
    def total_duration(self) -> float:
        return float(sum(self.node_dts))
        
    @property
    def cumulative_times(self) -> np.ndarray:
        return np.cumsum(self.node_dts)

def get_horizon_spec(variant_name: str) -> HorizonSpecification:
    """Factory creating HorizonSpecification for standard diagnostic variants."""
    if "fixed_0p4s" in variant_name:
        dts = [0.02] * 20
        return HorizonSpecification(name="fixed_0p4s", grid_type="fixed", node_dts=dts)
    elif "fixed_1p0s" in variant_name:
        dts = [0.02] * 50
        return HorizonSpecification(name="fixed_1p0s", grid_type="fixed", node_dts=dts)
    elif "fixed_2p0s" in variant_name:
        dts = [0.02] * 100
        return HorizonSpecification(name="fixed_2p0s", grid_type="fixed", node_dts=dts)
    elif "nonuniform_3p0s" in variant_name:
        # Preferred nonuniform grid:
        # 0.00-0.40s: 20 ms intervals (20 steps)
        # 0.40-1.20s: 50 ms intervals (16 steps)
        # 1.20-3.00s: 100 ms intervals (18 steps)
        dts = [0.02] * 20 + [0.05] * 16 + [0.10] * 18
        return HorizonSpecification(name="nonuniform_3p0s", grid_type="nonuniform", node_dts=dts)
    else:
        raise ValueError(f"Unknown horizon variant: {variant_name}")
