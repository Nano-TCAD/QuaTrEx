# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pydantic import BaseModel
import toml


class SimulationParameters(BaseModel):
    blocksize: int
    conduction_band_energy: float
    temperature: float
    solver_mode: str
    energy_grid: dict
    fermi_levels: dict
