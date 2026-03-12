"""
Hemisphere inference for gla_prep.

Infers hemisphere from a glacier's CenLat. Used to select the correct
seasonal / hydrological-year definition when sampling climate features.
"""

from __future__ import annotations


def get_hemisphere(cenlat: float, cfg_hemisphere: str | None) -> str:
    """Return 'NH' or 'SH' for a glacier centroid latitude.

    Parameters
    ----------
    cenlat : float
        Glacier centroid latitude (CenLat from RGI).
    cfg_hemisphere : str | None
        Explicit override from region config. If set to 'NH' or 'SH',
        that value is used regardless of latitude. Set to null/None for
        automatic inference. Tropical glaciers near the equator may need
        an explicit override.
    """
    if cfg_hemisphere and str(cfg_hemisphere).upper() in ("NH", "SH"):
        return str(cfg_hemisphere).upper()
    return "NH" if cenlat > 0 else "SH"
