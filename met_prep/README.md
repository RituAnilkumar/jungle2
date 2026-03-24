# met_prep

Aggregates ERA5 (or W5E5/CMIP) climate data into monthly, seasonal, or annual
NetCDF files for use by **gla_prep**.

NH and SH season definitions are applied per grid cell based on latitude sign
and combined into a single output file — no need to run twice.

---

## Quick start

```bash
cd met_prep
python main.py dataset=era5 aggregation=seasonal
```

Output: `<output_path>/era5_seasonal.nc`

Set `output_path` in [conf/config.yaml](conf/config.yaml). Run **once** — the
output covers all RGI regions.

---

## Aggregation modes

| Mode | Time coordinate | Description |
|---|---|---|
| `seasonal` | `season_year` | Ablation + accumulation seasons per grid cell |
| `annual_hydro` | `hydro_year` | Hydrological years (NH: Oct–Sep, SH: Apr–Mar) |
| `annual_calendar` | `time` | Jan–Dec calendar years |
| `monthly` | `time` | Monthly statistics |

Switch mode from the CLI:

```bash
python main.py aggregation=annual_hydro
python main.py aggregation=monthly
```

---

## Season-year convention

Season-year label = the calendar year in which the season **ends**.

| Season | Months | season_year label |
|---|---|---|
| NH accumulation | Oct – Apr | year Apr falls in |
| NH ablation | May – Sep | year Sep falls in |
| SH accumulation | Apr – Oct | year Oct falls in |
| SH ablation | Nov – Mar | year Mar falls in |

Example: NH accumulation Oct 2000 – Apr 2001 → `season_year = 2001`.

---

## Configuration

| File | Purpose |
|---|---|
| [conf/config.yaml](conf/config.yaml) | Root: dataset, aggregation, output path |
| [conf/dataset/era5.yaml](conf/dataset/era5.yaml) | ERA5 input paths and variable names |
| [conf/dataset/w5e5.yaml](conf/dataset/w5e5.yaml) | W5E5 input paths |
| [conf/aggregation/seasonal.yaml](conf/aggregation/seasonal.yaml) | Season month definitions, temperature stats |
| [conf/aggregation/annual_hydro.yaml](conf/aggregation/annual_hydro.yaml) | Hydro-year start months |

Override any value from the CLI with Hydra syntax:

```bash
python main.py dataset=w5e5 aggregation=seasonal output_path=/my/path
python main.py -m aggregation=seasonal,annual_hydro   # multirun
```
