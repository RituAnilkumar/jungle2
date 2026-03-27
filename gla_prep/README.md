# gla_prep

Prepares glacier climate features and mass balance targets for the modelling
directory. Features and targets are independent:

- `main_features_rNN.csv` — **always produced**; all RGI glaciers × all years in the met file; columns: `rgi_id, year, [RGI static attrs], [climate features]`
- `{target}_targets_rNN.csv` — per-glacier annual MB (`target=oggm` or `target=wgms`)
- `glambie_targets_rNN.csv` — regional sums (`target=glambie`, or `target=temporal_avg` with `glambie_path` set)
- `temporal_avg_targets_rNN.csv` — multi-year window averages (`target=temporal_avg`)

---

## Quick start

**Step 1 — run met_prep once** (produces the climate NetCDF for all regions):

```bash
cd ../met_prep
python main.py dataset=era5 aggregation=seasonal
```

**Step 2 — run gla_prep for a region:**

```bash
cd ../gla_prep
python main.py target=temporal_avg region=r06
```

Outputs land in `output_path/{target}/` (set in [conf/config.yaml](conf/config.yaml)):

```
<output_path>/temporal_avg/main_features_r06.csv           ← all glaciers × all met years
<output_path>/temporal_avg/temporal_avg_targets_r06.csv    ← Hugonnet window averages
```

**Multiple regions at once:**

```bash
python main.py -m target=temporal_avg region=r01,r02,r06,r07
```

> Features always cover the full met file year range regardless of target.
> Targets are written as separate files and are not merged into `main_features`.

---

## Targets

`main_features_rNN.csv` is always produced regardless of target (full RGI × met years).
The target only controls what additional file is written alongside it.

| Target | Additional output | Source |
|---|---|---|
| `oggm` | `oggm_targets_rNN.csv` — `rgi_id, year, mass_balance` | OGGM L5 fixed-geometry summary CSVs (auto-downloadable) |
| `wgms` | `wgms_targets_rNN.csv` — `rgi_id, year, mass_balance` | WGMS |
| `glambie` | `glambie_targets_rNN.csv` — `region, year, regional_sum, uncertainty` | User-provided GLAMBIE CSV |
| `temporal_avg` | `temporal_avg_targets_rNN.csv` — `rgi_id, start_date, end_date, avg_mb_mwe, avg_mb_gt, uncertainty_mwe, uncertainty_gt` | Hugonnet et al. (2021) cumulative dh |
| `custom` | `custom_targets_rNN.csv` | User-supplied CSV |

Switch target from the CLI:

```bash
python main.py target=oggm region=r06
python main.py target=glambie region=r06
python main.py target=temporal_avg region=r06
```

---

## Regions

One YAML per RGI region in [conf/region/](conf/region/) (`r01`–`r19`).
Each file sets `rgi_code`, `rgi_attr_path`, and `rgi_format`.

The `temporal_avg` target resolves the Hugonnet file path automatically
from `rgi_code` via Hydra interpolation in
[conf/target/temporal_avg.yaml](conf/target/temporal_avg.yaml).

---

## Configuration

| File | Purpose |
|---|---|
| [conf/config.yaml](conf/config.yaml) | Root: paths, met frequency, RGI columns |
| [conf/target/temporal_avg.yaml](conf/target/temporal_avg.yaml) | Hugonnet path, window size |
| [conf/target/oggm.yaml](conf/target/oggm.yaml) | OGGM summary directory |
| [conf/region/rNN.yaml](conf/region/) | Per-region RGI file paths |

Override any value from the CLI:

```bash
python main.py target=temporal_avg region=r06 target.window_years=10
python main.py target=temporal_avg region=r06 output_path=/my/outputs
```
