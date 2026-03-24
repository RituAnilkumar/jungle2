# gla_prep

Prepares glacier mass balance targets and climate features for the modelling
directory. For each RGI region it produces:

- `main_features_rNN.csv` — per-glacier × year climate + RGI static attributes
- `temporal_avg_targets_rNN.csv` — multi-annual window averages (if `target=temporal_avg`)
- `glambie_targets_rNN.csv` — regional sums (if GLAMBIE path is set)

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

Outputs land in `output_path` (set in [conf/config.yaml](conf/config.yaml)):

```
<output_path>/main_features_r06.csv
<output_path>/temporal_avg_targets_r06.csv
```

**Multiple regions at once:**

```bash
python main.py -m target=temporal_avg region=r01,r02,r06,r07
```

---

## Targets

| Target | Source | Per-glacier annual MB? | Multi-annual averages? |
|---|---|---|---|
| `temporal_avg` | Hugonnet et al. (2021) cumulative dh | yes (internal) | yes |
| `oggm` | OGGM fixed-geometry summary CSVs | yes | no |
| `wgms` | WGMS | yes | no |
| `glambie` | GLAMBIE regional sums | no (regional only) | no |
| `custom` | User-supplied CSV | configurable | no |

Switch target from the CLI:

```bash
python main.py target=oggm region=r06
python main.py target=wgms region=r06
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
