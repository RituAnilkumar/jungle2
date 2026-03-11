import xarray as xr
import glob

# pick only the daily files you want
tas_files  = sorted(glob.glob("w5e5daily/tas_W5E5v2.0_*.nc"))
pr_files   = sorted(glob.glob("w5e5daily/pr_W5E5v2.0_*.nc"))
rsds_files = sorted(glob.glob("w5e5daily/rsds_W5E5v2.0_*.nc"))

# good default chunking for big global grids; tweak to your HPC + analysis pattern
chunks = {"time": 31, "lat": 360, "lon": 720}

def open_var(files):
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks=chunks,
        data_vars="minimal",
        coords="minimal", 
        engine=None, 
        compat="override",
    )

tas  = open_var(tas_files)
pr   = open_var(pr_files)
rsds = open_var(rsds_files)

tas_da=tas['tas']
rsds_da=rsds['rsds']
pr_da=pr['pr']

def monthly_stats(da):
    g = da.resample(time="MS")
    return xr.Dataset({
        "mean":   g.mean("time"),
        "min":    g.min("time"),
        "max":    g.max("time"),
        "median": g.median("time"),
        "std":    g.std("time"),
    })

tas_m  = monthly_stats(tas_da).rename({k: f"tas_{k}"  for k in ["mean","min","max","median","std"]})
rsds_m = monthly_stats(rsds_da).rename({k: f"rsds_{k}" for k in ["mean","min","max","median","std"]})

# pr: kg m-2 s-1 -> mm/day then sum to mm/month
pr_mm_month = (pr_da * 86400.0).resample(time="MS").sum("time").rename("pr_monthly_total").to_dataset()
pr_mm_month["pr_monthly_total"].attrs["units"] = "mm month-1"

monthly = xr.merge([tas_m, rsds_m, pr_mm_month], compat="override")

monthly.to_netcdf("w5e5aggregrates/W5E5_monthly_aggregates.nc")