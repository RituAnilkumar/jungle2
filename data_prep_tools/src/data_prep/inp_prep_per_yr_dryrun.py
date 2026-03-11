# Preferred mode of run for preparing meteorological datasets and gee input data and debris data. To include albedo

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import re, os
import ee, geemap
ee.Initialize(project='graphic-boulder-279718')

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape
import chardet

def run_input_preparation_one_year(input_cfg: DictConfig) -> gpd.GeoDataFrame:
    """
    ONE-YEAR-AT-A-TIME VERSION
    - Assumes start_datetime and end_datetime bound a single calendar year.
    - No feature duplication by year; no per-feature 'year' attribute.
    - Builds year images (monthly-stacked) once and samples/reduces over all features.
    """

    # --- Temporal validation (keep as in your file) ---
    start_dt = input_cfg.time_ranges[input_cfg.time_indx].start_datetime
    end_dt   = input_cfg.time_ranges[input_cfg.time_indx].end_datetime
    gee_start_date = ee.Date.parse('YYYY-MM-dd\'T\'HH:mm:ss', start_dt)
    gee_end_date   = ee.Date.parse('YYYY-MM-dd\'T\'HH:mm:ss', end_dt)

    # --- Frequencies (kept) ---
    freq_download = input_cfg.temporal_freq_download         # "M","D","H"
    temporal_freq_agg = input_cfg.temporal_freq_aggregate    # "M","Y",None

    # --- RGI loading (unchanged in spirit, but no year-repeat) ---
    def drop_z(geom):
        if geom.is_empty: return geom
        g = mapping(geom)
        def _drop_z_rec(c):
            if isinstance(c[0], (float, int)): return tuple(c[:2])
            return [ _drop_z_rec(i) for i in c ]
        g["coordinates"] = _drop_z_rec(g["coordinates"])
        return shape(g)

    rgi_data_provider = input_cfg.rgi_data_provider  # "rgi_csv" or "rgi_shp"
    rgi_attr_path = input_cfg.rgi_attr_path
    debris_fill_mode = input_cfg.debris_fill_mode
    debris_path = input_cfg.debris_path
    use_debris = input_cfg.use_debris

    if rgi_data_provider == "rgi_csv":
        with open(rgi_attr_path,'rb') as f:
            # print('entered')
            enc=chardet.detect(f.read())
        rgi_df = pd.read_csv(rgi_attr_path, encoding=enc["encoding"])

        # rgi_df = pd.read_csv(rgi_attr_path)
        if (debris_fill_mode in ("median","mean")) and use_debris:
            deb_gdf = gpd.read_file(debris_path)
            rgi_df = rgi_df.merge(deb_gdf[["GLIMSId","perc_deb"]], on="GLIMSId", how="left")
            fillval = rgi_df["perc_deb"].median() if debris_fill_mode=="median" else rgi_df["perc_deb"].mean()
            rgi_df["perc_deb"] = rgi_df["perc_deb"].fillna(fillval)
        rgi_df = rgi_df.rename(columns={"RGIId":"rgi_id","CenLon":"cenlon","CenLat":"cenlat"})
        rgi_gdf = gpd.GeoDataFrame(rgi_df, geometry=gpd.points_from_xy(rgi_df.cenlon, rgi_df.cenlat), crs="EPSG:4326")
    elif rgi_data_provider == "rgi_shp":
        rgi_gdf = gpd.read_file(rgi_attr_path)
        rgi_gdf = rgi_gdf[rgi_gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
        rgi_gdf = rgi_gdf[~rgi_gdf.geometry.is_empty & rgi_gdf.geometry.notnull()].copy()
        rgi_gdf["geometry"] = rgi_gdf.geometry.make_valid().apply(drop_z) #type: ignore
        rgi_gdf = rgi_gdf.to_crs("EPSG:4326")
        rgi_gdf = rgi_gdf.rename(columns={"RGIId":"rgi_id","CenLon":"cenlon","CenLat":"cenlat"})
        if (debris_fill_mode in ("median","mean")) and use_debris:
            deb_gdf = gpd.read_file(debris_path)
            rgi_gdf = rgi_gdf.merge(deb_gdf[["GLIMSId","perc_deb"]], on="GLIMSId", how="left")
            fillval = rgi_gdf["perc_deb"].median() if debris_fill_mode=="median" else rgi_gdf["perc_deb"].mean()
            rgi_gdf["perc_deb"] = rgi_gdf["perc_deb"].fillna(fillval)

    rgi_cols = ["rgi_id","cenlat","cenlon","geometry"] + list(input_cfg.rgi_cols)
    if use_debris: rgi_cols.append("perc_deb")
    for col in rgi_cols:
        if col not in rgi_gdf.columns:
            raise ValueError(f"Missing RGI column: {col}")
    rgi_gdf_sel = rgi_gdf[rgi_cols].copy()

    # --- AOI (kept) ---
    area_mode = input_cfg.area_mode  # "reg_num","shp","bbox"
    if area_mode == "reg_num":
        rgi_code = input_cfg.reg_num.rgi_code
        rgi_regions = ee.FeatureCollection('projects/graphic-boulder-279718/assets/RGI2000_v7_o1regions')
        aoi = rgi_regions.filter(ee.Filter.eq('o1region', rgi_code)).geometry()
    elif area_mode == "shp":
        shp_path = input_cfg.shp.shp_path
        if not os.path.exists(shp_path): raise ValueError(f"Unsupported shp path: {shp_path}")
        aoi = geemap.shp_to_ee(shp_path).geometry() #type: ignore
    else:
        s, n, w, e = input_cfg.bbox.bbox_coords
        aoi = ee.Geometry.BBox(south=s, north=n, west=w, east=e)

    # --- Spatial aggregation target FC (no year-duplication) ---
    spatial_aggr_file = input_cfg.spatial_aggregation_file
    if spatial_aggr_file is not None:
        target_gdf = gpd.read_file(spatial_aggr_file).to_crs(4326)
    else:
        target_gdf = rgi_gdf_sel.copy()

    spatial_aggregation_opt = input_cfg.spatial_aggregation_opt  # "sampling","mean","median","min","max","sum", None
    attr_geom = input_cfg.attr_geom                                  # "pt" or "poly"
    if attr_geom == "pt" and spatial_aggregation_opt != "sampling":
        raise ValueError("Point input requires spatial_aggregation_opt='sampling'.")

    # If user asked "sampling" with polygon, use centroids implicitly (as in your file)
    if attr_geom == "poly" and spatial_aggregation_opt == "sampling":
        target_gdf = target_gdf.copy()
        target_gdf["geometry"] = gpd.points_from_xy(target_gdf["cenlon"], target_gdf["cenlat"])
        target_gdf.set_crs(4326, inplace=True)

    target_fc = geemap.geopandas_to_ee(target_gdf)

    # --- Helper to make monthly-suffixed names for a collection over the given year span ---
    def rename_by_month(img):
        month_str = img.date().format('MM')
        return img.rename(img.bandNames().map(lambda b: ee.String(b).cat('_').cat(month_str)))
    
    # Compute monthly std (σ) for ONE variable from an HOURLY collection.
    def monthly_std_12bands_for_var(hourly_ic, var_name, start_date, end_date, aoi):
        hourly_ic = hourly_ic.filterDate(start_date, end_date).filterBounds(aoi)
        year = ee.Date(start_date).get('year')
        months = ee.List.sequence(1, 12)

        def per_month(m):
            m = ee.Number(m)
            month_ic = hourly_ic.filter(ee.Filter.calendarRange(m, m, 'month'))
            # Std across hourly images for this var in this month (pixel-wise)
            std_im = month_ic.select([var_name]) \
                            .reduce(ee.Reducer.stdDev()) \
                            .rename(ee.String(var_name).cat('_std')) \
                            .clip(aoi)  # IMPORTANT: clip early to limit graph size

            # Stamp a stable monthly timestamp so rename_by_month can append _MM
            std_im = std_im.set('system:time_start', ee.Date.fromYMD(year, m, 1).millis())
            return std_im

        # 12 one-band images -> add _MM suffix -> toBands -> strip toBands prefixes
        ic_sigma = ee.ImageCollection(months.map(per_month)).map(rename_by_month).sort('system:time_start')
        img_sigma = ic_sigma.toBands()
        img_sigma = img_sigma.rename(
            img_sigma.bandNames().map(lambda b: ee.String(b).split('_').slice(1).join('_'))
        )
        # bands now: <var_name>_std_01 ... _12
        return img_sigma


    

    # --- Meteorological collection selection (kept logic, but single-year image stack) ---
    data_name = input_cfg.data_name  # "era5" or "era5land" or "others"
    if data_name == "era5":
        vars_ = input_cfg.era5.era5_vars
        if freq_download == "H": met = ee.ImageCollection("ECMWF/ERA5/HOURLY")
        elif freq_download == "D": met = ee.ImageCollection("ECMWF/ERA5/DAILY")
        else: met = ee.ImageCollection("ECMWF/ERA5/MONTHLY")
        met_scale = 27830
        met_sigma_imcol=ee.ImageCollection("ECMWF/ERA5/HOURLY")
        met_sigma_vars=input_cfg.sigma_vars
    elif data_name == "era5land":
        vars_ = input_cfg.era5land.era5land_vars
        if freq_download == "H": met = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        elif freq_download == "D": met = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        else: met = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        met_scale = 11132
        met_sigma_imcol=ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        met_sigma_vars=input_cfg.sigma_vars
        
    else:
        raise NotImplementedError("Simplified one-year path currently supports ERA5/ERA5-Land.")

    # Filter and select bands for the target year
    met_year_ic = met.filterDate(gee_start_date, gee_end_date).select(list(vars_)).sort('system:time_start')
    # Convert to monthly-suffixed multi-band image for that year
    met_year_img = met_year_ic.map(rename_by_month).toBands()
    # Strip toBands prefixes
    met_year_img = met_year_img.rename(met_year_img.bandNames().map(lambda b: ee.String(b).split('_').slice(1).join('_'))) # This is to ensure that the t2m_20200101T000000Z_01 becomes t2m_01

    # --- Other GEE dynamic/static collections (simplified) ---
    # Build a dict of {bandname: image or multi-band-year-image}, then sample/reduce once.
    stacked_images = {"met": met_year_img}

    # --- Monthly std (σ) from HOURLY met for requested vars ---
    # Uses: met_sigma_imcol (hourly IC), met_sigma_vars (list of band names)
    # --- Monthly std (σ) from HOURLY met for requested vars ---  (per-variable)
    if met_sigma_vars:
        for v in met_sigma_vars:
            # Build a 12-band σ image for just this variable
            v_sigma = monthly_std_12bands_for_var(
                hourly_ic=met_sigma_imcol,
                var_name=v,
                start_date=gee_start_date,
                end_date=gee_end_date,
                aoi=aoi
            )
            # Add bands under a unique key; they’ll be merged later into `big`
            stacked_images[f"{v}_sigma"] = v_sigma

    if input_cfg.gee_imcols:
        for dyn in input_cfg.gee_imcols:
            col_id = dyn.gee_imcol_id
            bands  = [b.band_name for b in dyn.gee_imcol_bands]
            make_static = [b.make_static for b in dyn.gee_imcol_bands]
            scales = [b.scale for b in dyn.gee_imcol_bands]

            ic = ee.ImageCollection(col_id).filterDate(gee_start_date, gee_end_date).filterBounds(aoi)
            # Process each requested band independently and add to the stack
            for bname, is_static, sc in zip(bands, make_static, scales):
                sel = ic.select(bname)
                if is_static:
                    img = sel.mean().rename(bname)  # single band
                else:
                    # monthly means within the same one-year window
                    monthly = sel.map(lambda im: im.set('month', im.date().format('MM'))) \
                                .map(rename_by_month) \
                                .sort('system:time_start')
                    img = monthly.toBands()
                    img = img.rename(img.bandNames().map(lambda b: ee.String(b).split('_').slice(1).join('_')))
                stacked_images[bname] = img

    # --- Merge all stacked images into one big multi-band year image ---
    # (Start from a constant to allow iterative addBands)
    big = ee.Image(0).rename('dummy')
    for k, img in stacked_images.items():
        big = big.addBands(img)
    big = big.select(big.bandNames().remove('dummy'))

    # --- Sample/reduce once over all features ---
    out_dir = HydraConfig.get().runtime.output_dir
    save = input_cfg.save_gee_data

    if attr_geom == "pt" or spatial_aggregation_opt == "sampling":
        # points -> sampleRegions
        sampled_fc = big.sampleRegions(collection=target_fc, scale=met_scale, geometries=True)
    else:
        # polygons -> reduceRegions with chosen reducer
        reducer_name = spatial_aggregation_opt or "mean"
        reducer = {
            "mean": ee.Reducer.mean(),
            "median": ee.Reducer.median(),
            "min": ee.Reducer.min(),
            "max": ee.Reducer.max(),
            "sum": ee.Reducer.sum()
        }[reducer_name]
        sampled_fc = big.reduceRegions(collection=target_fc, reducer=reducer, scale=met_scale)

    if save:
        # Optional: dump the big image stack for the year for debugging
        geemap.ee_export_image(big, os.path.join(out_dir, "big_year_stack.tif"), scale=met_scale, region=aoi)

    # Convert to GeoDataFrame (keeps geometry if present)
    gdf = geemap.ee_to_gdf(sampled_fc)

    # Add year column
    gdf['year']=pd.to_datetime(start_dt).year
    return gdf
