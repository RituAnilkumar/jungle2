# src/data_prep/input_prep.py

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import re
import os

import ee
import geemap
ee.Initialize(project='graphic-boulder-279718')

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping, shape

def run_input_preparation(input_cfg: DictConfig) -> gpd.GeoDataFrame:
    """
    Reads your input configuration, checks its validity and
    calls the right nested functions for preparing inputs. Divisions in the code are based on what info is extracted
    """
    ## --- Temporal information for all dynamic data --- ##
    start_dt = input_cfg.time_ranges[input_cfg.time_indx].start_datetime
    gee_start_date = ee.Date.parse('YYYY-MM-dd\'T\'HH:mm:ss', start_dt)
    end_dt = input_cfg.time_ranges[input_cfg.time_indx].end_datetime
    gee_end_date = ee.Date.parse('YYYY-MM-dd\'T\'HH:mm:ss', end_dt)
    # If the start and end times are not in the right format, raise error
    if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", start_dt):
        raise ValueError(f"Unsuported start date: {start_dt}")
    if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", end_dt):
        raise ValueError(f"Unsuported end date: {end_dt}")
    
    freq_download = input_cfg.temporal_freq_download
    # If freq_download not in list, raise error
    freq_download_list=["M", "D", "H"]
    if freq_download not in freq_download_list:
        raise ValueError(f"Unsuported temporal unit: {freq_download}")
    
    temporal_freq_agg = input_cfg.temporal_freq_aggregate
    # If temporal_freq_agg not in list, raise error
    temporal_freq_agg_list=["M", "Y",None]
    if temporal_freq_agg not in temporal_freq_agg_list:
        raise ValueError(f"Unsuported temporal unit: {temporal_freq_agg}")
    if temporal_freq_agg == "M":
        if freq_download not in ["H","D"]:
            raise ValueError(f"Frequency of download must be lower than temporal frequency of aggregation")
        # aggr_met_imcol=met_imcol_filter.map(aggregator_func) # Commenting as i will be using this in the met section
    else:
        if freq_download not in ["M","D","H"]:
            raise ValueError(f"Frequency of download must be lower than temporal frequency of aggregation")
        # aggr_met_imcol=met_imcol_filter.map(aggregator_func) # Commenting as i will be using this in the met section
    # else:
        # aggr_met_imcol=met_imcol_filter # Commenting as i will be using this in the met section
    
    ## --- Static datasets from RGI --- ##
    rgi_data_provider = input_cfg.rgi_data_provider
    # If stat data provider not 'rgi', raise error
    rgi_data_provider_list=["rgi_csv", "rgi_shp"]
    if rgi_data_provider not in rgi_data_provider_list:
        raise ValueError(f"Unsuported stat data provider: {rgi_data_provider}")

    rgi_attr_path = input_cfg.rgi_attr_path
    debris_fill_mode=input_cfg.debris_fill_mode
    debris_path = input_cfg.debris_path
    # If path is not valid, raise error
    if not os.path.exists(rgi_attr_path):
        raise ValueError(f"Unsuported rgi attribute path: {rgi_attr_path}")
    
    # If reading shp, we need to drop the z values in geometry
    def drop_z(geom):
        if geom.is_empty:
            return geom
        g = mapping(geom)
        coords = g["coordinates"]

        def _drop_z_rec(c):
            if isinstance(c[0], (float, int)):
                return tuple(c[:2])
            return [ _drop_z_rec(i) for i in c ]

        g["coordinates"] = _drop_z_rec(coords)
        return shape(g)
    
    if rgi_data_provider == "rgi_csv":
        rgi_df = pd.read_csv(rgi_attr_path)
        if debris_fill_mode == "median":
            deb_gdf= gpd.read_file(debris_path)
            rgi_df = rgi_df.merge(deb_gdf[['GLIMSId','perc_deb']], how="left", on="GLIMSId")
            # If null in joined df perc_deb, fill with mean
            rgi_df['perc_deb'] = rgi_df['perc_deb'].fillna(rgi_df['perc_deb'].median())
        elif debris_fill_mode == "mean":
            deb_gdf= gpd.read_file(debris_path)
            rgi_df = rgi_df.merge(deb_gdf[['GLIMSId','perc_deb']], how="left", on="GLIMSId")
            # If null in joined df perc_deb, fill with mean
            rgi_df['perc_deb'] = rgi_df['perc_deb'].fillna(rgi_df['perc_deb'].mean())
        else:
            pass
        # rename RGIId to rgi_id, CenLon to cenlon and CenLat to cenlat
        rgi_df = rgi_df.rename(columns={"RGIId": "rgi_id", "CenLon": "cenlon", "CenLat": "cenlat"})
        rgi_gdf = gpd.GeoDataFrame(rgi_df, geometry=gpd.points_from_xy(rgi_df.cenlon, rgi_df.cenlat), crs="EPSG:4326")
    elif rgi_data_provider == "rgi_shp":
        rgi_gdf = gpd.read_file(rgi_attr_path)
        rgi_gdf = rgi_gdf[rgi_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        # Drop rows with missing geometry
        rgi_gdf = rgi_gdf[~rgi_gdf.geometry.is_empty & rgi_gdf.geometry.notnull()].copy()
        rgi_gdf["geometry"] = rgi_gdf.geometry.make_valid()
        rgi_gdf["geometry"] = rgi_gdf.geometry.apply(drop_z) # type: ignore[attr-defined]
        rgi_gdf=rgi_gdf.to_crs("EPSG:4326")
        # rename RGIId to rgi_id, CenLon to cenlon and CenLat to cenlat
        rgi_gdf = rgi_gdf.rename(columns={"RGIId": "rgi_id", "CenLon": "cenlon", "CenLat": "cenlat"})
        if debris_fill_mode == "median":
            deb_gdf= gpd.read_file(debris_path)
            rgi_gdf = rgi_gdf.merge(deb_gdf[['GLIMSId','perc_deb']], how="left", on="GLIMSId")
            # If null in joined df perc_deb, fill with mean
            rgi_gdf['perc_deb'] = rgi_gdf['perc_deb'].fillna(rgi_gdf['perc_deb'].median())
        elif debris_fill_mode == "mean":
            deb_gdf= gpd.read_file(debris_path)
            rgi_gdf = rgi_gdf.merge(deb_gdf[['GLIMSId','perc_deb']], how="left", on="GLIMSId")
            # If null in joined df perc_deb, fill with mean
            rgi_gdf['perc_deb'] = rgi_gdf['perc_deb'].fillna(rgi_gdf['perc_deb'].mean())
        else:
            pass
    # RGI columns
    rgi_cols = ["rgi_id","cenlat","cenlon","perc_deb","geometry"]+list(input_cfg.rgi_cols)
    # print(rgi_cols)
    # If rgi cols are not valid, raise error
    # rgi_cols_list=["area_km2", "zmin_m", "zmax_m", "zmed_m", "slope_deg", "aspect_deg", "aspect_sec"] # Hardcodded columns
    rgi_cols_list=rgi_gdf.columns
    # If any element of rgi_cols not in list, raise error
    for col in rgi_cols:
        if col not in rgi_cols_list:
            raise ValueError(f"Unsuported static data provider: {col}")
    
    # Pick rgi columns and create a table with years
    rgi_gdf_sel = rgi_gdf[rgi_cols]
    # Create years column for the rgi data. First convert start_date into datetime
    start_yr=pd.to_datetime(start_dt).year
    end_yr=pd.to_datetime(end_dt).year
    years = list(range(start_yr, end_yr+1))
    nyears=len(years)
    rgi_gdf_sel_years = rgi_gdf_sel.loc[rgi_gdf_sel.index.repeat(nyears)].assign(year=list(years)*len(rgi_gdf_sel)).reset_index(drop=True)
    rgi_tosample_gee=geemap.geopandas_to_ee(gpd.GeoDataFrame(rgi_gdf_sel_years))

    ## --- Area of Interest/Boundary selection --- ##
    area_mode = input_cfg.area_mode
    # If area_mode not in list, raise error
    area_mode_list=["reg_num", "shp", "bbox"]
    if area_mode not in area_mode_list:
        raise ValueError(f"Unsuported area_mode: {area_mode}")
    
    if area_mode == "reg_num":
        rgi_code = input_cfg.reg_num.rgi_code
        # If reg_num not in list, raise error
        reg_num_list=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13","14","15", "16", "17", "18", "19", "20"]
        if rgi_code not in reg_num_list:
            raise ValueError(f"Unsuported rgi_code: {rgi_code}")
        rgi_regions=ee.FeatureCollection('projects/graphic-boulder-279718/assets/RGI2000_v7_o1regions')
        aoi=rgi_regions.filter(ee.Filter.eq('o1region', rgi_code)).geometry()
    elif area_mode == "shp": 
        shp_path = input_cfg.shp.shp_path
        # If path is not valid, raise error
        if not os.path.exists(shp_path):
            raise ValueError(f"Unsuported shp path: {shp_path}")
        aoi=geemap.shp_to_ee(shp_path).geometry() # type: ignore[attr-defined]
    else:
        bbox_coords = input_cfg.bbox.bbox_coords
        # If bbox items dont follow the bounds of latitude and longitude, raise error
        print(bbox_coords[0])
        print(bbox_coords[1])
        print(bbox_coords[2])
        print(bbox_coords[3])
        if (bbox_coords[0] < -90) or (bbox_coords[0] > 90):
            raise ValueError(f"Unsuported latitude value: {bbox_coords[0]}")
        if (bbox_coords[1] < -90) or (bbox_coords[1] > 90):
            raise ValueError(f"Unsuported latitude value: {bbox_coords[1]}")
        if (bbox_coords[2] < -180) or (bbox_coords[2] > 180):
            raise ValueError(f"Unsuported longitude value: {bbox_coords[2]}")
        if (bbox_coords[3] < -180) or (bbox_coords[3] > 180):
            raise ValueError(f"Unsuported longitude value: {bbox_coords[3]}")
        aoi=ee.Geometry.BBox(south=bbox_coords[0], north=bbox_coords[1], west=bbox_coords[2], east=bbox_coords[3])

    ## --- Set up the temporal aggregation --- ##
    def monthly_aggr_met(m):
        print("Not implemented temporal aggregations yet")
    
    def annual_aggr_met(m):
        print("Not implemented temporal aggregations yet")

    ## --- Set up the spatial aggregation --- ##
    spatial_aggr_file = input_cfg.spatial_aggregation_file
    if spatial_aggr_file is not None:
        # If path is not valid, raise error
        if not os.path.exists(spatial_aggr_file):
            raise ValueError(f"Unsuported spatial aggregation file path: {spatial_aggr_file}")
        spatial_aggr_file_gpd = gpd.read_file(spatial_aggr_file)
        # Add years as a column to the spatial_aggr_file_gpd
        spatial_aggr_file_gpd_years=spatial_aggr_file_gpd.loc[spatial_aggr_file_gpd.index.repeat(nyears)].assign(year=list(years)*len(spatial_aggr_file_gpd)).reset_index(drop=True)
        spatial_aggr_gee=geemap.geopandas_to_ee(gpd.GeoDataFrame(spatial_aggr_file_gpd_years))
    else:
        spatial_aggr_gee=rgi_tosample_gee

    spatial_aggregation_opt = input_cfg.spatial_aggregation_opt
    # If spatial_aggregation_opt not in list, raise error
    spatial_aggregation_opt_list=["sampling", "mean", "min", "max", "median", "sum",None]
    if spatial_aggregation_opt not in spatial_aggregation_opt_list:
        raise ValueError(f"Unsuported spatial aggregation option: {spatial_aggregation_opt}")
    
    attr_geom = input_cfg.attr_geom
    # If attribute geometry not in list, raise error
    attr_geom_list=["pt", "poly"]
    if attr_geom not in attr_geom_list:
        raise ValueError(f"Unsuported attribute geometry: {attr_geom}")
    if attr_geom == "pt":
        if spatial_aggregation_opt != "sampling":
            raise ValueError(f"Point shapefile can work only with sampling aggregation option")
        # Sample using sampleRegions for the centroid of the shapefile geometry
        # sampled_fc=rgi_tosample_gee.map(sample_by_ft)
        spatial_aggr_fin=spatial_aggr_gee
    else:
        if spatial_aggregation_opt == "sampling": 
            print("Sampling aggregation option with polygon defaults to centroid sampling")
            rgi_gdf_sel_years["geometry"] = gpd.points_from_xy(rgi_gdf_sel_years["cenlon"],rgi_gdf_sel_years["cenlat"])
            rgi_gdf_sel_years=gpd.GeoDataFrame(rgi_gdf_sel_years,geometry="geometry")
            rgi_gdf_sel_years.set_crs("EPSG:4326", inplace=True)
            spatial_aggr_fin=geemap.geopandas_to_ee(rgi_gdf_sel_years)
        else:
            spatial_aggr_fin=spatial_aggr_gee
            # print('test',spatial_aggr_fin.first().geometry().type().getInfo())
            

    ## --- Meteorological data checks and sampling --- ##
    data_provider = input_cfg.data_provider
    # If data provider not in list, raise error
    data_provider_list=["gee", "cds"]
    if data_provider not in data_provider_list:
        raise ValueError(f"Unsuported data provider: {data_provider}")
    
    data_name = input_cfg.data_name
    # If met data not in list, raise error
    data_name_list=["era5", "era5land", "others"]
    if data_name not in data_name_list:
        raise ValueError(f"Unsuported data name: {data_name}")
    
    # Based on data name, assign variables
    if data_name == "era5":
        era5_vars=input_cfg.era5.era5_vars
        if freq_download == "H":
            era5_imcol=ee.ImageCollection("ECMWF/ERA5/HOURLY")
        elif freq_download == "D":
            era5_imcol=ee.ImageCollection("ECMWF/ERA5/DAILY")
        elif freq_download == "M":
            era5_imcol=ee.ImageCollection("ECMWF/ERA5/MONTHLY")

        era5_bands=era5_imcol.first().bandNames().getInfo()
        # If era5 vars are not in the list of band names, raise error
        for var in era5_vars:
            if var not in era5_bands:
                raise ValueError(f"Unsuported variable: {var}")
        met_imcol=era5_imcol
        met_scale=27830
        met_vars=era5_vars
        met_sigma_imcol=ee.ImageCollection("ECMWF/ERA5/HOURLY")
        met_sigma_vars=input_cfg.sigma_vars

    elif data_name == "era5land":
        era5land_vars=input_cfg.era5land.era5land_vars
        if freq_download == "H":
            era5land_imcol=ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        elif freq_download == "D":
            era5land_imcol=ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        elif freq_download == "M":
            era5land_imcol=ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

        era5land_bands=era5land_imcol.first().bandNames().getInfo()
        # If era5 vars are not in the list of band names, raise error
        for var in era5land_vars:
            if var not in era5land_bands:
                raise ValueError(f"Unsuported variable: {var}")
        met_imcol=era5land_imcol
        met_scale=11132 # meters per pixel
        met_vars=era5land_vars
        met_sigma_imcol=ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        met_sigma_vars=input_cfg.sigma_vars
    else:
        others_paths=input_cfg.others.others_paths
        # Check if paths are valid
        for path in others_paths:
            if not os.path.exists(path):
                raise ValueError(f"Unsuported path: {path}")
        # Pick the parent folder of the paths
        met_im_folder = others_paths[0]
            
    # Prepare met data from era5/era5l for download
    met_imcol_filter = met_imcol.filterDate(gee_start_date,gee_end_date).select(list(met_vars))
    met_sigma_imcol_filter = met_sigma_imcol.filterDate(gee_start_date,gee_end_date).select(list(met_sigma_vars))

    # Variance computation for met_sigmas
    
    
    out_dir = HydraConfig.get().runtime.output_dir
    # Create directory for met data
    met_im_folder = os.path.join(out_dir, "met_data")
    if input_cfg.save_gee_data:
        geemap.ee_export_image_collection(met_imcol_filter, met_im_folder, scale=met_scale, region=aoi, file_per_band=False)

    ## --- Sampling from the met data --- ##
    # Rename each image's bands with a month suffix: temp_01, pr_01, ...
    def rename_by_month(img):
        date = img.date()
        month_str = date.format('MM')  # "01", "02", ...
        band_names = img.bandNames()
        new_names = band_names.map(
            lambda b: ee.String(b).cat('_').cat(month_str)
        )
        return img.rename(new_names)
    def year_image_from_met(yr):
        """
        Creates a multi-band image from the met data for a given year.

        Parameters:
            yr (ee.Number): The year of interest.

        Returns:
            ee.Image: A multi-band image containing the met data for the year.
        """
        yr = ee.Number(yr).toInt()
        start = ee.Date.fromYMD(yr, 1, 1)
        end   = start.advance(1, 'year')

        year_ic = met_imcol_filter \
            .filterDate(start, end) \
            .sort('system:time_start')  # make sure months are in order

        # Rename each image's bands with a month suffix: temp_01, pr_01, ...
        year_ic = year_ic.map(rename_by_month)

        # Stack all images into one multi-band image
        year_img = year_ic.toBands()

        # Now strip the leading "imageId_" prefix that toBands adds
        band_names = year_img.bandNames()

        def strip_prefix(b):
            b = ee.String(b)
            # Split on "_" and drop the first token (the imageId like "200004")
            parts = b.split('_')
            return parts.slice(1).join('_')

        clean_names = band_names.map(strip_prefix)

        return year_img.rename(clean_names)
    def sample_met(f):
        # Get year from feature (cast to Number to be safe)
        """
        Add meteorological data to a feature.
        
        Parameters:
        f (ee.Feature): The feature to which we add meteorological data.
        
        Returns:
        ee.Feature: The feature with added meteorological data.
        """
        yr = ee.Number(f.get('year'))

        # Build the year's multi-band image
        year_img = year_image_from_met(yr)

        # sampleRegions wants a FeatureCollection, so wrap `f` in a one-element FC
        sampled_fc = year_img.sampleRegions(
            collection=ee.FeatureCollection([f]),
            scale=met_scale,
            geometries=True  # keep the geometry
        )
        # If sampling produced at least one feature, use it; otherwise keep original feature.
        sampled_or_original = ee.Algorithms.If(
            sampled_fc.size().gt(0),
            sampled_fc.first(),
            f.set('met_sample_missing', 1)  # or whatever flag you like
        )

        return ee.Feature(sampled_or_original)
    
    def reduce_met_mean(f):
        # Get year from feature (cast to Number to be safe)
        """
        Add meteorological data to a feature.
        
        Parameters:
        f (ee.Feature): The feature to which we add meteorological data.
        
        Returns:
        ee.Feature: The feature with added meteorological data.
        """
        yr = ee.Number(f.get('year'))

        # Build the year's multi-band image
        year_img = year_image_from_met(yr)

        # sampleRegions wants a FeatureCollection, so wrap `f` in a one-element FC
        sampled_fc = year_img.reduceRegions(
            collection=ee.FeatureCollection([f]),
            reducer=ee.Reducer.mean(),
            scale=met_scale
        )
        # If sampling produced at least one feature, use it; otherwise keep original feature.
        sampled_or_original = ee.Algorithms.If(
            sampled_fc.size().gt(0),
            sampled_fc.first(),
            f.set('met_sample_missing', 1)  # or whatever flag you like
        )

        return ee.Feature(sampled_or_original)
    
    def reduce_met_median(f):
        # Get year from feature (cast to Number to be safe)
        """
        Add meteorological data to a feature.
        
        Parameters:
        f (ee.Feature): The feature to which we add meteorological data.
        
        Returns:
        ee.Feature: The feature with added meteorological data.
        """
        yr = ee.Number(f.get('year'))

        # Build the year's multi-band image
        year_img = year_image_from_met(yr)

        stats = year_img.reduceRegion(
            geometry=f.geometry(),
            reducer=ee.Reducer.median(),
            scale=met_scale,
            bestEffort=True,
            tileScale=2
        )

        return f.set(stats)

    
    if attr_geom == "pt":
        spatial_aggr_fin_samp = spatial_aggr_fin.map(sample_met)  # type: ignore[attr-defined]
    else:
        if spatial_aggregation_opt == "sampling": 
            spatial_aggr_fin_samp = spatial_aggr_fin.map(sample_met)  # type: ignore[attr-defined]
        elif spatial_aggregation_opt == "mean":
           spatial_aggr_fin_samp = spatial_aggr_fin.map(reduce_met_mean)  # type: ignore[attr-defined]
        elif spatial_aggregation_opt == "median":
            spatial_aggr_fin_samp = spatial_aggr_fin.map(reduce_met_median)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsuported spatial aggregation opt: {spatial_aggregation_opt}")
    
    
    

    ## --- Other Dynamic Datasets checks and sampling --- ##
    gee_imcols = input_cfg.gee_imcols
    for dyn_data in gee_imcols:
        gee_imcol_id = dyn_data.gee_imcol_id
        gee_imcol_bands = dyn_data.gee_imcol_bands
        gee_imcol_bands_names = [band.band_name for band in gee_imcol_bands]
        gee_imcol_bands_make_static = [band.make_static for band in gee_imcol_bands]
        gee_imcol_bands_scale = [band.scale for band in gee_imcol_bands]
        # Check if gee imcol ids are valid
        try:
            gee_imcol = ee.ImageCollection(gee_imcol_id)
        except:
            raise ValueError(f"Unsuported gee imcol id: {gee_imcol_id}")
        # Check if gee imcol bands are valid
        first_band_names = gee_imcol.first().bandNames().getInfo()
        for gee_imcol_band in gee_imcol_bands_names:
            if gee_imcol_band not in first_band_names:
                raise ValueError(f"Unsuported gee imcol band: {gee_imcol_band}")
        # Check fi scale is valid: between 10 and 1000000
        for gee_imcol_scale in gee_imcol_bands_scale:
            if gee_imcol_scale < 10 or gee_imcol_scale > 1000000:
                raise ValueError(f"Unsuported gee imcol scale: {gee_imcol_scale}")
        # Check if gee imcol make static is valid
        for gee_imcol_make_static in gee_imcol_bands_make_static:
            if not isinstance(gee_imcol_make_static, bool):
                raise ValueError(f"Unsuported gee imcol make static: {gee_imcol_make_static}")


        gee_imcol_filter = gee_imcol.filterDate(gee_start_date,gee_end_date)
        nMonths=gee_end_date.difference(gee_start_date, 'month').toInt()
        monthIndexes = ee.List.sequence(0, nMonths.subtract(1))

        
        for band, scale,static in zip(list(gee_imcol_bands_names), gee_imcol_bands_scale, gee_imcol_bands_make_static):
            gee_imcol_sel=gee_imcol_filter.filterBounds(aoi).select(band)
            print(f"static status {static}")
            print(f"band {band}")
            
            if static:
                gee_imcol_sel=gee_imcol_sel.mean()
                # Sample from static gee data
                spatial_aggr_fin_samp=gee_imcol_sel.sampleRegions(collection=spatial_aggr_fin_samp, scale=scale, geometries=True)
                if input_cfg.save_gee_data:
                    geemap.ee_export_image(gee_imcol_sel, os.path.join(out_dir, str(band)+'static.tif'), scale=scale, region=aoi)
            else:
                def monthly_aggr_dyn(m):
                    m = ee.Number(m).toInt()
                    month_start = gee_start_date.advance(m, 'month')
                    month_end = month_start.advance(1, 'month')
                    sel=gee_imcol_sel.filterDate(month_start, month_end)
                    mean_img=sel.mean()
                    return mean_img.set('system:time_start', month_start.millis())
                # Convert to monthly means
                gee_monthly=ee.ImageCollection(monthIndexes.map(monthly_aggr_dyn))
                print(f"gee_monthly size {gee_monthly.size().getInfo()}")
                if input_cfg.save_gee_data:
                    geemap.ee_export_image_collection(gee_monthly, os.path.join(out_dir, 'other_gee'), scale=scale, region=aoi)
                # Sample from dynamic gee data
                def year_image_from_gee(yr):
                    """
                    Creates a multi-band image from the met data for a given year.

                    Parameters:
                        yr (ee.Number): The year of interest.

                    Returns:
                        ee.Image: A multi-band image containing the met data for the year.
                    """
                    yr = ee.Number(yr).toInt()
                    start = ee.Date.fromYMD(yr, 1, 1)
                    end   = start.advance(1, 'year')

                    year_ic = gee_monthly \
                        .filterDate(start, end) \
                        .sort('system:time_start')  # make sure months are in order

                    year_ic = year_ic.map(rename_by_month)

                    # Stack all images into one multi-band image
                    year_img = year_ic.toBands()

                    # Strip the "imageId_" prefix introduced by toBands
                    band_names = year_img.bandNames()

                    def strip_prefix(b):
                        b = ee.String(b)
                        parts = b.split('_')
                        return parts.slice(1).join('_')

                    clean_names = band_names.map(strip_prefix)

                    return year_img.rename(clean_names)
                
                def sample_gee(f):
                    # Get year from feature (cast to Number to be safe)
                    """
                    Add meteorological data to a feature.
                    
                    Parameters:
                    f (ee.Feature): The feature to which we add meteorological data.
                    
                    Returns:
                    ee.Feature: The feature with added meteorological data.
                    """
                    yr = ee.Number(f.get('year'))

                    # Build the year's multi-band image
                    year_img = year_image_from_gee(yr)

                    # sampleRegions wants a FeatureCollection, so wrap `f` in a one-element FC
                    sampled_fc = year_img.sampleRegions(
                        collection=ee.FeatureCollection([f]),
                        scale=scale,
                        geometries=True  # keep the geometry
                    )

                    # If sampling produced at least one feature, use it; otherwise keep original feature.
                    sampled_or_original = ee.Algorithms.If(
                        sampled_fc.size().gt(0),
                        sampled_fc.first(),
                        f.set('gee_sample_missing', 1)  # or whatever flag you like
                    )

                    return ee.Feature(sampled_or_original)
                spatial_aggr_fin_samp=spatial_aggr_fin_samp.map(sample_gee)
    # Export to shp
    # Remove geometries before download
    # spatial_aggr_fin_samp_no_geom = spatial_aggr_fin_samp.map(
    #     lambda f: f.setGeometry(None)
    # )

    # task = ee.batch.Export.table.toDrive(
    #     collection=spatial_aggr_fin_samp_no_geom,
    #     description='met_stats_export',
    #     fileFormat='CSV'
    # )
    # task.start()
    # # Then convert to GeoDataFrame (attributes only)
    # inp_fts = geemap.ee_to_gdf(spatial_aggr_fin_samp_no_geom)
    inp_fts=geemap.ee_to_gdf(spatial_aggr_fin_samp)
    return inp_fts
    # return gpd.GeoDataFrame()

