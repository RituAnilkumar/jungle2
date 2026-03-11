# Downloading OGGM data into a folder and merging with RGI
1. For downloading OGGM data from the servers, cd into the folder you wnat and then do  wget -r -np -nH --cut-dirs=10 -A "*fixed_geometry_mass_balance_*.csv" https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.6/elev_bands/ERA5/per_glacier_spinup/RGI62/b_080/L5/summary/
2. Ensure the rgi files are also downlaoded. I have it at /scratch/b5at/ranil.b5at/met_raw/rgi
3. cd into and run /scratch/b5at/ranil.b5at/met_raw/oggm_summary/prepare_oggm.py

# Downloading W5E5 to the temporal step of your interest

# Downloading and aggregating ERA5 to the temporal step of your interest
1. generate CDS request code from the CDS website
2. Paste the code into the location you want and download it. I like to do this on local asystem as it seems to timeout on ISAMBARD
3. rsync it into your chosen directory and extract the grib file. 
4. Generate aggregates as required using /scratch/b5at/ranil.b5at/met_raw/era5aggregates/era5_dataprocess.ipynb and save them as nc files

# Extracting met data for csv of glaciers with lat lon and year columns
1. I assume that the static data from RGI that is required has been already merged. Its a simple command if not. df1.merge(df2,left_on='id',right_on'id2',how='left')
2. Run /scratch/b5at/ranil.b5at/met_raw/oggm_summary/prep_inp_target_df.py to extract. You will have to set the file path as desired. 