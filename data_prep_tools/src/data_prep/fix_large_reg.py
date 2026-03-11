## Run to combine all parts of alaska, central asia and south asia west as they have a large number of glaciers and gee times out.
import pandas as pd
base_dir='tmp_data/met/'
base_name='r13_cas' #Change here for the region num and shortname
files_to_merge=[base_name+'_pt1.csv',base_name+'_pt2.csv',base_name+'_pt3.csv',base_name+'_pt4.csv',base_name+'_pt5.csv',base_name+'_pt6.csv'] # change here for the number of parts there is
df_list=[pd.read_csv(base_dir+file) for file in files_to_merge]
merged_df=pd.concat(df_list,ignore_index=True)
merged_df.to_csv(base_dir+base_name+'.csv',index=False) 