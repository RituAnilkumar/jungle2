import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

rates=pd.read_csv("C:\\Users\\jz25104\\OneDrive - University of Bristol\\Desktop\\liquidice_data\\hug\\zipped_files\\pergla\\time_series_05\\dh_05_rgi60_pergla_rates.csv\\dh_05_rgi60_pergla_rates.csv")
# print(rates.head())

# cumul=pd.read_csv("C:\\Users\\jz25104\\OneDrive - University of Bristol\\Desktop\\liquidice_data\\hug\\zipped_files\\pergla\\time_series_05\\dh_05_rgi60_pergla_cumul.csv\\dh_05_rgi60_pergla_cumul.csv")
# # print(cumul.head())
# # Convert cumul.time to pd.Datetime from str of format yyyy-mm-dd
# # Drop nulls in cumul
# cumul = cumul.dropna()
# cumul['time'] = pd.to_datetime(cumul['time'], format='%Y-%m-%d', errors="coerce")

# def comp(group):
#     grp_sorted = group.sort_values('time')
#     time_diffs_yrs = grp_sorted.time.diff().dt.days / 365.25
#     dh_diffs = grp_sorted.dh.diff()
#     inst_rates = dh_diffs / time_diffs_yrs
#     group['inst_rate'] = inst_rates
#     return group

# test = cumul.groupby('rgiid', group_keys=False).apply(comp)
# print(test.groupby('rgiid').inst_rate.mean())



print(rates[rates['period'] == '2000-01-01_2020-01-01'].dhdt)

print('xxxxxxxxxxxxxxxxxxxx')
print('xxxxxxxxxxxxxxxxxxxx')

# Create new columns in rates that splits period into start and end date
rates[['start_date', 'end_date']] = rates['period'].str.split('_', expand=True)
# Convert start_date and end_date to datetime
rates['start_date'] = pd.to_datetime(rates['start_date'], format='%Y-%m-%d', errors="coerce")
rates['end_date'] = pd.to_datetime(rates['end_date'], format='%Y-%m-%d', errors="coerce")
# print(rates.head())

# If end_date is exactly year ahead of start date, then select those rows
rates_sel=rates[(rates['end_date'] - rates['start_date']).dt.days < 367]
print(rates_sel.groupby('rgiid').dhdt.mean())

tmp_dict={'rgiid':[], 'dhdt_rollmean':[], 'start_date':[], 'end_date':[],'roll_win':[],'dhdt_rollmean_err':[]}
def calculate_error_sum(column,n):
        return np.sqrt((column**2).sum())/n

for g,grp in rates_sel.groupby('rgiid'):
    grp_sorted=grp.sort_values('start_date')
    # print(grp_sorted)
    for roll_size in range(19,21):
        rolling_mean=grp_sorted['dhdt'].rolling(window=roll_size).mean()
        time_st_roll=grp_sorted['start_date'].dt.year.rolling(window=roll_size).min()
        time_end_roll=grp_sorted['end_date'].dt.year.rolling(window=roll_size).max()
        rolling_err=(grp_sorted['err_dhdt'].pow(2).rolling(roll_size).sum().pipe(np.sqrt)/roll_size)

        mask = rolling_mean.notna()

        tmp_dict['rgiid'].append(g)
        tmp_dict['dhdt_rollmean'].append(rolling_mean[mask].tolist())
        tmp_dict['start_date'].append(time_st_roll[mask].tolist())
        tmp_dict['end_date'].append(time_end_roll[mask].tolist())
        tmp_dict['roll_win'].append(roll_size)
        tmp_dict['dhdt_rollmean_err'].append(rolling_err[mask].to_list())
        
df_rollmean=pd.DataFrame(tmp_dict)
# print(df_rollmean.head())
print('xxxxxxxxxxxxxxxxxxxx')
df_exploded=df_rollmean.explode(['dhdt_rollmean','start_date','end_date','dhdt_rollmean_err'])

df_exploded.to_csv(out_file)