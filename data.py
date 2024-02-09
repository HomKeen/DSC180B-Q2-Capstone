import pandas as pd
import numpy as np

#Grabber function to simplify dataset loading in notebook
def grab_dataset(dataset, timeframe='monthly'):
    
    #check for available datasets and timeframe
    if dataset not in ['global_temp', 'electricity', 'co2', 'ch4']:
        raise ValueError("not a valid dataset")
    if timeframe not in ['yearly', 'monthly']:
        raise ValueError("not a valid timeframe")
    
    #check for timeframe reference
    if timeframe == "monthly":
        
        #global temp data grabber
        if dataset == 'global_temp':
            months_abv = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_rep = {'Jan': 1,
                        'Feb': 2,
                        'Mar': 3,
                        'Apr': 4,
                        'May': 5,
                        'Jun': 6,
                        'Jul': 7,
                        'Aug': 8,
                        'Sep': 9,
                        'Oct': 10,
                        'Nov': 11,
                        'Dec': 12}
            global_temp_raw = pd.read_csv('data/global-temp-monthly.csv')
            global_temp = pd.melt(global_temp_raw, id_vars=['Year'], value_vars=months_abv) \
                            .rename(columns={'Year': 'year', 'variable': 'month', 'value': 'temp_change'})
            global_temp = (global_temp
                            .assign(month_code=global_temp['month'].apply(lambda m: month_rep[m]))
                            .sort_values(by=['year', 'month_code'])
                            .drop(columns=['month_code'])
                            .reset_index(drop=True)
                           )
            return global_temp
        
        #electricity generation data grabber
        if dataset == 'electricity':
            elec_raw = pd.read_csv('data/electricity-overview-monthly.csv')
            
            #grab electricity net generation across all sectors
            elec_raw = elec_raw[elec_raw['Description'] == 'Electricity Net Generation, Total']
            
            #all previous years are yearly averages
            elec_raw = elec_raw[elec_raw['YYYYMM'] >= 197301]
            elec_raw['YYYYMM'] = elec_raw['YYYYMM'].astype("string")
            elec_raw['month'] = elec_raw['YYYYMM'].str[-2:]
            elec_raw['year'] = elec_raw['YYYYMM'].str[:4]

            #ensure that electricity generation value is numeric
            # elec_raw['Value'] = elec_raw['Value'].astype('float')
            
            #month replacement for dataset uniformity
            month_rep = {'01': 'Jan',
                        '02': 'Feb',
                        '03': 'Mar',
                        '04': 'Apr',
                        '05': 'May',
                        '06': 'Jun',
                        '07': 'Jul',
                        '08': 'Aug',
                        '09': 'Sep',
                        '10': 'Oct',
                        '11': 'Nov',
                        '12': 'Dec',}
            # elec_raw['month'] = elec_raw['month'].replace(month_rep)

            elec = (elec_raw[['year', 'month', 'Value']]
                    .replace({'month': month_rep})
                    .rename(columns = {'Value': 'elec_generation'})
                    .reset_index(drop = True)
                    .astype({'elec_generation': 'float', 'year': 'int'})
                    )
            
            # Drop rows where month==13, which are the yearly totals.
            elec = elec[elec['month'] != '13']

            return elec
            
        #co2 data grabber    
        if dataset == 'co2':
            co2_raw = pd.read_csv('data/co2-monthly.csv', comment = '#')
            
            #month replacement for dataset uniformity
            month_rep = {1: 'Jan',
                        2: 'Feb',
                        3: 'Mar',
                        4: 'Apr',
                        5: 'May',
                        6: 'Jun',
                        7: 'Jul',
                        8: 'Aug',
                        9: 'Sep',
                        10: 'Oct',
                        11: 'Nov',
                        12: 'Dec',}
            co2_raw['month'] = co2_raw['month'].replace(month_rep)
            
            co2 = co2_raw[['year', 'month', 'average']]
            co2 = co2.rename(columns = {'average': 'co2_average'})
            return co2
        
        #methane data grabber
        if dataset == 'ch4':
            ch4_raw = pd.read_csv('data/ch4-monthly.csv', comment = '#')
            
            month_rep = {1: 'Jan',
                        2: 'Feb',
                        3: 'Mar',
                        4: 'Apr',
                        5: 'May',
                        6: 'Jun',
                        7: 'Jul',
                        8: 'Aug',
                        9: 'Sep',
                        10: 'Oct',
                        11: 'Nov',
                        12: 'Dec',}
            ch4_raw['month'] = ch4_raw['month'].replace(month_rep)
            
            ch4 = ch4_raw[['year', 'month', 'average']]
            ch4 = ch4.rename(columns = {'average': 'ch4_average'})
            return ch4
        
        
    #yearly data grab (UNFINISHED)
    else:
        return 1
        