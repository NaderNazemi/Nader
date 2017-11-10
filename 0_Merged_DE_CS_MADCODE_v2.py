# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:05:34 2017
@author: Nader
"""
#%%
#%%
#                                           THIS IS VERSION_3 WHICH IS THE CLEAN VERSION OF VERSION_2 OF THE FILE
#                                                  MERGED ON "MAD_CODE" - FIXED MISSING TIME SERIES ROWS !
#%%
#                               This is Merged on "SC" on the files "Austin_Data_extraction" and "Austin_cust_sat"
#%%
############################################################################### START     
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime 
from statsmodels.tsa.seasonal import seasonal_decompose
#%%
# Read in the Cust_Data Data Set : df1
def parse(x):
    return datetime.strptime(x, '%Y%m')
df1 = pd.read_csv('Austin_Data_extraction.csv', encoding='ISO-8859-1', parse_dates=[0], date_parser=parse) 
print(df1.shape) # (6781650, 20)
#print(df1.info)
#%%
# Renaming column name from 'ORIGIN_SIC' into "SIC' to match 'Austin_cust_sat.csv' 
df1.rename(columns={'ORIGIN_SIC':'SC'}, inplace=True)
print(df1.shape) # (6781650, 20)
print(df1.info)
#%%
#                                       FINDING the '-777' "CUST_NBR" - # There are 3853 "-777" elements 
a1 = df1.loc[df1['CUST_NBR'] == -777] 
print(df1.loc[df1['CUST_NBR'] == -777].count())
print(df1.loc[df1['CUST_NBR'] == -777]['CUST_NBR'].count()) # 3853
#%%
#                                                                EXCLUDING "-777" etc.. OUTLIER
#df1 = df1[df1.CUST_NBR != -777] # works
df1 = df1[df1['CUST_NBR'] != -777]
print(df1.shape) # (6777797, 20)
print(df1.loc[df1['CUST_NBR'] == -777].count()) # Number of "-777" after removing : 0
#df1[df1['CUST_NBR'] == -777]['CUST_NBR'].count() # same as above line 
print(df1.shape) # (6777797, 20)
#print(df1.info)
#%%
#                                                     YEAR_TO_DATE for df1 and in between 2016->2017
#df1a = df1[(df1['YEAR_MONTH'] > '2016-12-31')].copy()  # YEAR TO DATE - 2017 , looks like there a 2018 element here ORIGNAL: '2016-12-31'
df1a = df1[(df1['YEAR_MONTH'] > '2016-12-31') & (df1['YEAR_MONTH'] <= '2017-10-31')].copy() # Year to Date
#df1 = df1[(df1['YEAR_MONTH'] > '2015-12-31')]  # 2016 FORWARD
df1b = df1[(df1['YEAR_MONTH'] > '2015-12-31') & (df1['YEAR_MONTH'] < '2016-12-31')].copy() # Between 2016 & 2017 - ORIGINAL: '2015-12-31'
print(df1a.shape) # (1121501, 20)
print(df1b.shape) # (1347444, 20)
#print(df1a.info)
#print(df1b.info)
#%%
#                                                            Read in the Cust_Sat Data Set
def parse(x):
    return datetime.strptime(x, '%m/%d/%y') # Original 
df2 = pd.read_csv('Austin_cust_sat.csv', encoding='ISO-8859-1', parse_dates=[2], date_parser=parse) # ORIGINAL
print(df2.shape) 
#print(df2.info)
#%%
#                                                   YEAR_TO_DATE for df2 and in between 2016->2017
#df2 = df2[(df2['Survey Date'] > '2016-12-31')] # ORIGINAL 
#df2a = df2[(df2['Survey Date'] > '2015-12-31')].copy()
df2a = df2[(df2['Survey Date'] > '2016-12-31') & (df2['Survey Date'] <= '2017-10-31')].copy()  # YEAR TO DATA - 2017 - ORIGNAL '2016-12-31'
df2b = df2[(df2['Survey Date'] > '2015-12-31') & (df2['Survey Date'] <= '2016-12-31')].copy()  # Between 2016 & 2017 - ORIGINAL: '2015-12-31'
print(df2a.shape) # (4355, 26)
print(df2b.shape) # (9330, 26)
#%%
#                                                                   Drop Columns in df2
df2a.drop(['Survey Year','Survey Month','Overall Improvement'], axis=1, inplace=True)
df2b.drop(['Survey Year','Survey Month','Overall Improvement'], axis=1, inplace=True)
print(df2a.shape) # (4355, 23)
print(df2b.shape) # (9330, 23)
#%%
# Read in the Safety Data Set - using the first column as Date Column 
def parse(x):
    return datetime.strptime(x, '%Y%m')
df3 = pd.read_csv('Austin_Safety08292017.csv', parse_dates=[0], date_parser=parse) # ORIGINAL 
#%%
# Renaming column name from 'LOC_SIC_SLT' into "SIC' to match 'Austin_cust_sat.csv' 
df3.rename(columns={'LOC_SIC_SLT':'SC'}, inplace=True) 
#%%
#                                                                     YEAR_TO_DATE for df3
df3a = df3[(df3['YYYYMM_NUMBER'] > '2016-12-31') & (df3['YYYYMM_NUMBER'] <= '2017-10-31')].copy() # ORIGINAL 
#df3 = df3[(df3['YYYYMM_NUMBER'] > '2015-12-31')]
df3a.shape # (1292, 4)
#%%
print(df1a.shape)   # (1121502, 20) 
print(df1b.shape)   # (1347444, 20)
print(df2a.shape)   # (4355, 26)    
print(df2b.shape)   # (9330, 23)
print(df3a.shape)   # (1292, 4)    
#%%
# df1 Description prior to Grouping of Data-This is a TimeSeries with Many Shippers etc..Needs to be applied again after GROUPER based on 'M'
df1a_description = df1a.describe()
df1b_description = df1b.describe()
#%%
#                                                                   Number of Unique CUSTNBR
numberOfUnique_CUST_NBR_a = df1a['CUST_NBR'].nunique() 
numberOfUnique_CUST_NBR_b = df1b['CUST_NBR'].nunique() 

print(numberOfUnique_CUST_NBR_a) # (357400)
print(numberOfUnique_CUST_NBR_b) # (394462)
#%%
#                                               Number of Unique NAICS6_DESC: 1044 industries / 1001 Y-to-D
numberOfUnique_INDUSTRY_a = df1a['NAICS6_DESC'].nunique() 
numberOfUnique_INDUSTRY_b = df1b['NAICS6_DESC'].nunique() 

print(numberOfUnique_INDUSTRY_a) # 1001
print(numberOfUnique_INDUSTRY_b) # 1008
#%%
#                                                     Number of Unique NAICS6_CD: 1044 industry CODES
numberOfUnique_INDUSTRY_CODE_a = df1a['NAICS6_CD'].nunique()
numberOfUnique_INDUSTRY_CODE_b = df1b['NAICS6_CD'].nunique()
 
print(numberOfUnique_INDUSTRY_CODE_a)
print(numberOfUnique_INDUSTRY_CODE_b)
#%%
#                                                                   Number of Unique SC: 534 
numberOfUnique_SC_a = df1a['SC'].nunique() 
numberOfUnique_SC_b = df1b['SC'].nunique() 

print(numberOfUnique_SC_a) # 534 for Y-to-D
print(numberOfUnique_SC_b) # 534 for 2016 - 2017   
#%%
#                                                                   Number of Unique CUSTNBR
numberOfUnique_MAD_CODE_a = df1a['MAD_CODE'].nunique() 
numberOfUnique_MAD_CODE_b = df1b['MAD_CODE'].nunique() 

print(numberOfUnique_MAD_CODE_a) # (169714)
print(numberOfUnique_MAD_CODE_b) # (183961)  
#%%
#                                                                   Number of Unique CUSTNBR
numberOfUnique_SHIPPER_a = df1a['SHIPPER'].nunique()  
numberOfUnique_SHIPPER_b = df1b['SHIPPER'].nunique()

print(numberOfUnique_SHIPPER_a) # 257559
print(numberOfUnique_SHIPPER_b) # 280571
#%%
frequencyOf_SC_a = df1a['SC'].value_counts()  # USB: 16830 / XML: 15692
frequencyOf_SC_b = df1b['SC'].value_counts()  # USB  20270 / XML: 18471

print(frequencyOf_SC_a.head(5))
print('\n')
print(frequencyOf_SC_b.head(5))
#%%
Total_sum_of_SC_frequency_a = df1a['SC'].value_counts().sum()
Total_sum_of_SC_frequency_b = df1b['SC'].value_counts().sum()

print(Total_sum_of_SC_frequency_a) # 1121501
print(Total_sum_of_SC_frequency_b) # 1347444
#%%
frequencyOf_SHIPPER_a = df1a['SHIPPER'].value_counts()  # WAL-MART STORES INC: 7389  | XPO LOGISTICS: 2845
frequencyOf_SHIPPER_b = df1b['SHIPPER'].value_counts()  # WAL-MART STORES INC: 13882 | XPO LOGISTICS: 3709

print(frequencyOf_SHIPPER_a.head(5)) 
print(frequencyOf_SHIPPER_b.head(5)) 
#%%
Total_sum_of_Shipper_frequency_a = df1a['SHIPPER'].value_counts().sum()  
Total_sum_of_Shipper_frequency_b = df1b['SHIPPER'].value_counts().sum()  

print(Total_sum_of_Shipper_frequency_a) # 1121499
print(Total_sum_of_Shipper_frequency_b) # 1347441
#%%
frequencyOf_CUST_NBR_a = df1a['CUST_NBR'].value_counts()  # 631569026    21
frequencyOf_CUST_NBR_b = df1b['CUST_NBR'].value_counts()  # 229830006    39

print(frequencyOf_CUST_NBR_a.head(5)) 
print('\n')
print(frequencyOf_CUST_NBR_b.head(5)) 

apple = df1a[df1a['CUST_NBR'] == 631569026] # who is CUST_NBR: "631569026" ? "NIKE NALC"
apple['REVENUE'].sum() 
#%%
print(df1a['CUST_NBR'].value_counts().sum()) # 1121501 / Total Number of Frequency Unique 
print(df1b['CUST_NBR'].value_counts().sum()) # 1347444
#%%
df1.isnull().sum() # Missing: MAD_CODE: 1806203
print('\n')
df2.isnull().sum() # Missing: MAD CD:   3
#%%
################################################################################ START
#                                          Plot of top 10 based on Frequency - This is prior to Grouping etc.. 
frequencyOf_CUST_NBR_a.head(10).plot(kind='bar')
#%%
frequencyOf_SHIPPER_a.head(10).plot(kind='bar')
#%%
frequencyOf_SC_a.head(10).plot(kind='bar')
#%%
frequencyOf_CUST_NBR_a.head(10).plot(kind='pie')
#%%
frequencyOf_SHIPPER_a.head(10).plot(kind='pie')
#%%
frequencyOf_SC_a.head(10).plot(kind='pie')
#%%
frequencyOf_CUST_NBR_b.head(10).plot(kind='bar')
#%%
frequencyOf_SHIPPER_b.head(10).plot(kind='bar')
#%%
frequencyOf_SC_b.head(10).plot(kind='bar')
#%%
frequencyOf_CUST_NBR_b.head(10).plot(kind='pie')
#%%
frequencyOf_SHIPPER_b.head(10).plot(kind='pie')
#%%
frequencyOf_SC_b.head(10).plot(kind='pie')
#%%
############################################################################### END

#%%
#                                                       Checking for missing values in columns 
print(df1a.isnull().sum()) # There are missing MAD_CODE: 343351 and COST: 90939 
print('\n')
print(df1b.isnull().sum()) #                   MAD_CODE: 388283 / COST: 182
#%%
#                                                       Dropping missing MAD_CODE in "df1a" and "df1b" - NEW 
df1a.dropna(subset=['MAD_CODE'], inplace=True) 
df1b.dropna(subset=['MAD_CODE'], inplace=True) 
print(df1a.isnull().sum())
print('\n')
print(df1b.isnull().sum())
#%%
#                                                       Dropping missing MAD_CODE in "df2a" and "df2b" - NEW
df2a.dropna(subset=['CUST MAD CD'], inplace=True) 
df2b.dropna(subset=['CUST MAD CD'], inplace=True) 
print(df2a.isnull().sum())
print('\n')
print(df2b.isnull().sum())
#%%
#                                                                   Dropped Missing: "SC" 
print(df2a.shape) # (4355, 23)
print(df2a.isnull().sum())
df2a.dropna(subset=['SC'], inplace=True) 
print(df2a.isnull().sum())
print(df2a.shape) # (4353, 23)
#%%
#                                                                   Dropped Missing: "SC" 
print(df2b.shape) # (9330, 23)
print(df2b.isnull().sum())
df2b.dropna(subset=['SC'], inplace=True) 
print(df2b.isnull().sum())
print(df2b.shape) # (9330, 23)
#%%
#                                                                       Renaming Columns 
df1a.rename(columns={'CUST_NBR':'CUSTNBR','NAICS6_CD':'IndustCD', 'REVENUE':'REV','ON TIME SHPMT %':'ON_T_SHP', 'SHPMT_CNT':'SHPCNT',
                     'TONNAGE':'TON', 'CLAIM FILED':'CF', 'CLAIM PAID':'CP', 'DAMAGE CLAIM FILED':'DCF', 'DAMAGE CLAIM PAID':'DCP' }, inplace=True)
    
df1b.rename(columns={'CUST_NBR':'CUSTNBR','NAICS6_CD':'IndustCD', 'REVENUE':'REV','ON TIME SHPMT %':'ON_T_SHP', 'SHPMT_CNT':'SHPCNT',
                     'TONNAGE':'TON', 'CLAIM FILED':'CF', 'CLAIM PAID':'CP', 'DAMAGE CLAIM FILED':'DCF', 'DAMAGE CLAIM PAID':'DCP' }, inplace=True)
#%%
#                                                                       Renaming Columns 
df2a.rename(columns={'CUST MAD CD':'MAD_CODE'}, inplace=True)
df2b.rename(columns={'CUST MAD CD':'MAD_CODE'}, inplace=True)
print(df2a.isnull().sum())
print('\n')
print(df2b.isnull().sum())
#%%
#                                                           Find Unique elements GROUPED_BY "CUST_NBR"
#a1 = df1a.groupby('CUSTNBR').nunique()
#a2 = df1a.groupby('CUSTNBR').SC.nunique() 
#a3 = df1a.groupby('CUSTNBR').MAD_CODE.nunique() 
#a4 = df1a.groupby('CUSTNBR').SHIPPER.nunique() 
#a5 = df1a.groupby('SHIPPER').nunique() 
#a6 = df1a['SHIPPER'].nunique()
#%%
#a11 = df1b.groupby('CUSTNBR').nunique()
#a22 = df1b.groupby('CUSTNBR').SC.nunique() 
#a33 = df1b.groupby('CUSTNBR').MAD_CODE.nunique() 
#a44 = df1b.groupby('CUSTNBR').SHIPPER.nunique() 
#a55 = df1b.groupby('SHIPPER').nunique() 
#a66 = df1b['SHIPPER'].nunique()
#%%
#                   Check for missing values and find manually the mean or median or frequency prior to applying imputer 
print(df1a.shape) # (1121501, 20)
print(df1a.isnull().sum())
print(df1a['COST'].mean()) # 2280.4190523140937
#%%
#                                                              Replacing missing COST values with "mean" 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) # can apply "median" but doing so does not take trending into consideration 
df1a['COST'] = imputer.fit_transform(df1a[['COST']])
print(df1a.isnull().sum()) # missing: COST: 0  
print(df1a.shape) # (1121501, 20)
#%%
#                                                              Replacing missing COST values with "mean"
print(df1b.shape) # (1347444, 20)
print(df1b.isnull().sum())
print(df1b['COST'].mean()) # 2230.7551550106487
#%%
#                                                               Replacing missing COST values with "mean"
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) # can apply "median" but doing so does not take trending into consideration 
df1b['COST'] = imputer.fit_transform(df1b[['COST']])
print(df1b.isnull().sum()) # missing: COST: 0  
print(df1b.shape) # (1347444, 20)
#%%
#                                                                   Creating Profit Column 
df1a['PROFIT'] = df1a['REV'] - df1a['COST']
df1a.isnull().sum()
#%%
#                                                                   Creating Profit Column 
df1b['PROFIT'] = df1b['REV'] - df1b['COST']
df1b.isnull().sum()
#%%
#                                                                   Creating DCF/REV Column 
#df1.drop(['DCF/REV'], axis=1, inplace=True) 
df1a['DCF_REV'] = df1a['DCF'] / df1a['REV']
df1a.isnull().sum()
#%%
#                                                                   Creating DCF/REV Column 
#df1.drop(['DCF/REV'], axis=1, inplace=True) 
df1b['DCF_REV'] = df1b['DCF'] / df1b['REV']
df1b.isnull().sum()
#%%
print(df1a.shape) # (778150, 22)
print(df1b.shape) # (959161, 22)
#%%
#                               First Replace "inf" and "-inf" with "na" THEN Replace "na" values with 0 or drop them 
df1a.replace([np.inf, -np.inf], 0, inplace=True) # Replace "inf" and "-inf" with 0 or np.nan
print(df1a.isnull().sum())
print(df1a.shape) # (1121501, 22) -> (778150, 22)
#%%
df1b.replace([np.inf, -np.inf], 0, inplace=True) # Replace "inf" and "-inf" with 0 or np.nan
print(df1b.isnull().sum())
print(df1b.shape) # (959161, 22)
#%%
#                                                               Replaced Blank Values by "0"
df1a['DCF_REV'].fillna(0, inplace=True)
print(df1a.isnull().sum())
print(df1a.shape) # (1121501, 22) -> (778150, 22)
#%%
#                                                               Replaced Blank Values by "0"
df1b['DCF_REV'].fillna(0, inplace=True)
print(df1b.isnull().sum())
print(df1b.shape) # (1347444, 22) -> (959161, 22)
#%%
#                                                       Creating On_Time_SHPMNT_TONNAGE Column NEW

df1a['ON_T_SHP_PERCENT_TON'] = (df1a['ON_T_SHP'] * df1a['TON']) / df1a['TON']
df1b['ON_T_SHP_PERCENT_TON'] = (df1b['ON_T_SHP'] * df1b['TON']) / df1b['TON']
#%%
print(df1a.shape) # (778150, 23)
print(df1b.shape) # (959161, 23)
#%%
print(df1a.isnull().sum())
print('\n')
print(df1b.isnull().sum())
#%%
#                                                           Changing the order of columns names in df1a
df1a = df1a[['YEAR_MONTH', 'SHIPPER', 'SC', 'MAD_CODE', 'CUSTNBR', 'ADDRESS', 'CITY',
       'STATE', 'ZIP6', 'IndustCD', 'NAICS6_DESC', 'REV', 'COST','PROFIT' ,'ON_T_SHP',
       'SHPCNT', 'TON', 'CF', 'CP', 'DCF', 'DCP', 'DCF_REV','ON_T_SHP_PERCENT_TON']]
#%%
#                                                           Changing the order of columns names in df1b
df1b = df1b[['YEAR_MONTH', 'SHIPPER', 'SC', 'MAD_CODE', 'CUSTNBR', 'ADDRESS', 'CITY',
       'STATE', 'ZIP6', 'IndustCD', 'NAICS6_DESC', 'REV', 'COST','PROFIT' ,'ON_T_SHP',
       'SHPCNT', 'TON', 'CF', 'CP', 'DCF', 'DCP', 'DCF_REV','ON_T_SHP_PERCENT_TON']]
#%%
#                                                    Dropping "na" values in Column "SHIPPER
df1a.dropna(subset=['SHIPPER'], inplace=True) 
print(df1a.isnull().sum())
print(df1a.shape) # (1,121,499, 23)
#%%
#                                                    Dropping "na" values in Column "SHIPPER
df1b.dropna(subset=['SHIPPER'], inplace=True) 
print(df1b.isnull().sum())
print(df1b.shape) # (1,347,441, 23) 
#%%
############################################################################### START - Optional 
############################################################################### START - Optional
#                                       This is to replace Numeric Values for elements in "Region" - OPTOINAL - START
df2 = df2[df2['Region'] != 'UNKNOWN']
df2 = df2[df2['Region'] != '(Not Answered)']
df2['Region'].value_counts()
df2.dropna(subset=['Region'], inplace=True)
print(df2['Region'].isnull().sum())
print(df2['Region'].unique())
#%%
print(df2['Region'].value_counts())
#hold = df2.Region.unique()
hold = df2['Region'].unique()
unique_Region = [i for i in hold]
list1 = [i for i in hold]
list2 = list(range(len(unique_Region)))
print(len(list1))
print(len(list2))
dict1 = dict(zip(list1, list2))
df2['Region'] = df2['Region'].map(dict1)
#%%
print(df2['Region'].value_counts())
print(df2['Region'].unique())
#%%
#                                                               Optional: Dropping "Region"
df2a.dropna(subset=['Region'], inplace=True)
df2b.dropna(subset=['Region'], inplace=True)
#%%
print(df2.isnull().sum())
print(df2.isnull().sum())
#%%
#                                       This is to replace Numeric Values for elements in "Region" - OPTOINAL - END
############################################################################### END - Optional 
############################################################################### END - Optional 
#%%
#                                                                       Renaming Columns 
df2a.rename(columns={'Customer Satisfaction Index':'CSI', 'SC':'SC', 'Theme: Billing':'T_Bill', 'Theme: Damages':'T-Damage',
                     'Theme: Delivery':'T-Del.', 'Theme: Pricing':'T-Price', 'Count':'Count'}, inplace=True)
    
df2b.rename(columns={'Customer Satisfaction Index':'CSI', 'SC':'SC', 'Theme: Billing':'T_Bill', 'Theme: Damages':'T-Damage',
                     'Theme: Delivery':'T-Del.', 'Theme: Pricing':'T-Price', 'Count':'Count'}, inplace=True)
#%%
#%%
#%%
#%%
#                                             Creating Subsets - Method - 1 - THIS HAS TIME INDEX
    
df1_Grouped_by_SC_1a = df1a.groupby(['MAD_CODE', pd.Grouper(key='YEAR_MONTH', freq='M')])['REV','COST','PROFIT','SHPCNT','TON','CF','CP','DCF','DCP'].sum()
df1_Grouped_by_SC_2a = df1a.groupby(['MAD_CODE', pd.Grouper(key='YEAR_MONTH', freq='M')])['ON_T_SHP','DCF_REV', 'ON_T_SHP_PERCENT_TON'].mean()
df1_Grouped_by_SC_3a = pd.concat([df1_Grouped_by_SC_1a, df1_Grouped_by_SC_2a], axis=1)
print(df1_Grouped_by_SC_3a.shape) # (777224, 12)
#%%
#                                             Creating Subsets - Method - 1 - THIS HAS TIME INDEX

df1_Grouped_by_SC_1b = df1b.groupby(['MAD_CODE', pd.Grouper(key='YEAR_MONTH', freq='M')])['REV','COST','PROFIT','SHPCNT','TON','CF','CP','DCF','DCP'].sum()
df1_Grouped_by_SC_2b = df1b.groupby(['MAD_CODE', pd.Grouper(key='YEAR_MONTH', freq='M')])['ON_T_SHP','DCF_REV','ON_T_SHP_PERCENT_TON'].mean()
df1_Grouped_by_SC_3b = pd.concat([df1_Grouped_by_SC_1b, df1_Grouped_by_SC_2b], axis=1)
print(df1_Grouped_by_SC_3b.shape) # (951262, 12)
#%%
print(df1_Grouped_by_SC_3a.shape) # (777224, 12)
print(df1_Grouped_by_SC_3b.shape) # (951262, 12)
print('\n')
#print(df1_Grouped_by_SC_3a.info()) # Year to Date
#print(df1_Grouped_by_SC_3b.info()) # All of 2016
#%%
#%%
############################################################################### START - 1 - df1_Grouped_by_SC_3a
############################################################################### START - 1 - df1_Grouped_by_SC_3a
# This is a NEW CODE BLOCK to populate missing rows of Data in "df1_Grouped_by_SC_3a"

A_df = pd.DataFrame() # Create Empty DataFrame
A_mycopy1 = df1_Grouped_by_SC_3a.copy() # ()
print(A_mycopy1.shape)                                                          # (777224, 12)
A_mycopy1 = A_mycopy1.head(500) # This is for Testing Purposes
A_mycopy1 = A_mycopy1.reset_index()
print(A_mycopy1.shape)
#%%
#                                                               This code needs to be optimized 
print(A_mycopy1.shape)
A_mycopy1 = A_mycopy1.set_index('YEAR_MONTH')
idx = pd.date_range('2017-01-01 00:00:00', '2017-10-31 00:00:00', freq='M') # Year to Date

print(A_mycopy1.shape)

for i in set(A_mycopy1['MAD_CODE']):
    A_mycopy2 = A_mycopy1.loc[A_mycopy1['MAD_CODE'] == i] 
    A_mycopy2 = A_mycopy2.reindex(idx, fill_value=0)
    A_mycopy2['MAD_CODE'].replace(0, i, inplace=True) 
    A_df = A_df.append(A_mycopy2)

print(A_df.shape)
df1_Grouped_by_SC_3a = A_df
print(df1_Grouped_by_SC_3a.shape)
############################################################################### END - 1 - df1_Grouped_by_SC_3a
############################################################################### END - 1 - df1_Grouped_by_SC_3a
#%%
#%%
############################################################################### START - 1 - df1_Grouped_by_SC_3b
############################################################################### START - 1 - df1_Grouped_by_SC_3b
#                     This is a NEW CODE BLOCK to populate missing rows of Data in "df1a" and "df1b" and "df2a" and "df2b"

A_df_b = pd.DataFrame() # Create Empty DataFrame
A_mycopy1 = df1_Grouped_by_SC_3b.copy() # ()
A_mycopy1 = A_mycopy1.head(500) # This is for Testing Purposes
A_mycopy1 = A_mycopy1.reset_index()
#%%
#                                                               This code needs to be optimized 
print(A_mycopy1.shape)
A_mycopy1 = A_mycopy1.set_index('YEAR_MONTH')
idx = pd.date_range('2016-01-01 00:00:00', '2016-12-31 00:00:00', freq='M') # All of 2016 

print(A_mycopy1.shape)

for i in set(A_mycopy1['MAD_CODE']):
    A_mycopy2 = A_mycopy1.loc[A_mycopy1['MAD_CODE'] == i] 
    A_mycopy2 = A_mycopy2.reindex(idx, fill_value=0)
    A_mycopy2['MAD_CODE'].replace(0, i, inplace=True) 
    A_df_b = A_df_b.append(A_mycopy2)

print(A_df_b.shape)
df1_Grouped_by_SC_3b = A_df_b
print(df1_Grouped_by_SC_3b.shape)
############################################################################### END - 1 - df1_Grouped_by_SC_3b
############################################################################### END - 1 - df1_Grouped_by_SC_3b
#%%
#%%
#                                             Creating Subsets - Method - 1 - THIS HAS TIME INDEX Continued ...

df2_Grouped_by_SC_1a = df2a.groupby(['MAD_CODE', pd.Grouper(key='Survey Date', freq='M')])['T_Bill','T-Damage','T-Del.', 'T-Price','Count'].sum()
df2_Grouped_by_SC_2a = df2a.groupby(['MAD_CODE', pd.Grouper(key='Survey Date', freq='M')])['CSI'].mean()
df2_Grouped_by_SC_3a = pd.concat([df2_Grouped_by_SC_1a, df2_Grouped_by_SC_2a], axis=1)
#%%
#                                             Creating Subsets - Method - 1 - THIS HAS TIME INDEX Continued ...

df2_Grouped_by_SC_1b = df2b.groupby(['MAD_CODE', pd.Grouper(key='Survey Date', freq='M')])['T_Bill','T-Damage','T-Del.', 'T-Price','Count'].sum()
df2_Grouped_by_SC_2b = df2b.groupby(['MAD_CODE', pd.Grouper(key='Survey Date', freq='M')])['CSI'].mean()
df2_Grouped_by_SC_3b = pd.concat([df2_Grouped_by_SC_1b, df2_Grouped_by_SC_2b], axis=1)
#%%
print(df2_Grouped_by_SC_3a.shape) # (4291, 6)
print(df2_Grouped_by_SC_3b.shape) # (9142, 6)
print('\n')
#print(df2_Grouped_by_SC_3a.info()) 
#print(df2_Grouped_by_SC_3b.info()) 
#%%
#%%
############################################################################### START - 1 - df2_Grouped_by_SC_3a
############################################################################### START - 1 - df2_Grouped_by_SC_3a
#                     This is a NEW CODE BLOCK to populate missing rows of Data in "df1a" and "df1b" and "df2a" and "df2b"

A_df_2 = pd.DataFrame() # Create Empty DataFrame
A_mycopy1 = df2_Grouped_by_SC_3a.copy() # Year to Date
A_mycopy1 = A_mycopy1.head(500) # This is for Testing Purposes
A_mycopy1 = A_mycopy1.reset_index()
#%%
#                                                               This code needs to be optimized 
print(A_mycopy1.shape)
A_mycopy1 = A_mycopy1.set_index('Survey Date')
idx = pd.date_range('2017-01-01 00:00:00', '2017-10-31 00:00:00', freq='M') # Year to Date

print(A_mycopy1.shape)

for i in set(A_mycopy1['MAD_CODE']):
    A_mycopy2 = A_mycopy1.loc[A_mycopy1['MAD_CODE'] == i] 
    A_mycopy2 = A_mycopy2.reindex(idx, fill_value=0)
    A_mycopy2['MAD_CODE'].replace(0, i, inplace=True) 
    A_df_2 = A_df_2.append(A_mycopy2)

print(A_df_2.shape)
df2_Grouped_by_SC_3a = A_df_2
print(df2_Grouped_by_SC_3a.shape)
############################################################################### END - 1 - df2_Grouped_by_SC_3a
############################################################################### END - 1 - df2_Grouped_by_SC_3a
#%%
#%%
############################################################################### START - 1 - df2_Grouped_by_SC_3b
############################################################################### START - 1 - df2_Grouped_by_SC_3b
#                     This is a NEW CODE BLOCK to populate missing rows of Data in "df1a" and "df1b" and "df2a" and "df2b"

A_df_2_B = pd.DataFrame() # Create Empty DataFrame
A_mycopy1 = df2_Grouped_by_SC_3b.copy() # ()
A_mycopy1 = A_mycopy1.head(500) # This is for Testing Purposes
A_mycopy1 = A_mycopy1.reset_index()
#%%
#                                                               This code needs to be optimized 
print(A_mycopy1.shape)
A_mycopy1 = A_mycopy1.set_index('Survey Date')
idx = pd.date_range('2016-01-01 00:00:00', '2016-12-31 00:00:00', freq='M') # All of 2016 

print(A_mycopy1.shape)

for i in set(A_mycopy1['MAD_CODE']):
    A_mycopy2 = A_mycopy1.loc[A_mycopy1['MAD_CODE'] == i] 
    A_mycopy2 = A_mycopy2.reindex(idx, fill_value=0)
    A_mycopy2['MAD_CODE'].replace(0, i, inplace=True) 
    A_df_2_B = A_df_2_B.append(A_mycopy2)

print(A_df_2_B.shape)
df2_Grouped_by_SC_3b = A_df_2_B
print(df2_Grouped_by_SC_3b.shape)
############################################################################### END - 1 - df2_Grouped_by_SC_3b
############################################################################### END - 1 - df2_Grouped_by_SC_3b
#%%
#%%
#                               AGGREGATED RESULTS BASED ON CUSTNBR - VERSION - 2 - THIS LACKS TIME INDEX

test1 = df1_Grouped_by_SC_3a.iloc[:, :-3] # Except the Last three Columns "ON_T_SHP" and "DEC_REV" and "ON_T_SHP_PERCENT_TON" 
test2 = test1.groupby('MAD_CODE').sum()
test3 = df1_Grouped_by_SC_3a.iloc[:, [0,-1,-2, -3]]
test4 = test3.groupby('MAD_CODE').mean() 
df1_Grouped_by_SC_3_agg_a = pd.concat([test2, test4], axis=1)
#%%
print(df1_Grouped_by_SC_3_agg_a.shape) # (169714, 12)
#print(df1_Grouped_by_SC_3_agg_a.info())
#%%
test1 = df1_Grouped_by_SC_3b.iloc[:, :-3] # Except the Last three Columns "ON_T_SHP" and "DEC_REV", "ON_T_SHP_PERCENT_TON"
test2 = test1.groupby('MAD_CODE').sum() 
test3 = df1_Grouped_by_SC_3b.iloc[:, [0,-1,-2,-3]]
test4 = test3.groupby('MAD_CODE').mean()  
df1_Grouped_by_SC_3_agg_b = pd.concat([test2, test4], axis=1)
#%%
print(df1_Grouped_by_SC_3_agg_b.shape) # (183961, 12)
print(df1_Grouped_by_SC_3_agg_b.info())
#%%
print(df1_Grouped_by_SC_3_agg_a.shape) # (169714, 12)
print(df1_Grouped_by_SC_3_agg_b.shape) # (183961, 12)
#%%
df2_Grouped_by_SC_3a.columns
print('\n')
df2_Grouped_by_SC_3b.columns
#%%
#                                       Dropping Columns: 'T_Bill', 'T-Damage', 'T-Del.', 'T-Price', 'Count'
df2_Grouped_by_SC_3a.drop(['T_Bill', 'T-Damage', 'T-Del.', 'T-Price', 'Count'], axis=1, inplace=True) 
df2_Grouped_by_SC_3a.columns
#%%
#                                       Dropping Columns: 'T_Bill', 'T-Damage', 'T-Del.', 'T-Price', 'Count'
df2_Grouped_by_SC_3b.drop(['T_Bill', 'T-Damage', 'T-Del.', 'T-Price', 'Count'], axis=1, inplace=True) 
df2_Grouped_by_SC_3b.columns
#%%
#                                                                         AGGREGATED Continued ....

df2_Grouped_by_SC_3_agg_a = df2_Grouped_by_SC_3a.groupby('MAD_CODE').mean() 
print(df2_Grouped_by_SC_3_agg_a.shape) # (265, 1)
print(df2_Grouped_by_SC_3_agg_a.info())
#%%
#                                                                         AGGREGATED Continued ....

df2_Grouped_by_SC_3_agg_b = df2_Grouped_by_SC_3b.groupby('MAD_CODE').mean() 
print(df2_Grouped_by_SC_3_agg_b.shape) # (277, 1)
print(df2_Grouped_by_SC_3_agg_b.info())
#%%
print(df2_Grouped_by_SC_3_agg_a.shape) # (3147, 1)
print(df2_Grouped_by_SC_3_agg_b.shape) # (6224, 1)
#%%
#                                                   Various Slices of "df1_Grouped_by_CUST_NBR_3_agg"

df1_Grouped_by_SC_3_agg_description_a = df1_Grouped_by_SC_3_agg_a.describe()
df1_Grouped_by_SC_3_agg_description_b = df1_Grouped_by_SC_3_agg_b.describe()
#%%
#                                            SORT - Extracting the TOP 10 SC based on REV 
df1_Grouped_by_SC_3_agg_sorted_a = df1_Grouped_by_SC_3_agg_a.sort_values(['REV'], ascending=False) 
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a = df1_Grouped_by_SC_3_agg_sorted_a.head(10)

df1_Grouped_by_SC_3_agg_sorted_b = df1_Grouped_by_SC_3_agg_b.sort_values(['REV'], ascending=False) 
df1_Grouped_by_SC_3_agg_sorted_on_TOP_b = df1_Grouped_by_SC_3_agg_sorted_b.head(10)
#%%
#%%
#%%
#                                            SORT - Extracting the TOP 10 SC based on "CSI" 
df2_Grouped_by_SC_3_agg_sorted_a = df2_Grouped_by_SC_3_agg_a.sort_values(['CSI'], ascending=False) 
df2_Grouped_by_SC_3_agg_sorted_on_TOP_a = df2_Grouped_by_SC_3_agg_sorted_a.head(10)

df2_Grouped_by_SC_3_agg_sorted_b = df2_Grouped_by_SC_3_agg_b.sort_values(['CSI'], ascending=False) 
df2_Grouped_by_SC_3_agg_sorted_on_TOP_b = df2_Grouped_by_SC_3_agg_sorted_b.head(10)
#%%
#%%
#                               This Converts the INDEX column which contains the CUSTNBRs into a NORMAL column 

df1_Grouped_by_SC_3_agg_sorted_Added_index_a = df1_Grouped_by_SC_3_agg_sorted_a.reset_index()
df2_Grouped_by_SC_3_agg_sorted_Added_index_b = df2_Grouped_by_SC_3_agg_sorted_b.reset_index()
#%%
print(df1_Grouped_by_SC_3_agg_sorted_Added_index_a.shape) # (169714, 13)
print(df2_Grouped_by_SC_3_agg_sorted_Added_index_b.shape) # (6224, 2)
#%%
############################################################################### START
############################################################################### START

#                                Optional: Applying Visualization on NEW VIEW BASED on "SC" on TOP TEN 
plt.style.use('ggplot')
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.plot(kind='barh', width=1) 
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='barh') 
plt.gca().invert_yaxis()
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='box') 
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='kde') 
df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='barh', stacked=True) 
#%%
#                                                             More Visuals 
df1_Grouped_by_SC_3_agg_sorted_a.head(20).plot(kind='barh', width=1) 
#%%
df1_Grouped_by_SC_3_agg_sorted_a.head(20).ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='barh') 
#%%
plt.gca().invert_yaxis()
#%%
df1_Grouped_by_SC_3_agg_sorted_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='box') 
#%%
df1_Grouped_by_SC_3_agg_sorted_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='box') 
#%%
import seaborn as sns
hist = sns.distplot(df1_Grouped_by_SC_3_agg_sorted_a['REV'])
#%%
hist = sns.distplot(df1_Grouped_by_SC_3_agg_sorted_a['COST'])
#%%
hist = sns.distplot(df1_Grouped_by_SC_3_agg_sorted_a['PROFIT'])
#%%
df1_Grouped_by_SC_3_agg_sorted_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='kde') 
#%%
df1_Grouped_by_SC_3_agg_sorted_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='barh', stacked=True) 
#%%
#                                                   Box and Whisker Plot for SPECIFIC "USB" Time Series 
df1_Grouped_by_SC_3_USB_a.plot(kind='box')  
df1_Grouped_by_SC_3_USB_a[['REV','COST','PROFIT']].plot(kind='box') 
#%%
#                                                           Scatter Plot on Top 10 "AGG" Data 
plt.scatter(df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['REV']]/1000, df1_Grouped_by_SC_3_agg_sorted_on_TOP_a.ix[:, ['PROFIT']]/1000, color='b') 
plt.xlabel('REV')
plt.ylabel('PROFIT')
#%%
#                                                               Total AGG REVENUE Year_to_Date
Total_Agg_Rev = df1_Grouped_by_SC_3_agg_sorted_a['REV'].sum()
print(Total_Agg_Rev)  # 2964354003.4500012
Top_50_Agg_Rev =df1_Grouped_by_SC_3_agg_sorted_a.head(60)['REV'].sum()
print(Top_50_Agg_Rev) # 1355074679.4299996
print('TOP 50 SC\'s provide:  ', (Top_50_Agg_Rev/Total_Agg_Rev)) # 0.5139 of REV - The Top 60 SC's are 0.11235 of Total 
#%%
#                                                               Another Way: PLOT based on "USB"                                                        
df1__plot = df1_Grouped_by_SC_3a.xs('USB', level=0)['REV']
df1__plot.plot()
#%%
#                                                      Another Way: Sum of Revenue in Time Series based on "USB" 
df1_Grouped_by_SC_3a.xs('USB', level=0)['REV'].sum()
#%%
############################################################################### END: Applied: Y_to_D, Grouping on "SC": Aust_Extract and Aust_Cust_Sat
############################################################################### END

#%%   
#                                                              Grouping on TIME SERIES - step-1
b1a = df1_Grouped_by_SC_3a.copy()
b1a = b1a.reset_index() # Works

b1b = df1_Grouped_by_SC_3b.copy()
b1b = b1b.reset_index() # Works

print(b1a.shape) # (777224, 14)
print(b1b.shape) # (951262, 14)
#%%
#                                                      Grouping on TIME SERIES  Continued ... step-2
b2a = df2_Grouped_by_SC_3a.copy()
b2a = b2a.reset_index() # Works

b2b = df2_Grouped_by_SC_3b.copy()
b2b = b2b.reset_index() # Works

print(b2a.shape) # (4291, 3)
print(b2b.shape) # (9142, 3)
#%%
#                                                         Grouping on TIME SERIES  Continued ...step-3

b3a = pd.merge(b1a, b2a, on=['MAD_CODE'], how='inner', indicator=True) 
b3b = pd.merge(b1b, b2b, on=['MAD_CODE'], how='inner', indicator=True) 
print(b3a.shape) # (37660, 17)
print(b3b.shape) # (91719, 17)
#%%
#                                                                        Drop Duplicate 
b3a = b3a.drop_duplicates(['REV'])
b3b = b3b.drop_duplicates(['REV'])
#%%
print(b3a.shape) # (26141, 17)
print(b3b.shape) # (56476, 17)
#%%
print(b3a.info())
print('\n')
print(b3b.info())
#%%
print(df1a['MAD_CODE'].nunique()) # 169714
print(df2a['MAD_CODE'].nunique()) # 3147
#%%
#                                                        Grouping on "agg" - There is no TIME_DATE COLUMN

#                                                     This is "Agg." Data for "a"=Y_to_Dat and "b"=2016-2017
 
print(df1_Grouped_by_SC_3_agg_sorted_a.shape)
print(df1_Grouped_by_SC_3_agg_sorted_b.shape) 

print(df2_Grouped_by_SC_3_agg_sorted_a.shape) 
print(df2_Grouped_by_SC_3_agg_sorted_b.shape)

a1a = df1_Grouped_by_SC_3_agg_sorted_a.copy()
a2a = df2_Grouped_by_SC_3_agg_sorted_a.copy()
a3a = pd.merge(a1a, a2a, left_index=True, right_index=True) # This is "AGG" format based on Common "SC" for "a" - Y_to_Date

a1b = df1_Grouped_by_SC_3_agg_sorted_b.copy()
a2b = df2_Grouped_by_SC_3_agg_sorted_b.copy()
a3b = pd.merge(a1b, a2b, left_index=True, right_index=True) # This is "AGG" format based on Common "SC" for "b" - 2016 to 2017
#%%
print(a3a.shape) # (3099, 13)
print(a3b.shape) # (5977, 13)
#%%
#                                                   Make "MAD_CODE" a normal index, in other words reset_index

a3_Added_index_a = a3a.reset_index()
a3_Added_index_b = a3b.reset_index()
print(a3_Added_index_a.shape) # (3099, 13)
print(a3_Added_index_b.shape) # (5977, 13)
#%%
############################################################################### FINDING OUTLIERS - START 
#        FINDING OUTLIERS - Exploring the Data "df1_Grouped_by_SC_3_agg_a" using Box and Whisker plot can applied on "b"                                                                 

a3_Added_index_a_explore = a3_Added_index_a.describe() # Description 
a3_Added_index_b_explore = a3_Added_index_b.describe() # Description 
#%%
plt.style.use('ggplot')
a3_Added_index_a['REV'].plot(kind='box')
#a3_Added_index_a.ix[:, ['REV', 'COST', 'PROFIT']].plot(kind='box') 
#%%
#
#                                                                     FINDING OUTLIERS
# IQR = (Q3 - Q1)
q1 = 3864960.430
q3 = 15138986.460
iqr = (q3-q1)
#%%
#                               Find values that are less than q1 - (1.5 * iqr) OR more that q3 + (1.5 * iqr) 
q1_less = q1 - (1.5 * iqr) 
q3_more = q3 + (1.5 * iqr) 
print(q1_less) # -13046078.615000002
print(q3_more) # 32050025.505000003
#%%
#                                                   Finding all the elements that are more than q3_more

a_1 = a3_Added_index_a[a3_Added_index_a['REV'] >= q3_more]
a_2 = a3_Added_index_a[a3_Added_index_a['REV'] < q1_less]
print(a_1.shape)
print(a_2.shape)
############################################################################### FINDING OUTLIERS - END
#%%
#                                               Finding elements in a column that start with a specific letter

Starting_with_X = a3_Added_index_a['MAD_CODE'].loc[a3_Added_index_a['MAD_CODE'].str.startswith('A', na=False)] # 147 elements start with 'X'
print(Starting_with_X.shape)
#%%
print(a3a['REV'].sum())              
print(a3_Added_index_a['REV'].sum())  
print(b3a['REV'].sum())              
print('\n')
print(a3b['REV'].sum())             
print(a3_Added_index_b['REV'].sum())  
print(b3b['REV'].sum())              
#%%
#                                                                       Dropping Columns
print(b3a.shape) # (26141, 17)
print(b3b.shape) # (56476, 17)
b3a.drop(['index_y','_merge'], axis=1, inplace=True)  
b3b.drop(['index_y','_merge'], axis=1, inplace=True)  
print(b3a.shape) # (26141, 15)
print(b3b.shape) # (56476, 15)
#%%
#                                                                       Renaming Columns 
b3a.rename(columns={'index_x':'YEAR_MONTH'}, inplace=True)
b3b.rename(columns={'index_x':'YEAR_MONTH'}, inplace=True)
#%%
#                                                                           Description 
b3_description_a = b3a.describe()
b3_description_b = b3b.describe()

a3_description_a = a3_Added_index_a.describe()
a3_description_b = a3_Added_index_b.describe()
#%%
#                                                                           Make Copy 
b4a = b3a.copy()
b4b = b3b.copy()
#%%
#                                                NOTE: Making the "YEAR_MONTH" the index for TIME SERIES Visuals
b4a = b4a.set_index('YEAR_MONTH')
b4b = b4b.set_index('YEAR_MONTH')
#%%
print(b4a.shape) # (26141, 14)
print(b4b.shape) # (56476, 14)
#%%
print(b4a.info()) # 2017-01-31 to 2017-10-31
print('\n')
print(b4b.info()) # 2016-01-31 to 2016-12-31
#%%
#                                               SLICE of merged datasets a3_Added_index based on MEDIAN
a3_Added_index_a.shape
a3_Added_index_median_a = a3_Added_index_a['REV'].median()
a3_Added_index_REV_above_Median_a = a3_Added_index_a[a3_Added_index_a['REV'] >= a3_Added_index_median_a]
a3_Added_index_REV_below_Median_a = a3_Added_index_a[a3_Added_index_a['REV'] < a3_Added_index_median_a]

a3_Added_index_b.shape
a3_Added_index_median_b = a3_Added_index_b['REV'].median()
a3_Added_index_REV_above_Median_b = a3_Added_index_b[a3_Added_index_b['REV'] >= a3_Added_index_median_b]
a3_Added_index_REV_below_Median_b = a3_Added_index_b[a3_Added_index_b['REV'] < a3_Added_index_median_b]
#%%
#                                                                   Descritopn of the data 

a3_Added_index_REV_above_Median_description_a = a3_Added_index_REV_above_Median_a.describe()
a3_Added_index_REV_below_Median_description_b = a3_Added_index_REV_below_Median_b.describe()
#%%
#                                                                           Visuals 
b4a[b4a['MAD_CODE'] == 'AABCCXDE000'].plot()
b4b[b4b['MAD_CODE'] == 'AABCCXDE000'].plot()
#%%
############################################################################### START
############################################################################### START
#                                                   This is SAVING All "SC" based on "DCF" - Save in "hold"
#%%
#                       BEST - WORKS VERSION - 4 - WORKS - overlayed plots - BEST - Overlayed and Subplots - BEST
for i in a3_Added_index_a['MAD_CODE']:
    plt.figure()
    f, axes = plt.subplots(2, 1)
    
    axes[0].tick_params(labelsize=5)
    axes[1].tick_params(labelsize=5)
    axes[0].set_title(i)
    axes[1].set_xlabel('Months')
    
    a0 = pd.concat([b4a[b4a['MAD_CODE'] == i][['DCF_REV']], b4b[b4b['MAD_CODE'] == i][['DCF_REV']]], axis=0) 
    del a0.index.name
    a00 = pd.pivot_table(a0, index=a0.index.month, columns=a0.index.year, values='DCF_REV', aggfunc='sum')
     
    b0 = pd.concat([b4a[b4a['MAD_CODE'] == i][['ON_T_SHP_PERCENT_TON']], b4b[b4b['MAD_CODE'] == i][['ON_T_SHP_PERCENT_TON']]], axis=0) 
    del b0.index.name
    b00 = pd.pivot_table(b0, index=b0.index.month, columns=b0.index.year, values='ON_T_SHP_PERCENT_TON', aggfunc='sum')
    
    axes[0].plot(a00, marker='o', label='APPLE') 
    axes[1].plot(b00, marker='o')  
    
    axes[0].set_ylabel('DCF % REV', fontsize=5)
    axes[1].set_ylabel('ON_T_SHP_PERCENT_TON', fontsize=5) 
    
    plt.legend(["Year 2016--2017 ", "Year 2017_To_Date"])
    plt.savefig('C:/Users/Nader/Desktop/hold2/'+i+'.jpg',dpi=900)
    plt.show()      
#%%
#                                                                       Pearson Correlation 
a3_Added_index_corr_a = a3_Added_index_a.corr(method='pearson')
a3_Added_index_corr_b = a3_Added_index_b.corr(method='pearson')
#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#%%
#                                                       Visual Representation of Pearson Correlation 
names = ['REV', 'COST', 'PROFIT', 'SHPCNT', 'TON', 'CF', 'CP', 'DCF','DCP','DCF_REV', 'ON_T_SHP_PERCENT_TON','DCF_REV','ON_T_SHP', 'CSI']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(a3_Added_index_corr_a, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation='vertical')
ax.set_yticklabels(names)
#%%
#                                                        Visual Representation of Pearson Correlation
names = ['REV', 'COST', 'PROFIT', 'SHPCNT', 'TON', 'CF', 'CP', 'DCF','DCP','DCF_REV', 'ON_T_SHP_PERCENT_TON','DCF_REV','ON_T_SHP', 'CSI']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(a3_Added_index_corr_b, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation='vertical')
ax.set_yticklabels(names)
#%%
#                                                                           ggplot
plt.style.use('ggplot')
#%%
from pandas.plotting import scatter_matrix
scatter_matrix(a3_Added_index_a, figsize=(8,8), s = 3, marker=".", diagonal = "kde")
plt.savefig("scatter_matrix_1.jpg", dpi = 700)
#%%
from pandas.plotting import scatter_matrix
scatter_matrix(a3_Added_index_b, figsize=(8,8), s = 3, marker=".", diagonal = "kde")
plt.savefig("scatter_matrix_2.jpg", dpi = 700)
#%%
#                                                 Annotating the points on the scatte plot - Version 1
plt.scatter(a3_Added_index_a['REV'], a3_Added_index_a['PROFIT'], alpha=0.5)
for i, j in enumerate(a3_Added_index_a['SC']):
    plt.annotate(j, (a3_Added_index_a['REV'][i], a3_Added_index_a['PROFIT'][i]), size=2)
#%%
#                                        Bubble Plot + Annotating the points on the scatte plot - Version 2
plt.scatter(a3_Added_index_a['REV'], a3_Added_index_a['PROFIT'], alpha=0.5, s=a3_Added_index_a['COST']/10000)
for i, j in enumerate(a3_Added_index_a['MAD_CODE']):
    plt.annotate(j, (a3_Added_index_a['REV'][i], a3_Added_index_a['PROFIT'][i]), size=1) 
plt.savefig('bubble_1.jpg', dpi=2000)
plt.show()
#%%
############################################################################### KMEANS CLUSTERING 
############################################################################### KMEANS CLUSTERING 
############################################################################### KMEANS CLUSTERING START
#%%
#                  Choosing the features we want to cluster based upon - RUN THIS PART 3 TIMES Choosing ONE OPTION BELOW

x = a3_Added_index_a.iloc[:,[13,11]].values # 12,11 (ON_T_SHP, DCF_REV) - (11, 12) = (ON_T_SHP, CSI) - Year to Date - ORIGINAL: [13,11]
#x = a3_Added_index_REV_above_Median.iloc[:,[12,10]].values 
#x = a3_Added_index_REV_below_Median.iloc[:,[12,10]].values # # SC / CSI ---> CSI / SC [0,6] Org. ---> [6, 0]
#%%
x = a3_Added_index_b.iloc[:,[13,11]].values # 12,10 (DCF_REV, CSI) - (11, 12) = (ON_T_SHP, CSI) - 2016 -> 2017
#x = a3_Added_index_REV_above_Median.iloc[:,[12,10]].values 
#x = a3_Added_index_REV_below_Median.iloc[:,[12,10]].values # # SC / CSI ---> CSI / SC [0,6] Org. ---> [6, 0]
#%%
from pandas import DataFrame
x_df = DataFrame(x)
#%%
# Find the ideal number of clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=4)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('wcss')
plt.show()
#%%
# Fit the kmeana model based on the ideal number clusters selected 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=5)
y_kmeans = kmeans.fit_predict(x)
#%%
#                                                                   Labels and Centroids 
centers = kmeans.cluster_centers_
labels = kmeans.labels_
#%%
#           Assigning CLASSES from the result of KMEANS back to a3_Added_index_a for later Training for Classification 

print(labels) # These are are the labels(classes) from kmeans 
labels2 = DataFrame(labels)
a3_Added_index_a_copy = a3_Added_index_a.copy()
type(a3_Added_index_a_copy)
a3_Added_index_a_copy_classes = pd.concat([a3_Added_index_a_copy, labels2], axis=1)
a3_Added_index_a_copy_classes.rename(columns={0:'Class'}, inplace=True) 
print(a3_Added_index_a_copy_classes.groupby('Class').size()) 
print(a3_Added_index_a_copy_classes.groupby('Class').size().sum()) 
#%%
mylist_in_class_0 = a3_Added_index_a_copy_classes[a3_Added_index_a_copy_classes['Class'] == 0]
mylist_in_class_1 = a3_Added_index_a_copy_classes[a3_Added_index_a_copy_classes['Class'] == 1]
mylist_in_class_2 = a3_Added_index_a_copy_classes[a3_Added_index_a_copy_classes['Class'] == 2]
#%%
plt.hist(a3_Added_index_a_copy_classes['Class'])
#%%
a3_Added_index_a_copy_classes['Class'].plot(kind='kde')
#%%
########################### METHOD-3-START ####################################
#                                                       BEST- Visualize the KMEANS - METHOD - 2 - BEST
plt.style.use('ggplot')
#%%
mylist_in_class_0_SC = mylist_in_class_0['MAD_CODE']
mylist_in_class_1_SC = mylist_in_class_1['MAD_CODE']
mylist_in_class_2_SC = mylist_in_class_2['MAD_CODE'] 

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s = 10, c= 'red', label='Cluster 0')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s = 10, c= 'orange', label='Cluster 1')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s = 10, c= 'pink', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'magenta', label = 'Centroids') # ADDED
plt.title('KMEANS Clusters_All')
plt.xlabel('CSI')
plt.ylabel('DCF_REV')

apple = list(mylist_in_class_0['MAD_CODE'])
banana = list(mylist_in_class_1['MAD_CODE'])
orange = list(mylist_in_class_2['MAD_CODE'])

for i, j in enumerate(apple): # mylist_in_class_0['SC']
    plt.annotate(apple[i],(x[y_kmeans == 0,0][i], x[y_kmeans == 0,1][i]), size=1.5)
for i, j in enumerate(banana):
    plt.annotate(banana[i],(x[y_kmeans == 1,0][i], x[y_kmeans == 1,1][i]), size=1.5)
for i, j in enumerate(orange):
    plt.annotate(orange[i],(x[y_kmeans == 2,0][i], x[y_kmeans == 2,1][i]), size=1.5)
plt.savefig('scatter_matrix_kmeans_1.jpg', dpi=900)   
###################### METHOD-3-END ###########################################    
#%%
############################################################################### KMEANS CLUSTERING 
############################################################################### KMEANS CLUSTERING 
############################################################################### KMEANS CLUSTERING END 
#%%
############################################################################### START: More Visuals 

import seaborn as sns
a3_Added_index_a.columns
z1 = a3_Added_index_a.groupby('SC').mean()
z1 = a3_Added_index_a.groupby(['SC','CSI']).mean()
hist = sns.distplot(a3_Added_index_a['CSI'])
#count = sns.countplot('SC', data=a3_Added_index)
scatter = sns.regplot(x='SHPCNT', y='TON', data=a3_Added_index_a)
bar = sns.barplot(x='SHPCNT', y='TON' , data=a3_Added_index_a)
box = sns.boxplot(x='SHPCNT', y='TON', data=a3_Added_index_a)
violin = sns.violinplot(x='SHPCNT', y='TON', data=a3_Added_index_a)
#---------
pair_grid = sns.PairGrid(a3_Added_index_a)
pair_grid = pair_grid.map_upper(sns.regplot)
pair_grid = pair_grid.map_lower(sns.kdeplot)
pair_grid = pair_grid.map_diag(sns.distplot, rug=True)
#%%
from matplotlib import pyplot
a3_Added_index_a.plot(kind='box', subplots=True, layout=(3,5), sharex=False, sharey=False)
pyplot.show()
#%%
b4a_USB = b4a[b4a['SC'] == 'USB']
#%%
#b3a_USB = b3a.xs('USB', level=0)
#b4a_USB
#b4a_USB['REV'].sum()
#%%
b4a_USB_SLICE = b4a_USB.ix['2017-03-31':'2017-12-31']
b4a_USB_SLICE['REV'].plot() 
b4a_USB_SLICE.plot(y=['REV','PROFIT','COST']) 
#%%
df1__plot = b4a_USB['PROFIT'] # REVENUE
df1__plot.plot()
#%%
b4a_USB.plot(y=['REV', 'PROFIT','COST'])
b4a_USB.plot(y=['REV', 'COST'])
df1_result_USB = seasonal_decompose(b4a_USB['REV'], model='additive', freq=1)
df1_result_USB.plot()
plt.show()
#%%
# Lag_plot 
from pandas.tools.plotting import lag_plot
lag_plot(b4a_USB['PROFIT'])
#%%
# Auto Correlation
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(b4a_USB['REV'])
from statsmodels.tsa.stattools import adfuller
answer = adfuller(b4a_USB['REV'])
print('Dickey-Fuller: ', answer[0])
print('p-val: ', answer[1])
for k, v in answer[4].items():
    print('	%s: %.4f' % (k, v))
#%%
############################################################################### END: More Visuals     
#%%
#                                                                       APPLYING PCA 
###############################################################################
###############################################################################
###############################################################################
a3_Added_index_cols_a = a3_Added_index_a.iloc[:, 1:].values
#a3_Added_index_cols_b = a3_Added_index_b.iloc[:, 1:].values
x = a3_Added_index_cols_a.copy()
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x = pca.fit_transform(x)
Explained_Variance_Ratio = pca.explained_variance_ratio_
#%%
# Find the ideal number of clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=4)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('wcss')
plt.show()
#%%
#                                           Fit the kmeana model based on the ideal number clusters selected 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=5) # ORGINAL: n_clusters=3
y_kmeans = kmeans.fit_predict(x)
#%%
#                                                                   Centroids and Labels 
centers = kmeans.cluster_centers_
labels = kmeans.labels_
#%%
#                                                                       Added - Works
#%%
#           Assigning CLASSES from the result of KMEANS back to a3_Added_index_a for later Training for Classification 

print(labels) 
labels2 = DataFrame(labels)
#print(labels2)
#labels3 = list(labels)
a3_Added_index_a_copy = a3_Added_index_cols_a.copy()
type(a3_Added_index_a_copy)
a3_Added_index_a_copy = DataFrame(a3_Added_index_a_copy) # ADDED NEW
labels2.rename(columns={0:'Class'}, inplace=True) #ADDED NEW to change column name 
a3_Added_index_a_copy_classes = pd.concat([a3_Added_index_a_copy, labels2], axis=1)
print(a3_Added_index_a_copy_classes.groupby('Class').size()) 
print(a3_Added_index_a_copy_classes.groupby('Class').size().sum()) 
#%%
#                                                           Need to Add the "SC" Column as well
a3_Added_index_a_copy_classes_added_SC_column = pd.concat([a3_Added_index_a_copy_classes, a3_Added_index_a['MAD_CODE']], axis=1)
#%%
#                                                                      Renaming columns 
a3_Added_index_a_copy_classes_added_SC_column.rename(columns={0:'REV', 1:'COST', 2:'PROFIT', 3:'SHPCNT',
                                                              4:'TON', 5:'CF', 6:'CP', 7:'DCF', 8:'DCP',
                                                              9:'ON_T_SHP_PERCENT_TON', 10:'DCF_REV', 11:'ON_T_SHP', 12:'CSI'}, inplace=True)
#%%
mylist_in_class_0 = a3_Added_index_a_copy_classes_added_SC_column[a3_Added_index_a_copy_classes_added_SC_column['Class'] == 0]
mylist_in_class_1 = a3_Added_index_a_copy_classes_added_SC_column[a3_Added_index_a_copy_classes_added_SC_column['Class'] == 1]
mylist_in_class_2 = a3_Added_index_a_copy_classes_added_SC_column[a3_Added_index_a_copy_classes_added_SC_column['Class'] == 2]
mylist_in_class_3 = a3_Added_index_a_copy_classes_added_SC_column[a3_Added_index_a_copy_classes_added_SC_column['Class'] == 3] # ADDED
print(mylist_in_class_0.shape) # 172 / 131
print(mylist_in_class_1.shape) # 54 / 32
print(mylist_in_class_2.shape) # 39 / 35
print(mylist_in_class_3.shape) # 39 / 67 # ADDED
#%%
############################ METHOD-3-START ###################################
#                                                                    BEST - Method-2 - BEST

mylist_in_class_0_SC = mylist_in_class_0['MAD_CODE']
mylist_in_class_1_SC = mylist_in_class_1['MAD_CODE']
mylist_in_class_2_SC = mylist_in_class_2['MAD_CODE'] 
mylist_in_class_3_SC = mylist_in_class_3['MAD_CODE'] 

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s = 10, c= 'red', label='Cluster 0')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s = 10, c= 'orange', label='Cluster 1')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s = 10, c= 'pink', label='Cluster 2')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s = 10, c= 'green', label='Cluster 3') # ADDED
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'orchid', label = 'Centroids') # ADDED
plt.title('PCA Clusters_All')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.legend()


apple = list(mylist_in_class_0['MAD_CODE'])
banana = list(mylist_in_class_1['MAD_CODE'])
orange = list(mylist_in_class_2['MAD_CODE'])
pear = list(mylist_in_class_3['MAD_CODE']) # ADDED


for i, j in enumerate(apple): # mylist_in_class_0['SC']
    plt.annotate(apple[i],(x[y_kmeans == 0,0][i], x[y_kmeans == 0,1][i]), size=1.5)
for i, j in enumerate(banana):
    plt.annotate(banana[i],(x[y_kmeans == 1,0][i], x[y_kmeans == 1,1][i]), size=1.5)
for i, j in enumerate(orange):
    plt.annotate(orange[i],(x[y_kmeans == 2,0][i], x[y_kmeans == 2,1][i]), size=1.5)
for i, j in enumerate(pear): # ADDED
    plt.annotate(pear[i],(x[y_kmeans == 3,0][i], x[y_kmeans == 3,1][i]), size=1.5) # ADDED 
plt.savefig('scatter_matrix_PCA_1.jpg', dpi=900)

############################ METHOD-3-END #####################################    
#%%
# x[0]    
x_t = sc.transform([[  6.18328116e+07,   5.50180836e+07,   6.81472799e+06,
         2.71558000e+05,   2.80096840e+08,   1.50220778e+06,
         3.24526700e+05,   1.56218420e+05,   1.18301420e+05,
         8.76088601e-01,   2.89618624e-03,   8.76088601e-01,
         8.43614583e+01]])
#%%
x_t_pca = pca.transform(x_t)
#%%
print(kmeans.predict(x_t_pca))
#%%
###############################################################################
###############################################################################
###############################################################################
#%%
#                                                                       Applying SOM - 2
#%%
data = a3_Added_index_cols_a
data.shape
#data = a3_Added_index_cols_b
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data = sc.fit_transform(data)
#%%
from minisom import MiniSom
SOM = MiniSom(x=10, y=10, input_len=13, sigma=1.0, learning_rate=0.1)
SOM.random_weights_init(data)
SOM.train_random(data=data, num_iteration=500)
#%%
from pylab import bone, pcolor, colorbar, plot, show
pcolor(SOM.distance_map().T)
colorbar()
#%%
mappings = SOM.win_map(data)
outliers = np.concatenate((mappings[(1,4)], mappings[(2,5)]), axis=0)
outliers = sc.inverse_transform(outliers)
#%%
from pandas import DataFrame
outliers_df = DataFrame(outliers)
#%%
outliers_df_list = list(outliers_df[0])
#%%
apple1 = a3_Added_index_a[a3_Added_index_a['REV'] == 4468869.300] # WORKS finding individual - UCO
apple2 = a3_Added_index_a[a3_Added_index_a['REV'] == 1013989.4300000002] # WORKS finding individual - LLB
apple3 = a3_Added_index_a[a3_Added_index_a['REV'] == 20662395.839999996] # WORKS finding individual - NNC
#%%
list(set(outliers_df_list).intersection(a3_Added_index_a['REV']))  # finding the matching elements 
#%%            
#                                                                           WORKS
for i in a3_Added_index_a['REV']:
    for j in outliers_df_list:
        if i == j:
            print(a3_Added_index_a[a3_Added_index_a['REV'] == j])   
#%%       
#                                                                           WORKS            
apple6 = [a3_Added_index_a[a3_Added_index_a['REV'] == j] for i in a3_Added_index_a['REV'] for j in outliers_df_list if i == j]                  
#%%
#                                               Finding elements in a column that start with a specific letter
Starting_with_X = a3_Added_index_a['SC'].loc[a3_Added_index_a['SC'].str.startswith('X', na=False)] # 147 elements start with 'X'
print(len(Starting_with_X))
print(a3_Added_index_a.shape)
#%%
#%%
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%%
############################################################################### START - NEW 3MS
#A_df = pd.DataFrame() # Create Empty DataFrame
#A_mycopy1 = b4a.copy() 
#print(A_mycopy1.shape)                                                         # (26141, 15)
#
#A_mycopy1 = A_mycopy1.set_index('YEAR_MONTH')
#idx = pd.date_range('2017-01-01 00:00:00', '2017-10-31 00:00:00', freq='M')
#
#print(A_mycopy1.shape)                                                         # (26141, 14)
#
#for i in set(A_mycopy1['MAD_CODE']):
#    A_mycopy2 = A_mycopy1.loc[A_mycopy1['MAD_CODE'] == i] 
#    A_mycopy2 = A_mycopy2.reindex(idx, fill_value=0)
#    A_mycopy2['MAD_CODE'].replace(0, i, inplace=True) 
#    A_df = A_df.append(A_mycopy2)
#
#print(A_df.shape)                                                              #(30980, 14)
#%%
#                                                                   Using 3MS on "A_df" - VERSION 2
#                                         USING 3MS - Excludes Current Month from MEAN of Last 3 Months - CORRECT !

#b4a_copy = A_df.copy()
#b4a_copy = A_df.reset_index()
#print(b4a_copy.shape)                                                          # (30980, 15)
#b4a_copy.rename(columns={'index':'YEAR_MONTH'}, inplace=True)
#%%
b4a_copy = b4a.copy()
b4a_copy = b4a_copy.reset_index()
#%%
b4a_finding_defection = b4a_copy.groupby(['MAD_CODE', pd.Grouper(key='YEAR_MONTH', freq='QS')])['REV','COST','PROFIT','SHPCNT','TON','CF','CP','DCF','DCP'].mean()
print(b4a_finding_defection.shape) # (11374, 9) -> (11266, 9) -> (12392, 9)
#%%
#                                               Make "b4a_finding_defection" have its index become a column 
b4a_finding_defection_add_index = b4a_finding_defection.reset_index()
print(b4a_finding_defection_add_index.shape) # (11374, 11) -> (11266, 11) -> (12392, 11)
#%%
#                                                               Extracting the mean of last 3 months
b4a_finding_defection_add_index_Mean_LastThreeMonths = b4a_finding_defection_add_index.loc[b4a_finding_defection_add_index['YEAR_MONTH'] == '2017-07-01 00:00:00'] # OR: 2017-10-01 00:00:00 = 3MS OR 2017-10-31 00:00:00 using 3M
print(b4a_finding_defection_add_index_Mean_LastThreeMonths.shape) # (2939, 11) -> (2923, 11) -> (3098, 11)
#%%
#                                                               Extracting Only on the month
#                                                    TEST TEST TEST: removing duplicate entries 
b4a_test = b4a_copy.loc[b4a_copy['YEAR_MONTH'] == '2017-10-31 00:00:00'] 
b4a_test_2 = b4a_test[['YEAR_MONTH','MAD_CODE','REV']] # taking specific columns
print(b4a_test_2.shape) # (2402, 3) -> (3098, 3)
#%%
print(b4a_test_2.shape)                                           # (2402, 3)  -> (3098, 3)
print(b4a_finding_defection_add_index_Mean_LastThreeMonths.shape) # (2923, 11) -> (3098, 11)
#%%
print(b4a_test_2['MAD_CODE'].nunique())                                           # 2402 -> 3098
print(b4a_finding_defection_add_index_Mean_LastThreeMonths['MAD_CODE'].nunique()) # 2923 -> 3098
#%%
oak_test_1 = pd.merge(b4a_finding_defection_add_index_Mean_LastThreeMonths, b4a_test_2, on=['MAD_CODE'], how='inner')
oak_test_freq = oak_test_1['MAD_CODE'].value_counts()
oak_test_2 = oak_test_1.drop_duplicates()
oak_test_2['REV_RETURN_%'] = (((oak_test_2['REV_y'] / oak_test_2['REV_x']) -1) * 100)
print(oak_test_2.shape) # (2375, 14) -> (3098, 14)
#%%
oak3 = oak_test_2.loc[oak_test_2['REV_RETURN_%'] < -100] # Drop of More than 100%, compared to Mean of the past 3 months 
print(oak3.shape) # (1959, 14)
#%%
print(oak_test_2.shape[0]) # 2375
print(oak3.shape[0])       # 1959
Percent_Defection = (oak3.shape[0] / oak_test_2.shape[0])
print(Percent_Defection)
#%%
#                                                       Using 3MS on "A_df" - VERSION 2
############################################################################### END
#%%