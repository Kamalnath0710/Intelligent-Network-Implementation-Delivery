# Databricks notebook source
#importing python and spark modules
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time
# import pandas_profiling
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,Getting the holidays from Lumen calendar
dates_calendar= spark.read.format("jdbc") \
                            .option("url", sqlDwUrl) \
                            .option("query", """
select date from ga_calendar
where isworkday = 0
and year(date) >=2015
""").load().toPandas()
holiday_list=dates_calendar['date'].to_numpy()
dates_calendar.shape[0]

# COMMAND ----------

# DBTITLE 1,Importing preprocessed ga_test_12m_data
#importing preprocessed data
pd_prediction = spark.table('ga_test_subintervals_2022_preprocessed').toPandas()
                             

# COMMAND ----------

#renaming startdate and enddate columns
pd_prediction.rename(columns={'startdate':'startDate','completedate':'completeDate'},inplace=True)

# COMMAND ----------

# DBTITLE 1,Converting datetime to date
#datatype conversion for start date and end date
pd_prediction['startDate']=pd.to_datetime(pd_prediction['startDate']).dt.date
pd_prediction['completeDate']=pd.to_datetime(pd_prediction['completeDate']).dt.date

# COMMAND ----------

#datatype conversion for subintid, PID, sub interval days
pd_prediction['subIntID']=pd_prediction['subIntID'].astype('int16')
pd_prediction['PID']=pd_prediction['PID'].astype('int16')
pd_prediction['subIntIntervalsDays']=pd_prediction['subIntIntervalsDays'].astype('int16')

# COMMAND ----------

# DBTITLE 1,Excluding tasks which occurred only once in an order
stats=pd_prediction[['orderID','subIntID','subIntervalName']].groupby(['orderID','subIntID']).agg({'subIntervalName':'count'}).reset_index()
stats.columns=['orderID','subIntID','count']

# COMMAND ----------

df=pd_prediction.merge(stats,on=['orderID','subIntID'])
df_one=df[df['count']==1]
df_process=df[df['count']>1]
print(df_one.shape)
print(df_process.shape)

# COMMAND ----------

display(df_process.head())

# COMMAND ----------

def All_dates(sd,cd):
  dates=[]
  range_dates=pd.date_range(sd,cd)
  date_list =[d.date() for d in range_dates ]
  for i in date_list:
    if i not in holiday_list:
       dates.append(i)
  return dates

# COMMAND ----------

df_process['B_dates']=df_process.apply(lambda row :All_dates(row['startDate'],row['completeDate']),axis=1)

# COMMAND ----------

display(df_process)

# COMMAND ----------

agg_dict={}
for col in df_process.columns:
  if col=='B_dates':
    agg_dict[col]='sum'
  else:
    agg_dict[col]='first'

# COMMAND ----------

df_process_grouped=df_process.groupby(['orderID','subIntID'], as_index=False).agg(agg_dict)

# COMMAND ----------

df_process_grouped.drop(columns=['subIntIntervalsDays'],inplace=True)

# COMMAND ----------

df_process_grouped['subIntIntervalsDays']=df_process_grouped['B_dates'].apply(lambda x :len(set(x)))
df_process_grouped['B_dates']=df_process_grouped['B_dates'].astype('str')

# COMMAND ----------

display(df_process_grouped)

# COMMAND ----------

df_one.columns

# COMMAND ----------

df_one['B_dates']=df_one.apply(lambda row :All_dates(row['startDate'],row['completeDate']),axis=1)


# COMMAND ----------

df_one['B_dates']=df_one['B_dates'].astype('str')

# COMMAND ----------

df_final=pd.concat([df_process_grouped,df_one],axis=0)

# COMMAND ----------

display(df_final)

# COMMAND ----------

df_final.shape

# COMMAND ----------

df_final_sp=spark.createDataFrame(df_final)

# COMMAND ----------

#final count of number of records for training
print("Final number of records for training",df_final_sp.count())

# COMMAND ----------

df_final_sp.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_test_2022_parallel_task_preprocessed")
# spark.sql("""CREATE TABLE ga_test_2022_parallel_task_preprocessed USING DELTA LOCATION '/mnt/delta/ga_test_2022_parallel_task_preprocessed'""")

# COMMAND ----------

#checking no of records after preprocessing
spark.table('ga_test_2022_parallel_task_preprocessed').count()

