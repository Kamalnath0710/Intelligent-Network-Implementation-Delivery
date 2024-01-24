# Databricks notebook source
#importing python and spark modules
from pyspark.sql.functions import *
from pyspark.sql.types import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# import pandas_profiling
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Outlier removal function

# COMMAND ----------

def outlier_removal_bottom(dataset,task_list,task_dictionary):
  new_df = pd.DataFrame()
  for i in task_list:
    dataset_2 = dataset[dataset['subIntervalName']== i].sort_values(by='subIntIntervalsDays')
    dataset_2 = dataset_2.reset_index()
    lower_limit = task_dictionary[i]
    dataset_3 = dataset_2[lower_limit::]
    new_df = new_df.append(dataset_3,ignore_index = True)
   
  return new_df

# COMMAND ----------

def task_group(df,groupby_col):
     grouped=(df
    .groupby(groupby_col)
    .agg(count('subIntIntervalsDays').alias('count')))         
     return grouped

# COMMAND ----------

# MAGIC %md
# MAGIC ###Stats function

# COMMAND ----------

def stats_col(df,groupby_col):
  out_df=(df
    .groupby(groupby_col)
    .agg(count('subIntIntervalsDays').alias('count'),
         mean('subIntIntervalsDays').alias('mean'),
         stddev('subIntIntervalsDays').alias('std'),
        min('subIntIntervalsDays').alias('min'),
         expr('percentile(subIntIntervalsDays, array(0.5))')[0].alias('percentile_50'),
         expr('percentile(subIntIntervalsDays, array(0.6))')[0].alias('percentile_60'),
         expr('percentile(subIntIntervalsDays, array(0.75))')[0].alias('percentile_75'),
         expr('percentile(subIntIntervalsDays, array(0.80))')[0].alias('percentile_80'),
         expr('percentile(subIntIntervalsDays, array(0.95))')[0].alias('percentile_95'),
         expr('percentile(subIntIntervalsDays, array(0.99))')[0].alias('percentile_99'),
         max('subIntIntervalsDays').alias('max')))
  return out_df


# COMMAND ----------

# MAGIC %md
# MAGIC # Data Warehouse Configuration

# COMMAND ----------

# Azure Blob Connection config
spark.conf.set(
  "fs.azure.account.key.storageaccountrgnpg9476.blob.core.windows.net",
  dbutils.secrets.get('npgroomtest','npgroomblob'))

spark.conf.set("fs.azure.account.key.dlsnpgroom.dfs.core.windows.net",dbutils.secrets.get('npgroomtest','npgroomdls'))
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

# Azure SQL Data Warehouse configuration
dwDatabase = dbutils.secrets.get('npgroomtest','dwDatabase')
dwServer = "asw-npgroomtest.sql.azuresynapse.net" #The Azure SQL Server
dwUser = dbutils.secrets.get('npgroomtest','dwUser') #The dedicated loading user login
dwPass = dbutils.secrets.get('npgroomtest','dwPass') #The dediciated loading user login password
dwJdbcPort =  "1433"
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ":" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser+";password=" + dwPass + ";" + "encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task Level Data - January 2022 - December 2022

# COMMAND ----------

#data extraction from Azure SQL Datawarehouse excluding inserted tasks
df_tasks = spark.read.format("jdbc") \
                            .option("url", sqlDwUrl) \
                            .option("query", """
SELECT distinct
   o.OrderID as orderID,
   s.subIntID,
   s._id,
   s.taskid,
   tbd.SubIntervalName as subIntervalName,
   s.owner,
   tbd.PID,
   p.phaseName,
   o.sourceSystem,
   s.startDate as startdate,
   s.completeDate as completedate,
   cc.daynuminyear - cs.daynuminyear AS subIntIntervalsDays,
   isnull(o.[ProjectType], 'None') as ProjectType,
   CASE When o.T_projecttype = '' Then 'None'
        When o.T_projecttype is Null then 'None'
        Else o.T_projecttype
        END as T_projecttype,
   isnull(o.IMPPOS2, 'None') as IMPPOS2,
   CASE When o.orderclasstype = '' Then 'None'
        When o.orderclasstype is Null then 'None'
        When o.orderclasstype = 'Null' then 'None'
        Else o.orderclasstype
        END as orderClassType,
   isnull(o.OrderAction, 'None') as orderAction,
   CASE When sa.orderDescription = '' Then 'None'
         When sa.orderDescription  is Null then 'None'
         Else sa.orderDescription 
         END as orderDescription,
   isnull(o.is3rdpartycoloneeded, 'false') as is3rdpartycoloneeded,
   isnull(sa.creatorGroup, 'None') as creatorGroup,
   isnull(sa.WorkGroup, 'None') as WorkGroup,
   isnull(sa.PDNOrder, 'false') as PDNOrder,
   isnull(sa.IsBuildingAdd, 'false') as isBuildingAdd,
   isnull(o.categoryCd, 'None') as categoryCD,
   isnull(sa.QtyOfCircuits , 0) as QtyOfCircuits,
   CASE When sa.Expedite = '' Then 'N'
         When sa.Expedite is Null then 'N'
         Else sa.Expedite
         END as Expedite,
   isnull(s.taskJeop, 'None') as taskJeop 
FROM
   [dbo].[GA_Orders] o 
   inner join
      [dbo].[GA_Orders_SupportingAttribute] sa 
      on sa.sourcekey = o.sourcekey 
   inner join
      [dbo].[GA_Subintervals] s 
      on s.sourcekey = o.sourcekey 
   inner join
      [dbo].[GA_tbd_SubInterval] tbd 
      on tbd.SubIntID = s.subIntID 
   inner join
      [dbo].[GA_tbd_Phases] p 
      on p.PID = tbd.PID 
   INNER JOIN
      dbo.ga_calendar cs 
      ON cs.date = Cast(s.startdate AS DATE) 
   INNER JOIN
      dbo.ga_calendar cc 
      ON cc.date = Cast(s.completedate AS DATE) 
where
   o.Status in ('Completed') and o.orderAction not in ('None', ' ')
   and o.sourceSystem in ('CORE','Netbuild','BPM-QS')
   and o.inidordertype != 3
   and o.completedate between '2022-01-01' and '2022-12-31' 
   and IncludedFlag in ('Y','F')
   and  s.deleteind = 'N'
   and s.isinserted != 1
   and tbd.isfdtask=0
""").load().toPandas()
df_tasks.shape[0]#1221460

# COMMAND ----------

#data display
display(df_tasks)

# COMMAND ----------

#Excluding Negative Intervals
df_tasks = df_tasks[df_tasks['subIntIntervalsDays']>=0]
#Task Intervals Days 0-->1
df_tasks.subIntIntervalsDays.replace(0,1,inplace=True)
#Replacing Order Action 'INS' as 'Install' and 'DIS' as 'Disconnect'
df_tasks.orderAction.replace(['INS','DIS'],['Install','Disconnect'],inplace=True)
#replacing isBuildingAdd unique value
df_tasks.isBuildingAdd.replace('False','false',inplace = True)
df_tasks.is3rdpartycoloneeded.replace('False','false',inplace = True)
df_tasks.Expedite.replace(['No','Yes'],['N','Y'],inplace = True)

# COMMAND ----------

#task interval days distribution
df_tasks.subIntIntervalsDays.describe().apply("{0:.5f}".format)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Outlier removal bottom 10

# COMMAND ----------

sp_tasks = spark.createDataFrame(df_tasks)
task_counts = task_group(sp_tasks, 'subIntervalName').toPandas()
task_counts['lower_limit'] = (task_counts['count']/10)
task_counts['lower_limit'] = task_counts['lower_limit'].astype('int64')
display(task_counts)

# COMMAND ----------

task_dict = {subIntervalName:int(lower_limit) for (subIntervalName,lower_limit) in zip(task_counts['subIntervalName'],task_counts['lower_limit'])}

# COMMAND ----------

print('Tasks with count greater than or equal to 10',task_counts[task_counts['count']>=10].shape[0])
print('Tasks with count lesser than 10',task_counts[task_counts['count']<10].shape[0])

task_check = list(task_counts[task_counts['count']>=10]['subIntervalName'].unique())
lesser_count_tasks = list(task_counts[task_counts['count']<10]['subIntervalName'].unique())

# COMMAND ----------

df_lesser_count = df_tasks[df_tasks['subIntervalName'].isin(lesser_count_tasks)]

# COMMAND ----------

print('Before removing outliers', df_tasks[df_tasks['subIntervalName'].isin(task_check)].shape[0])
df_tasks = outlier_removal_bottom(dataset=df_tasks,task_list = task_check,task_dictionary = task_dict)
print('After removing outliers',df_tasks.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ###Outlier removal threshold

# COMMAND ----------

pd_tasks = spark.createDataFrame(df_tasks)

# COMMAND ----------

# Stats function call
df_stats = stats_col(pd_tasks, ['subIntervalName','subIntID'])
display(df_stats)

# COMMAND ----------

df_stats=df_stats.toPandas()

# COMMAND ----------

# Categorizing the Sub Interval days based on 99th percentile 
conditions = [(df_stats['percentile_99'] <=10),
              (df_stats['percentile_99'] >10) & (df_stats['percentile_99'] <= 30),
              (df_stats['percentile_99'] >30) & (df_stats['percentile_99'] <= 90),
              (df_stats['percentile_99'] >90)]

values = ['0-10', '11-30', '31-90', '90+'] 
df_stats['Category'] = np.select(conditions, values) 
display(df_stats)

# COMMAND ----------

# Outlier removal based on threshold
conditions_outlier = [(df_stats['Category'] =='0-10'),
              (df_stats['Category'] =='11-30') ,
              (df_stats['Category'] =='31-90') ,
              (df_stats['Category'] =='90+')]

values_outlier = [30,90,180,360]
df_stats['Outlier_threshold']=np.select(conditions_outlier, values_outlier) 

# COMMAND ----------

display(df_stats)

# COMMAND ----------

df_tasks_stats=df_tasks.merge(df_stats[['subIntervalName','Category',
      'Outlier_threshold']],on='subIntervalName')

# COMMAND ----------

display(df_tasks_stats)

# COMMAND ----------

print(df_tasks_stats.shape)
print(df_tasks_stats[df_tasks_stats['subIntIntervalsDays'] <=df_tasks_stats['Outlier_threshold']].shape)

# COMMAND ----------

# Filtering out outliers from the data
df_tasks_process=df_tasks_stats[df_tasks_stats['subIntIntervalsDays'] <=df_tasks_stats['Outlier_threshold']]
display(df_tasks_process)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving in Delta Table

# COMMAND ----------

#Create Spark DataFrame
task_preprocessed = spark.createDataFrame(df_tasks_process)
#Delta Table
task_preprocessed.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_2022_preprocessed_tasks")
# spark.sql("""CREATE TABLE ga_2022_preprocessed_tasks USING DELTA LOCATION '/mnt/delta/ga_2022_preprocessed_tasks'""")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Number of records after preprocessing

# COMMAND ----------

#checking no of records after preprocessing
spark.table('ga_2022_preprocessed_tasks').count()

# COMMAND ----------

display(df_tasks_process[(df_tasks_process['subIntervalName'] == 'Project Data Enrichment')  & (df_tasks_process['T_projecttype'] == 'Mileage Reduction') & (df_tasks_process['orderAction'] == 'Grooms')])