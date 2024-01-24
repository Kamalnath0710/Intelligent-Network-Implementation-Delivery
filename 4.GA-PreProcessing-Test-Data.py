# Databricks notebook source
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

# Percentile Function
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

# COMMAND ----------

# Azure Blob Connection config
spark.conf.set(
  "fs.azure.account.key.storageaccountrgnpg9476.blob.core.windows.net",
  "6ybfxqn1wdp9fazbLG3aTShQBl9Bs51rHnNnv1/vdobntHoE+WjApOODhbCIPR5T9KaGZjgPezZkC9ZhNmtkHw==")
# Azure Blob Connection config
spark.conf.set("fs.azure.account.key.dlsnpgroom.dfs.core.windows.net", "BrAj4aua7RglVzIcFEs3z4AGBxN716f3is6YlSVaDGsLqcxI0NdygNKQT+u1grLzVX4gTfs2xVhDHU/FYdwEZg==")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

# Azure SQL Data Warehouse configuration
dwDatabase = "groomdw"
dwServer = "asw-npgroomtest.sql.azuresynapse.net" #The Azure SQL Server
dwUser = 'sqladminuser' #The dedicated loading user login
dwPass = 'Grooms$123' #The dediciated loading user login password
dwJdbcPort =  "1433"
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ":" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser+";password=" + dwPass + ";" + "encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task Level Data - Jan2023 to Mar2023

# COMMAND ----------

df_tasks = spark.read.format("jdbc") \
                            .option("url", sqlDwUrl) \
                            .option("query", """
SELECT distinct
   o.OrderID as orderID,
   s.subIntID,
   tbd.SubIntervalName as subIntervalName,
   s.owner,
   tbd.PID,
   p.phaseName,
   o.sourceSystem,
   s.originalstartDate as startDate,
   s.originalcompleteDate as completeDate,
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
      ON cs.date = Cast(s.originalstartdate AS DATE) 
   INNER JOIN
      dbo.ga_calendar cc 
      ON cc.date = Cast(s.originalcompletedate AS DATE) 
where
   o.Status in ('Completed') and o.orderAction not in ('None', ' ')
   and o.completedate between '2023-01-01' and '2023-03-31' 
   and o.sourceSystem in ('CORE','Netbuild','BPM-QS')
   and IncludedFlag in ('Y','F')
   and  s.deleteind = 'N'
   and o.inidordertype != 3
   and s.isinserted != 1
   and tbd.isfdtask=0
""").load().toPandas()
df_tasks.shape[0]

# COMMAND ----------

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

df_tasks.subIntIntervalsDays.describe().apply("{0:.5f}".format)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Excluding Jeop records

# COMMAND ----------

pd_tasks = df_tasks[df_tasks['taskJeop']=='None']
pd_tasks.subIntIntervalsDays.describe().apply("{0:.5f}".format)

# COMMAND ----------

# pd_task1 = pd_tasks[pd_tasks['subIntervalName']=='Issue Manage ASR'].sort_values(by='subIntIntervalsDays')
# pd_task1 = pd_task1.reset_index()
# lower_limit = int(pd_task1.shape[0]/10)
# upper_limit = int((pd_task1.shape[0]/10) * 9)
# print(upper_limit,lower_limit)
# pd_task2 = pd_task1[lower_limit:upper_limit]
# pd_task2

# COMMAND ----------

# MAGIC %md
# MAGIC ###Outlier removal function()

# COMMAND ----------

def outlier_removal(dataset,new_df):
  tasks = pd_tasks.subIntervalName.unique()
  for i in tasks:
    dataset_2 = dataset[dataset['subIntervalName']== i].sort_values(by='subIntIntervalsDays')
    if dataset_2.shape[0] > 10:
      dataset_2 = dataset_2.reset_index()
      lower_limit = int(dataset_2.shape[0]/10)
      upper_limit = int((dataset_2.shape[0]/10) * 9)
      dataset_3 = dataset_2[lower_limit:upper_limit]
      new_df = new_df.append(dataset_3,ignore_index = True)
    else:
      new_df = new_df.append(dataset_2,ignore_index = True)
  return new_df   

# COMMAND ----------

#create Empty DataFrame
pd_tasks_preprocessed = pd.DataFrame()
#Function Call - Final DataFrame
pd_subint = outlier_removal(dataset=pd_tasks,new_df=pd_tasks_preprocessed) 
print("Number of records after removing outliers",pd_subint.shape[0])
display(pd_subint)

# COMMAND ----------

pd_subint.subIntIntervalsDays.describe().apply("{0:.5f}".format)

# COMMAND ----------

pd_subint.drop('index',axis=1,inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Adding Jeop Records

# COMMAND ----------

pd_jeops = df_tasks[df_tasks['taskJeop'] != 'None']
print("Number of records with Jeops",pd_jeops.shape[0])
pd_subintervals = pd.concat([pd_subint,pd_jeops],axis=0)
print("Number of records after adding Jeops",pd_subintervals.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving in Delta Table

# COMMAND ----------

#Create Spark DataFrame
task_preprocessed = spark.createDataFrame(pd_subintervals)
#Delta Table
task_preprocessed .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_model_test_data_v3_2023")
# spark.sql("""CREATE TABLE ga_model_test_data_v3_2023 USING DELTA LOCATION '/mnt/delta/ga_model_test_data_v3_2023'""")

# COMMAND ----------

