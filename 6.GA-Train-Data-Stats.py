# Databricks notebook source
# %sh python -m pip install --upgrade pip

# COMMAND ----------

# MAGIC %pip install lightgbm
# MAGIC %pip install graphviz

# COMMAND ----------

from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import lightgbm
import graphviz
import shap

# COMMAND ----------

def stats_col(df,groupby_col):
  out_df=(df
    .groupby(groupby_col)
    .agg(count('subIntIntervalsDays').alias('count'),
         min('subIntIntervalsDays').alias('min'),
         mean('subIntIntervalsDays').alias('mean'),
         expr('percentile(subIntIntervalsDays, array(0.5))')[0].alias('median'),
         expr('percentile(subIntIntervalsDays, array(0.6))')[0].alias('percentile_60'),
         expr('percentile(subIntIntervalsDays, array(0.70))')[0].alias('percentile_70'),
         expr('percentile(subIntIntervalsDays, array(0.75))')[0].alias('percentile_75'),
         expr('percentile(subIntIntervalsDays, array(0.80))')[0].alias('percentile_80'),
         max('subIntIntervalsDays').alias('max'),
         stddev('subIntIntervalsDays').alias('stddev')))
  return out_df

# COMMAND ----------

# # Percentile Function
# def percentile(n):
#     def percentile_(x):
#         return np.percentile(x, n)
#     percentile_.__name__ = 'percentile_%s' % n
#     return percentile_

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
# MAGIC ###Import Train Data

# COMMAND ----------

df_train = spark.table('ga_train_data_Jan_dec_2022')
print(df_train.count)
display(df_train)

# COMMAND ----------

df_train.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Training Data Statistics

# COMMAND ----------

train_stats=stats_col(df_train,['sourceSystem','subIntID','PID','orderAction' ,'Expedite','orderDescription','orderClassType','T_projecttype','categoryCD'])

for c in train_stats.columns:
  if c not in ['sourceSystem','subIntID','PID', 'orderAction' ,'Expedite','orderDescription','orderClassType','T_projecttype','categoryCD'] :
    train_stats=train_stats.withColumn(c,round(c))
display(train_stats)

# COMMAND ----------

pd_train_stats = train_stats.toPandas()
#Checking the count
print(pd_train_stats["count"].sum())
#Total combination in train data
print("Number of different combinations in trained data",pd_train_stats.shape[0])

# COMMAND ----------

# train_stats = spark.createDataFrame(pd_train.groupby([pd_train.sourceSystem,pd_train.subIntID, pd_train.PID,pd_train.is3rdpartycoloneeded,pd_train.orderAction,pd_train.Expedite, pd_train.isBuildingAdd,pd_train.orderDescription,pd_train.orderClassType,pd_train.T_projecttype,pd_train.categoryCD])[['subIntIntervalsDays']].agg(['count','min','mean',percentile(25),'median', percentile(60), percentile(75),percentile(80),percentile(99),'max','std']).reset_index(level=['sourceSystem','subIntID','PID', 'is3rdpartycoloneeded','orderAction' ,'Expedite','isBuildingAdd','orderDescription','orderClassType','T_projecttype','categoryCD']).round()).withColumnRenamed("('sourceSystem', '')","sourceSystem").withColumnRenamed("('T_projecttype', '')","T_projecttype").withColumnRenamed("('IMPPOS2', '')","IMPPOS2").withColumnRenamed("('orderClassType', '')","orderClassType").withColumnRenamed("('is3rdpartycoloneeded', '')","is3rdpartycoloneeded").withColumnRenamed("('creatorGroup', '')","creatorGroup").withColumnRenamed("('PID', '')","PID").withColumnRenamed("('subIntID', '')","subIntID").withColumnRenamed("('subIntervalName', '')","subIntervalName").withColumnRenamed("('orderAction', '')","orderAction").withColumnRenamed("('orderDescription', '')","orderDescription").withColumnRenamed("('categoryCD', '')","categoryCD").withColumnRenamed("('isBuildingAdd', '')","isBuildingAdd").withColumnRenamed("('Expedite', '')","Expedite").withColumnRenamed("('subIntIntervalsDays', 'min')", "min").withColumnRenamed("('subIntIntervalsDays', 'percentile_60')","percentile_60").withColumnRenamed("('subIntIntervalsDays', 'percentile_75')","percentile_75").withColumnRenamed("('subIntIntervalsDays', 'percentile_80')","percentile_80").withColumnRenamed("('subIntIntervalsDays', 'percentile_25')","percentile_25").withColumnRenamed("('subIntIntervalsDays', 'percentile_99')","percentile_99").withColumnRenamed("('subIntIntervalsDays', 'count')","count").withColumnRenamed("('subIntIntervalsDays', 'mean')","mean").withColumnRenamed("('subIntIntervalsDays', 'median')","median").withColumnRenamed("('subIntIntervalsDays', 'std')","std").withColumnRenamed("('subIntIntervalsDays', 'max')","max").drop_duplicates()
# display(train_stats)

# COMMAND ----------

pd_train_stats.columns

# COMMAND ----------

pd_train_stats['subIntID'] = pd_train_stats['subIntID'].astype('str')
pd_train_stats['PID'] = pd_train_stats['PID'].astype('str')

# COMMAND ----------

pd_train_stats['config'] = pd_train_stats['sourceSystem'] + '|' + pd_train_stats['subIntID'] + '|' + pd_train_stats['PID'] + '|' +    pd_train_stats['orderAction'] + '|' +  pd_train_stats['Expedite'] + '|' +  pd_train_stats['orderDescription'] + '|' + pd_train_stats['orderClassType'] + '|' +  pd_train_stats['T_projecttype'] + '|' + pd_train_stats['categoryCD']  

# COMMAND ----------

display(pd_train_stats)

# COMMAND ----------

pd_train_stats.columns

# COMMAND ----------

#reorder columns for train stats
column_names = ['config','sourceSystem', 'subIntID', 'PID',
       'orderAction', 'Expedite', 'orderDescription',
       'orderClassType', 'T_projecttype', 'categoryCD', 'count', 'min', 'mean',
       'median', 'percentile_60', 'percentile_70', 'percentile_75',
       'percentile_80', 'max', 'stddev']
pd_train_stats = pd_train_stats.reindex(columns = column_names)
display(pd_train_stats)

# COMMAND ----------

df_tasks = spark.read.format("jdbc") \
                            .option("url", sqlDwUrl) \
                            .option("query", """
SELECT 
   tbd.subIntID as subIntID,
   tbd.SubIntervalName as subIntervalName,
   tbd.PID,
   p.phaseName
   
FROM [dbo].[GA_tbd_SubInterval] tbd 
inner join [dbo].[GA_tbd_Phases] p 
      on p.PID = tbd.PID 
where tbd.isfdtask=0  

""").load().toPandas()
df_tasks.shape[0]

# COMMAND ----------

df_tasks['subIntID'] = df_tasks['subIntID'].astype('str')

# COMMAND ----------

pd_final_stats = pd_train_stats.merge(df_tasks[["subIntID","subIntervalName","phaseName"]],on ='subIntID',how='inner')
display(pd_final_stats)

# COMMAND ----------

pd_final_stats["count"].sum()

# COMMAND ----------

column_names_1 = ['config','sourceSystem', 'subIntID', 'subIntervalName','PID','phaseName',
       'orderAction', 'Expedite', 'orderDescription',
       'orderClassType', 'T_projecttype', 'categoryCD', 'count', 'min', 'mean',
       'median', 'percentile_60', 'percentile_70', 'percentile_75',
       'percentile_80', 'max', 'stddev']
pd_final_stats = pd_final_stats.reindex(columns = column_names_1)
display(pd_final_stats)

# COMMAND ----------

df_train_data_stats = spark.createDataFrame(pd_final_stats)

# COMMAND ----------

df_train_data_stats.write.format("delta").mode("overwrite").option("overwriteSchema","true").save("/mnt/delta/ga_train_data_Jan_dec_2022_stats")
#spark.sql("""CREATE TABLE ga_train_data_Jan_dec_2022_stats USING DELTA LOCATION '/mnt/delta/ga_train_data_Jan_dec_2022_stats'""")

# COMMAND ----------

delta_stats = spark.table('ga_train_data_Jan_dec_2022_stats')
display(delta_stats)

# COMMAND ----------

delta_stats.count()

# COMMAND ----------

display(spark.table('ga_train_data_Jan_dec_2022'))

# COMMAND ----------

display(spark.table('ga_train_data_Jan_dec_2022_stats'))

# COMMAND ----------

display(pd_final_stats[(pd_final_stats['subIntervalName'] == 'Project Data Enrichment') & (pd_final_stats['T_projecttype'] == 'Mileage Reduction') & (pd_final_stats['orderDescription'] == 'Offnet Groom')])