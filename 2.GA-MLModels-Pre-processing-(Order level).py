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

# COMMAND ----------

#Order level data -- completed orders of 2022
df_orders = spark.read.format("jdbc") \
                            .option("url", sqlDwUrl) \
                            .option("query", """
SELECT 
     o.OrderID as orderID,
    o.sourceSystem,
    cc.daynuminyear - cs.daynuminyear  AS orderIntervalDays,
    isnull(o.ProjectType , 'None') as ProjectType,
    CASE When o.T_projecttype = '' Then 'None'
        When o.T_projecttype is Null then 'None'
        Else o.T_projecttype
        END as T_projecttype,
    isnull(o.IMPPOS2,'None') as IMPPOS2,
    CASE When o.orderclasstype = '' Then 'None'
        When o.orderclasstype is Null then 'None'
        Else o.orderclasstype
        END as orderClassType,
	o.ordertype as orderType
    ,o.is3rdpartycoloneeded as is3rdpartycoloneeded
    ,sa.creatorGroup as creatorGroup
    ,sa.workGroup
    ,sa.IsBuildingAdd as isBuildingAdd
	,o.categoryCd as categoryCD
    ,o.OrderAction as orderAction,
    CASE When sa.orderDescription = '' Then 'None'
         When sa.orderDescription  is Null then 'None'
         Else sa.orderDescription 
         END as orderDescription ,
    CASE When sa.Expedite = '' Then 'N'
         When sa.Expedite is Null then 'N'
         Else sa.Expedite
         END as Expedite,
    o.CLECILEC
FROM [dbo].[GA_Orders] o
  inner join [dbo].[GA_Orders_SupportingAttribute] sa on sa.sourcekey = o.sourcekey
    INNER JOIN dbo.ga_calendar cs 
            ON cs.date = Cast(o.orderWfStart AS DATE) 
    INNER JOIN dbo.ga_calendar cc 
            ON cc.date = Cast(o.completedate AS DATE) 
where o.completedate is not null 
and o.inidordertype != 3
and o.sourceSystem in ('CORE','Netbuild','BPM-QS')
and o.orderwfStart is not null 
and o.orderAction not in ('None', ' ')
and o.status = 'Completed'
and o.completedate between '2022-01-01' and '2022-12-31' 
and IncludedFlag in ('Y','F')
""").load().toPandas()
df_orders.shape

# COMMAND ----------

#Removing Negative Values
df_orders = df_orders[df_orders['orderIntervalDays'] >= 0]
#Replacng 0 with 1
df_orders.orderIntervalDays.replace(0,1,inplace=True)
df_orders.orderAction.replace(['INS','DIS'],['Install','Disconnect'],inplace=True)
#replacing isBuildingAdd unique value
df_orders.isBuildingAdd.replace('False','false',inplace = True)
df_orders.is3rdpartycoloneeded.replace('False','false',inplace = True)
df_orders.Expedite.replace(['No','Yes'],['N','Y'],inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Category CD Binning

# COMMAND ----------

#number of unique category codes
print("Total unique values for categoryCD ", df_orders.categoryCD.nunique(),'\n')
df_orders.categoryCD.value_counts()

# COMMAND ----------

#dataframe consisiting CORE tasks
df_core = df_orders[df_orders.sourceSystem == 'CORE']
#category codes which have occured in less than 100 orders
ccd_to_replace = df_core.groupby('categoryCD').count()[df_core.groupby('categoryCD')['orderID'].count() < 100].index
print(ccd_to_replace)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Importing Preprocessed Data

# COMMAND ----------

#Getting the pre processed data 
pd_prediction = spark.table('ga_2022_preprocessed_tasks').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Binning Category CD

# COMMAND ----------

#Replacing category codes as "Others" if the category codes have occured in less than 100 orders
pd_prediction.categoryCD.replace(ccd_to_replace,'Others',inplace=True)

# COMMAND ----------

#Converting back to spark DataFrame
pd_subint= spark.createDataFrame(pd_prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Cancelled Orders for AMO Voice/911

# COMMAND ----------

#Cancelled Order records of AMO/VOICE 911 jeops of 2021
df_jeops = spark.read.format("jdbc") \
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
   s.startDate as startdate,
   s.completeDate completedate,
   cc.daynuminyear - cs.daynuminyear AS subIntIntervalsDays,
   isnull(o.[ProjectType], 'None') as ProjectType,
   isnull(o.[T_projectType], 'None') as T_projectType,
   isnull(o.IMPPOS2, 'None') as IMPPOS2,
   isnull(o.orderclasstype, 'None') as orderClassType,
   isnull(o.ordertype, 'None') as orderType
    ,isnull(o.OrderAction, 'None') as orderAction,
    isnull(sa.OrderDescription, 'None') as orderDescription,
   isnull(o.is3rdpartycoloneeded, 'false') as is3rdpartycoloneeded,
   isnull(sa.creatorGroup, 'None') as creatorGroup,
   isnull(sa.WorkGroup, 'None') as WorkGroup,
   isnull(sa.PDNOrder, 'false') as PDNOrder,
   isnull(sa.IsBuildingAdd, 'false') as isBuildingAdd,
   isnull(o.categoryCd, 'None') as categoryCD,
   isnull(sa.QtyOfCircuits , 0) as QtyOfCircuits,
   case when sa.Expedite = '' then 'N' else sa.Expedite end as Expedite,
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
   o.orderAction not in ('None', ' ')
  and o.sourceSystem in ('CORE','Netbuild','BPM-QS')
  and o.inidordertype != 3
  and s.completedate between '2022-01-01' and '2022-12-31'
  and IncludedFlag ='Y'
  and o.status = 'Cancelled'
  and s.taskjeop in ('AMO_VOIDEP(Voice Dependency)','AMO_911DEP(e911 Dependency)')
  and cc.daynuminyear - cs.daynuminyear >=0
""").load()
df_jeops.count()

# COMMAND ----------

# #spark to pandas dataframe
# jeops = df_jeops.toPandas()
# #replacing isBuildingAdd 'False' with 'false'
# jeops.isBuildingAdd.replace('False','false',inplace = True)
# df_jeops = spark.createDataFrame(jeops)

# COMMAND ----------

# #dropping extra columns
# pd_jeops = df_jeops.select('orderID', 'subIntID','subIntervalName', 'owner', 'PID', 'phaseName',
#        'sourceSystem', 'startDate', 'completeDate', 'subIntIntervalsDays',
#        'ProjectType', 'T_projecttype', 'IMPPOS2', 'orderClassType',
#        'orderAction', 'orderDescription', 'is3rdpartycoloneeded',
#        'creatorGroup', 'WorkGroup', 'PDNOrder', 'isBuildingAdd', 'categoryCD',
#        'QtyOfCircuits', 'Expedite', 'taskJeop').replace(0,1)

# COMMAND ----------

#adding cancelled orders of AMO/Voice 911 cancelled orders to the main dataframe
pd_subint = pd_subint.select('orderID', 'subIntID','subIntervalName', 'owner', 'PID', 'phaseName',
       'sourceSystem', 'startDate', 'completeDate', 'subIntIntervalsDays',
       'ProjectType', 'T_projecttype', 'IMPPOS2', 'orderClassType',
       'orderAction', 'orderDescription', 'is3rdpartycoloneeded',
       'creatorGroup', 'WorkGroup', 'PDNOrder', 'isBuildingAdd', 'categoryCD',
       'QtyOfCircuits', 'Expedite', 'taskJeop')#.union(pd_jeops)
pd_subint.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Saving in Delta table

# COMMAND ----------

pd_subint.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_test_subintervals_2022_preprocessed")
# spark.sql("""CREATE TABLE ga_test_subintervals_2022_preprocessed USING DELTA LOCATION '/mnt/delta/ga_test_subintervals_2022_preprocessed'""")

# COMMAND ----------

#checking no of records after preprocessing
spark.table('ga_test_subintervals_2022_preprocessed').count()
