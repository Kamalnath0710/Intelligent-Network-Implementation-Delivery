# Databricks notebook source
#%sh python -m pip install --upgrade pip

# COMMAND ----------

# MAGIC %sh pip install lightgbm

# COMMAND ----------

# MAGIC %sh pip install mlflow

# COMMAND ----------

#Inporting the libraries
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, auc, roc_auc_score,median_absolute_error
import mlflow
import mlflow.sklearn
from datetime import date
import datetime

# COMMAND ----------

def stats_col(df,groupby_col):
  out_df=(df
    .groupby(groupby_col)
    .agg(count('subIntIntervalsDays').alias('count'),
         mean('subIntIntervalsDays').alias('mean'),
         stddev('subIntIntervalsDays').alias('std'),
        min('subIntIntervalsDays').alias('min'),
         expr('percentile(subIntIntervalsDays, array(0.5))')[0].alias('median'),
         expr('percentile(subIntIntervalsDays, array(0.6))')[0].alias('percentile_60'),
         expr('percentile(subIntIntervalsDays, array(0.70))')[0].alias('percentile_70'),
         expr('percentile(subIntIntervalsDays, array(0.75))')[0].alias('percentile_75'),
         expr('percentile(subIntIntervalsDays, array(0.80))')[0].alias('percentile_80'),
         max('subIntIntervalsDays').alias('max')))
  return out_df

# COMMAND ----------

#Import Preprocessed data from Delta lake spark tables
pd_subint = spark.table('ga_test_2022_parallel_task_preprocessed').select('orderID','subIntID','subIntervalName', 'sourceSystem', 'PID','phaseName','subIntIntervalsDays', 'is3rdpartycoloneeded', 'orderClassType', 'IMPPOS2', 'creatorGroup', 'workGroup','T_projecttype', 'taskJeop', 'Expedite', 'orderAction', 'categoryCD', 'QtyOfCircuits', 'orderDescription','isBuildingAdd').toPandas()
#Checking the number of records 
print(pd_subint.shape)
#Display
display(pd_subint)

# COMMAND ----------

 # Converting 'object' and 'bool' datatypes into 'category' datatype. LighGBM Model can handle categorical features, that are of 'category' type, not 'object' 
categorical_features = pd_subint.select_dtypes(['object','bool'])
for column in categorical_features:
    pd_subint[column] = pd_subint[column].astype('category')
pd_subint['subIntID'] = pd_subint['subIntID'].astype('category')
pd_subint['PID'] = pd_subint['PID'].astype('category')

# COMMAND ----------

#Filtering Out the necessary columns from the train data
pd_subint = pd_subint[['orderID','subIntID','subIntervalName', 'sourceSystem', 'PID','phaseName','subIntIntervalsDays', 'orderClassType', 'T_projecttype', 'Expedite', 'orderAction', 'orderDescription','categoryCD','taskJeop']]

# COMMAND ----------

pd_subint.subIntIntervalsDays.replace(0,1,inplace=True)
pd_subint[pd_subint['subIntIntervalsDays']==0].shape



# COMMAND ----------

# MAGIC %md
# MAGIC ##Adding Flag

# COMMAND ----------

pd_subint['taskJeop_flagged']=np.where(pd_subint['taskJeop'] == "None", "N", "Y")

# COMMAND ----------

pd_subint['taskJeop_flagged'] = pd_subint['taskJeop_flagged'].astype('category')

# COMMAND ----------

pd_subint['taskJeop_flagged'].value_counts()

# COMMAND ----------

pd_subint.info()

# COMMAND ----------

#Predictors
X_train = pd_subint.drop('subIntIntervalsDays',axis=1)
#Label
Y_train = pd_subint['subIntIntervalsDays']

# COMMAND ----------

# MAGIC %md
# MAGIC ###Import Test Data

# COMMAND ----------

#Import test data from delta lake
pd_test = spark.table('ga_model_test_data_v3_2023').toPandas()
# pd_test['taskJeop'] = np.where(pd_test['taskJeop'] =="None", "N", "Y")

# COMMAND ----------

#Filtering out the columns from test data
pd_test = pd_test[['orderID','subIntID', 'sourceSystem', 'PID','subIntIntervalsDays','orderClassType', 'T_projecttype', 'Expedite', 'orderAction', 'orderDescription','categoryCD']]

# COMMAND ----------

# Converting 'object' and 'bool' datatypes into 'category' datatype. LighGBM Model can handle categorical features, that are of 'category' type, not 'object'
categorical_features_test = pd_test.select_dtypes(['object','bool'])
for column in categorical_features_test:
    pd_test[column] = pd_test[column].astype('category')
pd_test['subIntID'] = pd_test['subIntID'].astype('category')
pd_test['PID'] = pd_test['PID'].astype('category')

# COMMAND ----------

#Predictors
X_test = pd_test.drop('subIntIntervalsDays',axis=1)
#Label
Y_test = pd_test['subIntIntervalsDays']

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Training

# COMMAND ----------

def train_model( numleaves, nestimators, learningrate, maxdepth,objective,alpha):
  
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      med_ae = median_absolute_error(actual, pred)
      return rmse, mae, r2,med_ae
    
  np.random.seed(40)
  
  # Train, Test 
  x_train = X_train[['orderID','subIntID','subIntervalName', 'sourceSystem', 'PID','phaseName', 'orderClassType', 'T_projecttype', 'Expedite', 'orderAction', 'orderDescription','categoryCD']]
  y_train = Y_train
  x_test = X_test
  y_test = Y_test
  
  #setting up MLFlow experiment
  #mlflow.set_experiment('/Users/dheeraj.si@lumen.com/Ml-Model-Prac/GA-Test-Subinterval-Prediction_Prac') 
  name="Grooms-sprint-36 " + str(date.today())
  with mlflow.start_run(run_name = name):
    # loading lightgbm regression model 
    lgbModel = lgb.LGBMRegressor(learning_rate=learningrate, num_leaves=numleaves, n_estimators=nestimators, max_depth=maxdepth,objective=objective,alpha=alpha)
    
    lgbModel.fit(x_train.drop(['orderID','subIntervalName','phaseName'],axis = 1), y_train.values.ravel())
    predicted_days = lgbModel.predict(x_test.drop('orderID',axis=1))
    training_data = pd.concat([x_train,y_train],axis=1)
    test_data=x_test
    test_data['Actual']=y_test
    test_data['Prediction']=predicted_days
    #test_data = pd.concat([x_test,y_test,pd.DataFrame(predicted_days,columns = 'Predicted')],axis=1)
    (rmse, mae, r2,med_ae) = eval_metrics(y_test, predicted_days)
    
    # Log mlflow attributes for mlflow UI
    mlflow.log_param("learningrate", learningrate)
    mlflow.log_param("numleaves", numleaves)
    mlflow.log_param("n_estimators", nestimators)
    mlflow.log_param("max_depth", maxdepth)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lgbModel, "model")
    modelpath = "/dbfs/mlflow/ga/test/sprint46/lgbmodel-%f-%f-%f-%f-%d-%s" % (numleaves, nestimators,learningrate,maxdepth,alpha,str(datetime.datetime.now()))
    print("Model Path  -", modelpath)
    mlflow.sklearn.save_model(lgbModel, modelpath)
    mlflow.end_run()
  
    # Print out LightGBM metrics
    print("\nLightGBM Regression Model (numleaves=%f, nestimators=%f, learningrate=%f,maxdepth=%f):" % (numleaves, nestimators,learningrate,maxdepth))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    print("  Median Absolute Error: %s" % med_ae)
    return training_data,test_data

# COMMAND ----------

#Defining the paramters 
objective = 'quantile'
alpha = 0.75
numleaves = 128
nestimators = 300
learningrate = 0.1
maxdepth  = 12
#Model training function call
train_data,test_data=train_model(numleaves, nestimators, learningrate, maxdepth,objective,alpha)


# COMMAND ----------

#Previous model path :/dbfs/mlflow/ga/test/sprint36/lgbmodel-128.000000-300.000000-0.100000-12.000000-0-2023-07-18 09:49:19.086752

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train and Test Data sets for Model evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ##Test Data

# COMMAND ----------

#Round off prediction
test_data['Prediction']=test_data['Prediction'].round()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Saving Test data in Delta lake

# COMMAND ----------

# #converting to spark Data Frame
# ga_test_data = spark.createDataFrame(test_data)
# #Writing into Delta
# ga_test_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_test_data_V3")
# #spark.sql("""CREATE TABLE ga_test_data_V3 USING DELTA LOCATION '/mnt/delta/ga_test_data_V3'""")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Train Data

# COMMAND ----------

# MAGIC %md
# MAGIC ####Saving Train Data in Delta

# COMMAND ----------

#converting to spark Data Frame
ga_train_data = spark.createDataFrame(train_data)
#Writing into Delta
ga_train_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/delta/ga_train_data_Jan_dec_2022")
# spark.sql("""CREATE TABLE ga_train_data_Jan_dec_2022 USING DELTA LOCATION '/mnt/delta/ga_train_data_Jan_dec_2022'""")

# COMMAND ----------

display(spark.table('ga_train_data_Jan_dec_2022'))

# COMMAND ----------

spark.table('ga_train_data_Jan_dec_2022').count()