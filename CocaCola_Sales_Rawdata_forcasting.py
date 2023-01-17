# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:16:08 2022

@author: Mohd Ariz Khan
"""
# Importing data
import pandas as pd
df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df

# Get information about dataset
df.shape
list(df)
df.describe()

# Data Visvualization & EDA (Exploratory Data Analysis)

# let's scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'Sales')

# Boxplot
df.boxplot(column="Sales")

# Histogram
df["Sales"].hist()

# Bar graph
df.plot(kind="bar")

# Line Plot 
df.Sales.plot()

df.plot(kind="kde")

# Displot
import numpy as np
np.array(df["Sales"])

import seaborn as sns
sns.set_theme()
rk= sns.distplot(df['Sales'],kde=True)

# lag Plot
from pandas.plotting import lag_plot
lag_plot(df['Sales'])

len(df)


df['quarter'] = 0
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]

df
df['quarter'].value_counts()


df_dummies=pd.DataFrame(pd.get_dummies(df['quarter']),columns=['Q1','Q2','Q3','Q4'])
cc=pd.concat([df,df_dummies],axis= 1)
df.head()

cc

cc['t'] = np.arange(1,43)
cc['t_squared'] = cc['t']**2
cc["Sales_log"] =np.log(df['Sales'])  

cc.head()

# Split the data
train =cc.head(32)
test =cc.tail(10)

df['Sales'].plot()

# Use all forcasting models
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# Linear Model
linear_model =smf.ols("Sales~t",data =train).fit()
linear_pred = pd.Series(linear_model.predict(test['t']))
linear_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(linear_pred)))
linear_rmse

# Quadratic Model
quad_model =smf.ols("Sales~t+t_squared",data=train).fit()
quad_pred = pd.Series(quad_model.predict(test[['t','t_squared']]))
quad_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(quad_pred)))
quad_rmse

# Exponential model
exp_model  =smf.ols("Sales_log~t",data=train).fit()
exp_pred =pd.Series(exp_model.predict(test['t']))
exp_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(exp_pred)))
exp_rmse

# Additive seasonality model
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

# Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

# Multiplicative Seasonality
Mul_sea = smf.ols('Sales_log~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

# Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('Sales_log~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

# Compare the results 
data = {"MODEL":pd.Series(["linear_rmse","exp_rmse","quad_rmse","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([linear_rmse,exp_rmse,quad_rmse,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

# Predict for new time period
new_data = pd.read_excel("CocaCola_Sale_prediction.xlsx")
new_data

# Build the model on entire data set
model_full = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=cc).fit()

pred_new  = pd.Series(model_full.predict(new_data))
pred_new

# Gnearate new data
new_data["forecasted_Sales"] = pd.Series(pred_new)

df.shape

new_var = pd.concat([df,new_data])
new_var.shape
new_var.head()
new_var.tail()


new_var[['Sales','forecasted_Sales']].reset_index(drop=True).plot()

#====================================================================

