# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:16:08 2022

@author: Mohd Ariz Khan
"""
# Importing data
import pandas as pd
df = pd.read_excel("Airlines+Data.xlsx")

# Get information about dataset
df.shape
list(df)
df.describe()

# Data Visvualization & EDA (Exploratory Data Analysis)

# let's scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'Passengers')

# Boxplot
df.boxplot(column="Passengers", vert=False)

# Histogram
df["Passengers"].hist()

# Bar graph
df.plot(kind="bar")

# Line Plot 
df.Passengers.plot()

# Providing date, month, year 
df["Date"] = pd.to_datetime(df.Month,format="%b-%y")
df

df["month"] = df.Date.dt.strftime("%b") # month extraction
df["year"] = df.Date.dt.strftime("%Y") # year extraction

df

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index="year",columns="month",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# Boxplot for every
plt.figure(figsize=(8,6))
#plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=df)
#plt.subplot(212)
plt.figure(figsize=(8,6))
sns.boxplot(x="year",y="Passengers",data=df)

plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=df)



df

#==============================================================================
# Splitting data
df.shape
Train = df.head(84)
Test = df.tail(12)

Test

# use all forcasting models
import statsmodels.formula.api as smf 
import numpy as np

# Linear Model
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
#np.sqrt(np.mean((Test['Footfalls']-np.array(pred_linear))**2))
rmse_linear

# Exponential
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

# Quadratic 
Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
#pred_Quad = pd.Series(Exp.predict(pd.DataFrame(Test[["t","t_square"]))) # we hve to verify
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

# Additive seasonality 
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

# Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

# Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

# Compare the results 
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# Predict for new time period
new_data = pd.read_csv("Predict_new.csv")

new_data

#Build the model on entire data set
model_full = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=df).fit()

pred_new  = pd.Series(model_full.predict(new_data))
pred_new

#===========================================

# Gnearate new data
new_data["forecasted_Passengers"] = pd.Series(pred_new)

df.shape

new_var = pd.concat([df,new_data])
new_var.shape
new_var.head()
new_var.tail()


new_var[['Passengers','forecasted_Passengers']].reset_index(drop=True).plot()


