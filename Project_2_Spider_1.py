# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:34:43 2018

@author: 1000091
"""
#
import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from math import sqrt

from matplotlib import pyplot as plt
#################
# My Jupyter notebook gived the error: hence writing it in Spyder
#################
import seaborn as sns

cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

print (df.head(5))
print (df.columns)
df.describe().transpose()
print (df.describe().transpose())
print(df.isnull().any())
df1 = df[df.overall_rating.isnull()]
print(df1.head(5))

# droo the nulls

df = df[~df.overall_rating.isnull()]

print(df.isnull().any())

df["volleys"].fillna(df["volleys"].mean(),inplace=True)
df["curve"].fillna(df["curve"].mean(),inplace=True)
df["agility"].fillna(df["agility"].mean(),inplace=True)
df["balance"].fillna(df["balance"].mean(),inplace=True)
df["jumping"].fillna(df["jumping"].mean(),inplace=True)
df["vision"].fillna(df["vision"].mean(),inplace=True)
df["sliding_tackle"].fillna(df["sliding_tackle"].mean(),inplace=True)

# Data to Numeric

print (df.preferred_foot.value_counts())

df['preferred_foot'] = df['preferred_foot'].map( {'right': 0, 'left': 1} ).astype(int)

df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('_0','0')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('ormal','5')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('o','0')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('l0w','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('0','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('1','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('2','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('3','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('4','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('5','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('6','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('7','high')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('8','high')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('9','high')

print (df.defensive_work_rate.value_counts())

df = df[(df.defensive_work_rate == 'medium') | (df.defensive_work_rate == 'high') | (df.defensive_work_rate == 'low')]

# Check for attacking work rate
df.attacking_work_rate.value_counts()

# Change "norm" to "medium" and drop the rest having "None" and "Null" values.

df['attacking_work_rate'] = df['attacking_work_rate'].str.replace('norm','medium')
df = df[(df.attacking_work_rate == 'medium') | (df.attacking_work_rate == 'high') | (df.attacking_work_rate == 'low')]

print (df.head(50))

# Categorical data to be nullified
df_dummified = pd.get_dummies(df,columns=['attacking_work_rate','defensive_work_rate'])
print (df_dummified.columns)

# prepare X and Y data for regression

X = df_dummified[['potential', 'preferred_foot', 'crossing', 'finishing',
       'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
       'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration',
       'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
       'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
       'interceptions', 'positioning', 'vision', 'penalties', 'marking',
       'standing_tackle', 'sliding_tackle',
       'attacking_work_rate_high', 'attacking_work_rate_low',
       'attacking_work_rate_medium', 'defensive_work_rate_high',
       'defensive_work_rate_low', 'defensive_work_rate_medium']]

Y = df_dummified["overall_rating"]

model = LinearRegression()
model = model.fit(X, Y)
model.score(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)
model2 = LinearRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)


print("The accuracy is= \n",metrics.r2_score(y_test, predicted))

# Coefficients?

pd.DataFrame({"features": X.columns, "co-efficients": model.coef_})

plt.scatter(y_test, predicted)
plt.plot([40, 100], [40, 100], '--k')
plt.xlabel("True overall score")
plt.ylabel("Predicted overall score")
plt.title("True vs Predicted overall score")
plt.show()

plt.figure(figsize=(9,6))
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=30, xmax=100)
plt.ylabel('Residuals')
plt.title('Residual plot including training(blue) and test(green) data')
plt.show()



scores  =  cross_val_score(LinearRegression(), X, Y, cv=10)
print ("Scored and mean \n===",scores, scores.mean())

predictions = cross_val_predict(LinearRegression(), X, Y, cv=7)
print(metrics.r2_score(y_test, predicted))

# Now Decision tree
model = DecisionTreeRegressor()
model = model.fit(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model2 = DecisionTreeRegressor(random_state=0)
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
print ("Accuracy score is", round(metrics.r2_score(y_test, predicted) * 100, 2), '%')

scores = cross_val_score(DecisionTreeRegressor(), X, Y, cv=10)
scores, scores.mean()

predictions = cross_val_predict(DecisionTreeRegressor(), X, Y, cv=7)
r2_score = metrics.r2_score(y_test, predicted)
print("Accuracy after Cross validation: {}".format(r2_score * 100))

parameters = [{'max_depth': range(25, 30), 'min_samples_split': range(2, 10)}]

reg = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters, scoring='neg_mean_squared_error')
reg.fit(X_train, y_train)

print("Best parameters set found on development set:\n")
print(reg.best_params_)

print("Accuracy for test data set:\n")
predicted = reg.predict(X_test)
print("FInal accuracy now is = \n\n",metrics.r2_score(y_test, predicted))

print ("\n=============Visualize============\n")
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']

correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]

# create a function for plotting a dataframe with string columns and numeric values
def plot_dataframe(df, y_label):  
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75); #Notice the ; (remove it and see what happens !)
    plt.show()
    
# create a dataframe using cols and correlations
df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 

# let's plot above dataframe using the function we created
plot_dataframe(df2, 'Player\'s Overall Rating')





