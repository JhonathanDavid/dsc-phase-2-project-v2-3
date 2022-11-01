# Phase 2 Project Description

King County Seattle

![awesome](https://beautifulwashington.com/images/king-county/Seattle%20downtown%20Alaskan%20way%20King%20county.jpg)


### Background
In this notebook, an analysis of King County sales data in the United States for years 2014-2015 will be conducted. The purpose of the analysis is to derive conclusions for business decision making purposes, affecting current homeowners and prospective buyers of this specific area. King county, is one of three Washington state counties that include Seattle, Bellevue and Tacoma area. It covers an area of of approximately 39 towns and cities. U.S Census Bureau stats indicate the county has a population of approximately 2.2 million people as of 2020.

### Business Understanding & Business Problem
Understanding that my business stakeholder can be a real estate agency, who would want to advice both buyers and sellers on this market, it is important to note that in this type of business, both buyers and sellers are interested in price. Therefore, it is important to understand the database first, navigage its features, identify what other categories besides price are availble to try to define and predict what exactly is the best correlation to price.


### The Data

This project uses the King County House Sales dataset, which can be found in  `kc_house_data.csv` in the data folder in this assignment's GitHub repository. The description of the column names can be found in `column_names.md` in the same folder. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions about what the data means.

### Hypotheses

Null hypothesis (H0): There is no relationship between our features and our target variable, price.  

Alternative hypothesis (Ha): There is a relationship between our features and our target variable, price.

I will be using a significance level (alpha) of 0.05 to make our determination, and will make our final recommendations accordingly.


5 Regression analysis and visualizations
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
10/26
 10/29/22, 7:44 PM student - Jupyter Notebook
  fig, ax = plt.subplots(1, 2,figsize=(7,6))
sns.histplot(df['price'], ax=ax[0]) ax[0].set_title('Original') sns.histplot(np.log(df['price']), ax=ax[1]) ax[1].set_title('Adjusting log') plt.show()
In [19]:
 localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
11/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  # visualize the relationship between the predictors and the target (price)
fig, axs= plt.subplots (1,3, sharey= True, figsize=(18,6))
for idx, channel in enumerate (['sqft_living', 'Grade1', 'bathrooms']):
df.plot(kind= 'scatter', x=channel, y='price', ax=axs[idx], label=chann plt.legend()
plt.show()
In [23]:
 localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
12/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  In [36]:
Out[36]:
# look for the outlier on the far right of sf living
df.loc[df['sqft_living']== 13540].T
  id date price bedrooms bathrooms sqft_living sqft_lot floors condition sqft_above sqft_basement yr_built yr_renovated zipcode lat long sqft_living15 sqft_lot15 Grade1
12764
1225069038 5/5/2014 2280000.00 7 8.00 13540 307752 3.00 Average 9410 4130.0 1999 0.00 98053 47.67 -121.99 4850 217800 12
 In [37]:
# drop this record by using the record the index
df.drop(12764, inplace=True)
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
13/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  #exploring other correlations
df_pairplot = df[['price','sqft_living','sqft_above','Grade1','bathrooms']] sns.pairplot(df_pairplot)
plt.show()
In [48]:
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
14/26

 10/29/22, 7:44 PM student - Jupyter Notebook
 It can be concluded that the highest correlation to price are sqft_living, sqft_above,grade 1 and bathrooms.
Visualizing these relationships we can see the linearity. We can take these relationships and run the model. the first one I would like to pick is sqft_living since it is the most linear to me.
5.0.1 Running a simple regression in Stats model with SF as a predictor
 # import libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf # build the formula
f = 'price~sqft_living'
# create a fitted model in one line
model=smf.ols(formula=f, data=df ).fit()
In [30]:
5.0.2 Regression Diagnostics Summary
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
15/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  In [31]:
model.summary()
Out[31]: OLS Regression Results Dep. Variable:
Model: Method: Date: Time: No. Observations: Df Residuals: Df Model: Covariance Type:
sqft_living 399.3236
Omnibus: Prob(Omnibus): Skew: Kurtosis:
395.468 0.000 1.820 14.302
Durbin-Watson: Jarque-Bera (JB): Prob(JB): Cond. No.
1.918 4905.344 0.00 6.90e+03
price OLS Least Squares Sat, 29 Oct 2022 16:08:37 835 833 1 nonrobust
R-squared: Adj. R-squared: F-statistic: Prob (F-statistic): Log-Likelihood: AIC: BIC:
0.631 0.630 1423.
2.21e-182 -12127. 2.426e+04 2.427e+04
coef
Intercept -3.755e+05 3.53e+04 -10.648 0.000 -4.45e+05 -3.06e+05
t
10.587 37.719 0.000 378.544 420.103
std err
P>|t|
[0.025 0.975]
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. [2] The condition number is large, 6.9e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
5.0.3 Drawing a prediction line X(square feet living) and Y(price)
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
16/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  # create a DataFrame with the minimum and maximum values of sf
X_new=pd.DataFrame({'sqft_living': [df.sqft_living.min(), df.sqft_living.ma print(X_new.head())
# make predictions for those x values and store them
preds= model.predict(X_new)
print(preds)
# first, plot the observed data and the least squares line
df.plot(kind= 'scatter', x='sqft_living', y='price') plt.plot(X_new, preds, c='red', linewidth =2) plt.show()
In [32]:
   sqft_living
0          370
1        13540
0   -227744.63
1   5031347.07
dtype: float64
 5.0.4 Visualize error term for variance and heteroscedasticy
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
17/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "sqft_living", fig=fig) plt.show()
In [34]:
 In [35]:
5.0.5 Checking for normality assumptions by creating QQ plots
 # Code for QQ-plot here
import scipy.stats as stats
residuals =model.resid
sm.graphics.qqplot (residuals, dist=stats.norm, line='45', fit=True) plt.show()
 5.0.6 Repeating the above also for Grade as a predictor
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
18/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  # code for model, prediction line plot, heteroscedasticity check and QQ nor #step 1 through 3 is looking at database as a whole.
#Step 4 run Simple regression on radio only, just we did on TV only f='price~Grade1'
model= smf.ols(formula=f, data=df).fit() print ('R-Squared',model.rsquared) print (model.params)
#get regression diagnostics model.summary()
#Step 6 Draw a prediction line on scatter plot
X_new= pd.DataFrame({'Grade1':[df.Grade1.min(),df.Grade1.max()]}); preds= model.predict(X_new)
df.plot(kind='scatter', x='Grade1', y='price'); plt.plot(X_new,preds, c='red', linewidth=2);
plt.show()
#Visualize error term for variance Heteroscedasticity
fig= plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Grade1", fig=fig) plt.show()
#Normality check with QQ Plot
import scipy.stats as stats
residuals= model.resid
fig=sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
In [46]:
R-Squared 0.4652008564849478
Intercept   -1981572.05
Grade1        325851.16
dtype: float64
 localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
19/26

 10/29/22, 7:44 PM student - Jupyter Notebook
   5.0.7 Exploring more Correlations & Regression
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
20/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  In [56]:
sns.scatterplot(data =df,x = 'bathrooms',y= 'price',hue ='Grade1');
  In [58]:
In [59]:
In [60]:
number of bathrooms is positively correlated to price and it also helps us to conclude that the better the grade of a house, the more expensive it is
5.0.8 Creating X in a train and test models
X = df[['sqft_living','sqft_above','bathrooms','bedrooms','Grade1']] y = df['price']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, ra
 sqft = LinearRegression() sqft.fit(X_train[['sqft_living']], y_train) sqft.score(X_train[['sqft_living']], y_train) y_hat_train = sqft.predict(X_train[['sqft_living']]) y_hat_test = sqft.predict(X_test[['sqft_living']])
localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
21/26

 10/29/22, 7:44 PM student - Jupyter Notebook
  plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter(X_train[['sqft_living']], y_train, color = "blue") plt.plot(X_train[['sqft_living']] ,y_hat_train) plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('train')
plt.subplot(1,2,2)
plt.scatter(X_test[['sqft_living']], y_test, color = "green") plt.plot(X_test[['sqft_living']] ,y_hat_test) plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('test');
In [61]:
 localhost:8888/notebooks/dsc-phase-2-project-v2-3/student.ipynb
22/26

10/29/22, 7:44 PM student - Jupyter Notebook
Both train and set with linear correlation
reg = sm.add_constant(X, has_constant='add') model = sm.OLS(y, X)
result1 = model.fit()
result1.summary()
In [75]:
Out[75]: OLS Regression Results Dep. Variable:
Model: Method: Date: Time: No. Observations: Df Residuals: Df Model: Covariance Type:
R-squared (uncentered): Adj. R-squared (uncentered): F-statistic: Prob (F-statistic): Log-Likelihood: AIC: BIC:
sqft_living sqft_above bathrooms
bedrooms Grade1
coef
563.3322 -151.8250 7.487e+04 -1.476e+05 -1.164e+04
std err
32.620
36.106 3.03e+04 1.96e+04 1.01e+04
t
17.270 -4.205 2.470 -7.521 -1.147
P>|t| [0.025
0.000 499.305 0.000 -222.695 0.014 1.54e+04 0.000 -1.86e+05 0.252 -3.16e+04
1.910 3688.194 0.00 8.51e+03
Omnibus: Prob(Omnibus): Skew: Kurtosis:
368.835 0.000 1.736 12.699
Durbin-Watson: Jarque-Bera (JB): Prob(JB): Cond. No.
price OLS Least Squares Sat, 29 Oct 2022 17:54:14 834 829 5 nonrobust
0.829 0.828 803.4
6.86e-315 -12070. 2.415e+04 2.417e+04
0.975]
627.359 -80.955 1.34e+05 -1.09e+05 8280.837
Notes:
[1] R2 is computed without centering (uncentered) since the model does not contain a constant. [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[3] The condition number is large, 8.51e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
R-Squared indicates almost 83% can be explained by the model. The p-value less than 5% so we can reject null hypothesis and say that this model is statistically significant.

## Summary
```
├── README.md                        <- The top-level README for reviewers of this project
├── student.ipynb                    <- Jupyter notebook. Overview, Business Problem, Data Understanding.Regression.
├── RealEstate.pdf                   <- PDF version of project presentation
├── data                             <- the database used 
└── images                           <- sourced externally
```
