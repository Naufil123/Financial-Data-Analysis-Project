import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

print ("Part 1")
file_path = r'C:\Users\HP\Desktop\Naufil\STZ (1).csv'
data = pd.read_csv(file_path, engine='python')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('ME').last()
monthly_data['Return'] = monthly_data['Close'].pct_change() * 100
monthly_data = monthly_data.dropna()
small_constant = 1e-8
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(monthly_data.index, np.log(monthly_data['Close'] + small_constant))
plt.title('Logarithm of Stock Price and Return')
plt.ylabel('Log(Stock Price)')
plt.subplot(2, 1, 2)
plt.plot(monthly_data.index, np.log(1 + monthly_data['Return'] + small_constant))
plt.xlabel('Date')
plt.ylabel('Log(Return)')
plt.show()
######################### Part  2 ##################################


print ("Part 2")
print(monthly_data.columns)
spy_returns_data = monthly_data[['Adj Close']].dropna()
stock_summary_stats = monthly_data[['Return']].describe()
stock_skewness = monthly_data[['Return']].skew()
stock_kurtosis = monthly_data[['Return']].kurtosis()
print("Part 2: Summary Statistics for the Assigned Stock:")
print(stock_summary_stats)
print("\nSkewness for the Assigned Stock:")
print(stock_skewness)
print("\nKurtosis for the Assigned Stock:")
print(stock_kurtosis)
spy_summary_stats = spy_returns_data.describe()
spy_skewness = spy_returns_data.skew()
spy_kurtosis = spy_returns_data.kurtosis()
print("\nSummary Statistics for SPY:")
print(spy_summary_stats)
print("\nSkewness for SPY:")
print(spy_skewness)
print("\nKurtosis for SPY:")
print(spy_kurtosis)

######################### Part  3 ##################################
print ("Part 3")
plt.figure(figsize=(8, 6))
sns.regplot(x=monthly_data['Return'], y=spy_returns_data['Adj Close'], ci=None)
plt.title('Scatter Plot of Assigned Stock Returns vs. SPY Returns')
plt.xlabel('Assigned Stock Returns')
plt.ylabel('SPY Returns')
plt.show()

######################### Part  4 ##################################

X = sm.add_constant(spy_returns_data)
model = sm.OLS(monthly_data['Return'], X)
results = model.fit()
print("\nPart 4: Linear Regression Summary:")
print(results.summary())


######################### Part  5 ##################################

spy_returns_data['SPY_Squared'] = spy_returns_data['Adj Close']**2
X_quad = sm.add_constant(spy_returns_data[['Adj Close', 'SPY_Squared']])
model_quad = sm.OLS(monthly_data['Return'], X_quad)
results_quad = model_quad.fit()
print("\nPart 5: Quadratic Regression Summary:")
print(results_quad.summary())





