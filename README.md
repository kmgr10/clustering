# clustering
Clustering historical financial time series of returns into 3 regimes\
PCA used to describe visually the relationship between variables in each cluster\
Regress the performance of dependent and independent variables within each cluster to encode relationships\
Use the regression model to simulate scenarios 

Dependencies:
- Yahoo finance yfinance API to get historical data
- Usual libraries: pandas, numpy, sklearn

How to use:
- main.py calls the appropriate functions and market data
- Specify the working path as the 'directory' variable in main.py

To do:
- Data seems clean for my test cases, NaN removed by the pd diff() method used to calculate returns
- But need something more robust to look out for errors and outliers
