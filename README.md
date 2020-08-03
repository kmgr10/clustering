# clustering
Clustering historical financial time series of returns into 3 regimes\
PCA used to explore the meaning of the clusters

Dependencies:
- Yahoo finance yfinance API to get historical data
- Usual libraries: pandas, numpy, sklearn

To do:
- Data seems clean for my test cases, NaN removed by the pd diff() method used to calculate returns
- But need something more robust to look out for errors and outliers
