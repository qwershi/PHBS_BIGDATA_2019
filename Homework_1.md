# Spread Detection on China's Stock Market Capitalization Based on Big Data Analysis

## Xihang Yu 
## 1801212965

## Overview

### 1. Introduction to China's Stock Index Futures
`Stock index futures` are standardized futures contracts with certain stock index as the underlying asset. The prices quoted by the buyers and the sellers are the stock index prices in a certain period in future assumed by them. After the contract expires, stock index futures will be settled in cash.

The underlying assets this strategy is going to trade are `SSE 50 Index Futures` and `CSI 500 Index Futures`. 

`SSE 50 stock index` is the SSE 50 index futures’ underlying asset. This index selects 50 largest stocks of good liquidity and representativeness from Shanghai security market by scientific and objective method. The objective is to reflect the complete picture of those good quality large enterprises, which are most influential in Shanghai security market. 

`CSI 500 stock index` is the CSI 500 index futures’ underlying asset. This index selects 500 Middle and small stocks of good liquidity and representativeness from Shanghai and Shenzhen security market by scientific and objective method, aiming to comprehensively reflect the price fluctuation and performance of the small-cap companies in Shanghai and Shenzhen securities market.

### 2. Motivation and Feasibility

The trading of stock index futures differs from that of stocks in an important aspect:

Stock index futures can be sold **short**, either buying first or selling first. So, the trading of stock index futures is a two-way trade. However, there is no short selling mechanism in the stock markets of some countries, where stocks can only be sold after being bought, which is called one-way trade.

China’s stock lacks short-selling mechanism, and the stock index future contract price is highly related to stock price. Based on the above points, the stock index futures are picked to build the **arbitrage portfolio**. 

The access to **large amount of data** in combination with the rapid development of software and hardware technology enables the training of large and complex models. In particular, deep learning algorithms show promising results in improving the prediction accuracy of regression and classification problems encountered in other areas of science, for example, image processing, speech recognition and medical drug discovery, see Krizhevsky et al. (2012).

## Workflow

### 3. Strategy

Because the market capitalization is quite different between these two indexes, the return spread of these two assets could be detected significantly through both linear model and non-linear model. The price information of stock index futures and stock index itself is highly related. 

Using the stock index future as the investing underlying is more pragmatic especially in the environment of lacking short mechanism in China. We must short the loser and long the winner to capture the spread and finally earn profit. 

### 4. Methodology

The workflow has the following three steps:

- `Predicting`: Use Machine Learning Model (XGBoost or LightGBM) to predict daily return rank of SSE 50 index futures and CSI 500 index futures. 
- `Trading`: Use long/short strategy to get the spread difference.
- `Backtesting`: Compute the sharpe ratio of the strategy based on back testing data.

## Data

### 5. Data's Properties

Data are mainly used in `Predicting` and `Backtesting`. Data are in the same formation, though the specific data are totally different in these two steps.

In `Predicting`, the basic logic is using the **historical data** containing hundreds of features to predict the daily return rank of next period.

The historical daily data in `3NF` can be shown as follow:

| Daily_Return_Rank  | Factor_1      | Factor_2      | ...  | Factor_N      |
| :----------------: |:-------------:| :------------:| :---:| :------------:|
| 1                  | X<sub>1</sub> | X<sub>2</sub> | ...  | X<sub>N</sub> |
| 2                  | Y<sub>1</sub> | Y<sub>2</sub> | ...  | Y<sub>N</sub> |

The factors can be generated from the weighted-average `alpha101` factor in each constituent stock. The factors can also be generated from sentiment analysis.

$$N = 101$$

The total amount of data should be:

$$2(futures)×101(factors)×250(days/year)×3(years) = 151500$$

The case meets the 3Vs' properties.

- `Volume` : If the three-years data are stored as csv file, the size of the whole file is about 4G. We can use more years and more factors to enhance the predicting accuracy and the volume of the data must be more.
- `Velocity` : For sake of transaction fee, the strategy is based on daily data. Theoretically, however, this strategy could be use the intra-daily data in the context of high-frequency trade.
- `Variety` : The  factors not only are focus on the historical price or volume information but also contain more types of information such as market sentiment factors.

### 6. Database Selection

For the label and features of each instances are well structured in `3NF`, the most suitable database is `SQL`.

In additional, the SQL database has one more advantage is that data format of SQL is similar to that of `Pandas` in Python. 

When using these data as input of Machine Learning Methods, it is convenient to compose them in `DataFrame` which is a common data type in `Pandas`.