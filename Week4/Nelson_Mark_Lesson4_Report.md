# Nelson Mark – Lesson 4 Report  
**Course:** [Your course name]  
**Lesson:** 4 – Multivariate OHLCV Cleaning and EDA

---

## 1. Title and Identification

**Multivariate OHLCV Time Series Cleaning and Exploratory Data Analysis**

Mark Nelson  
[Course name]  
Lesson 4

---

## 2. Dataset Description

The dataset consists of multivariate OHLCV (open, high, low, close, volume) 5-minute candle data for seven crypto assets: ADAUSD, BNBUSDT, BTCUSD, DOGEUSDT, ETHUSD, SOLUSDT, and XRPUSD. Each asset is a univariate time series of 5-minute bars; after merging and alignment they form a single multivariate time series keyed by a common timestamp. The raw inputs are CSV files (one per asset) that were combined and cleaned for joint analysis.

---

## 3. Data Preprocessing

**Merge approach.** The seven asset series were merged with an outer join on timestamp so that every bar present in any asset is represented. The result was then reindexed to a full 5-minute calendar grid over the observed range, so that missing bars appear as missing values rather than omitted rows. This yields a regular time index and a rectangular table suitable for multivariate analysis.

**Missing values.** Missingness was detected from the reindexed grid: any cell in an OHLCV column that was absent after the merge is treated as missing. Boolean “was missing” flags were added for each OHLCV series so that filled values can be distinguished from originally observed ones. OHLC series were filled using forward fill then backward fill so that gaps are bridged from the last and next known price. Volume missing values were set to zero, under the assumption that no trade corresponds to zero volume. After filling, the dataset has no remaining missing cells in the OHLCV columns (missing cells after fill equals zero).

**Normalization.** Each asset’s close series was normalized to start at 1.0 by dividing by the first valid close. This does not change the return series but makes it easier to compare price paths across assets with different scales on a single chart.

**Outlier handling.** Log returns were computed from the close series. Outliers were identified using a z-score threshold of 5.0: returns with absolute z-score above 5 were flagged as outliers. These flags were stored in dedicated columns; no observations were removed. The analysis can therefore distinguish extreme returns without dropping them.

---

## 4. Summary Statistics Observations

The cleaned dataset has 4,608 rows and 52 columns, covering the period from 2023-08-10 00:00:00 UTC to 2023-08-25 23:55:00 UTC. Before filling, 690 rows had at least one missing OHLCV value, and there were 3,482 missing cells in total across the OHLCV columns. After the described fill steps, missing cells in those columns are zero.

Per-asset close statistics illustrate the different price levels and volatility. For example, BTCUSD’s close ranges from roughly 23,733 to 29,695 over the window; ETHUSD from about 1,471 to 1,863. Other assets (e.g., ADAUSD, DOGEUSDT, SOLUSDT, XRPUSD) show their own ranges and scales, which is why the normalized-close plot is useful for comparing paths. Return statistics (mean, standard deviation, min, max) were computed per asset from the log-return series; means are near zero and standard deviations differ by asset. Outlier counts from the z-score flags vary by symbol: for instance ADAUSD had 10 flagged return outliers, BNBUSDT 16, BTCUSD 4, DOGEUSDT 9, and ETHUSD 5 in this run. The remaining symbols’ counts can be read from the pipeline log or summary JSON if needed.

---

## 5. Visualization Insights

**Close trends.** The raw close plot shows each asset’s price path over the two-week window; levels differ widely (e.g., BTC in the tens of thousands vs. DOGE in cents). The normalized-close plot, where each series starts at 1.0, makes it easier to compare relative movement: which assets gained or lost more relative to their own starting point, and how they moved together or diverged over time. Over this window one can observe overall direction and relative strength without being dominated by scale.

![Close Trends](outputs/close_trends.png)

*Figure 1. Closing price trends across all cryptocurrencies showing price dispersion and volatility clustering.*

![Normalized Close Trends](outputs/close_trends_normalized.png)

*Figure 2. Normalized closing price series rebased to 1.0, enabling cross-asset comparative growth analysis.*

**Correlation heatmap.** The return correlation heatmap shows pairwise correlations between the seven assets’ log-return series. In typical crypto data, BTC and ETH often move together and show positive correlation; other pairs can be positive, negative, or weak depending on the period. The heatmap helps identify which assets tend to move in the same or opposite direction over this short window, without over-interpreting exact coefficient values from a single sample.

![Returns Correlation Heatmap](outputs/returns_corr_heatmap.png)

*Figure 3. Correlation heatmap of log returns illustrating clustering behavior among large-cap assets.*

**Return distributions.** Histograms and boxplots of log returns show that most assets have returns centered near zero with symmetric or slightly heavy tails. Some assets show more extreme positive or negative returns (heavier tails) and possibly a few visible outliers, consistent with the flagged outlier counts. The boxplots summarize central tendency, spread, and extremes in a compact way.

![ADAUSD Return Distribution](outputs/hist_ADAUSD_returns.png)

*Figure 4. Histogram of ADAUSD log returns showing heavy tails and skewness.*

![BNBUSDT Return Distribution](outputs/hist_BNBUSDT_returns.png)

*Figure 5. Histogram of BNBUSDT log returns showing heavy tails and skewness.*

![BTCUSD Return Distribution](outputs/hist_BTCUSD_returns.png)

*Figure 6. Histogram of BTCUSD log returns showing heavy tails and skewness.*

![DOGEUSDT Return Distribution](outputs/hist_DOGEUSDT_returns.png)

*Figure 7. Histogram of DOGEUSDT log returns showing heavy tails and skewness.*

![ETHUSD Return Distribution](outputs/hist_ETHUSD_returns.png)

*Figure 8. Histogram of ETHUSD log returns showing heavy tails and skewness.*

![SOLUSDT Return Distribution](outputs/hist_SOLUSDT_returns.png)

*Figure 9. Histogram of SOLUSDT log returns showing heavy tails and skewness.*

![XRPUSD Return Distribution](outputs/hist_XRPUSD_returns.png)

*Figure 10. Histogram of XRPUSD log returns showing heavy tails and skewness.*

![ADAUSD Return Boxplot](outputs/box_ADAUSD_returns.png)

*Figure 11. Boxplot of ADAUSD log returns showing quartiles, median, and extreme values.*

![BNBUSDT Return Boxplot](outputs/box_BNBUSDT_returns.png)

*Figure 12. Boxplot of BNBUSDT log returns showing quartiles, median, and extreme values.*

![BTCUSD Return Boxplot](outputs/box_BTCUSD_returns.png)

*Figure 13. Boxplot of BTCUSD log returns showing quartiles, median, and extreme values.*

![DOGEUSDT Return Boxplot](outputs/box_DOGEUSDT_returns.png)

*Figure 14. Boxplot of DOGEUSDT log returns showing quartiles, median, and extreme values.*

![ETHUSD Return Boxplot](outputs/box_ETHUSD_returns.png)

*Figure 15. Boxplot of ETHUSD log returns showing quartiles, median, and extreme values.*

![SOLUSDT Return Boxplot](outputs/box_SOLUSDT_returns.png)

*Figure 16. Boxplot of SOLUSDT log returns showing quartiles, median, and extreme values.*

![XRPUSD Return Boxplot](outputs/box_XRPUSD_returns.png)

*Figure 17. Boxplot of XRPUSD log returns showing quartiles, median, and extreme values.*

---

## 6. Limitations and Notes

- **Filling.** Forward and backward filling can smooth over gaps and may understate volatility or autocorrelation in thinly traded periods. Volume filled with zero is an assumption that may not hold if “missing” sometimes indicates unreported data rather than no trading.

- **Outliers.** Outliers were only flagged, not removed. Downstream models or statistics that are sensitive to extremes may still be affected unless the user explicitly filters or winsorizes using the flags.

- **Short window.** The analysis covers about two weeks (August 10–25, 2023). Conclusions about correlations, trends, or tail behavior are specific to this window and may not generalize to other periods.

---

## 7. Conclusion

This lesson illustrated a full workflow for cleaning and exploring multivariate OHLCV time series: merging multiple assets on a common timestamp, aligning to a full 5-minute grid, handling missing values with explicit flags and fill rules, normalizing closes for comparison, and flagging return outliers without deletion. The resulting dataset (4,608 rows, 52 columns) and the EDA outputs (summary statistics, close trends, return distributions, and a correlation heatmap) support a clear view of the seven assets over the given window. The main takeaways are the importance of a consistent time index, transparent missing-value handling, and the use of both raw and normalized views plus return-based diagnostics when working with multivariate financial time series.

---

## 8. Appendix: Output Files

The following files were produced in the `outputs/` directory:

- close_trends.png  
- close_trends_normalized.png  
- returns_corr_heatmap.png  
- hist_ADAUSD_returns.png  
- hist_BNBUSDT_returns.png  
- hist_BTCUSD_returns.png  
- hist_DOGEUSDT_returns.png  
- hist_ETHUSD_returns.png  
- hist_SOLUSDT_returns.png  
- hist_XRPUSD_returns.png  
- box_ADAUSD_returns.png  
- box_BNBUSDT_returns.png  
- box_BTCUSD_returns.png  
- box_DOGEUSDT_returns.png  
- box_ETHUSD_returns.png  
- box_SOLUSDT_returns.png  
- box_XRPUSD_returns.png  
- summary.json  
