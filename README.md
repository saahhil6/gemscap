Overview
This project is a real-time trading analytics application designed to analyze the relationship between two correlated assets and identify statistical arbitrage opportunities. It ingests live market data, processes it across multiple timeframes, computes key quantitative metrics, and presents actionable insights through an interactive dashboard.


Features
	•	Live data ingestion from Binance WebSocket streams
	•	Resampling of tick data into 1s, 1m, and 5m intervals
	•	Hedge ratio estimation using OLS regression
	•	Spread and rolling Z-score computation
	•	Rolling correlation and ADF test for cointegration
	•	Real-time interactive visualizations (prices, spread, z-score, volume)
	•	Rule-based alerts (e.g., high Z-score, low correlation)
	•	Export of processed asset data for offline analysis


System Workflow
	1.	Live tick data is streamed from Binance using WebSockets.
	2.	Incoming data is aggregated and resampled based on user-selected timeframes.
	3.	Quantitative analytics such as regression, spread, Z-score, correlation, and ADF tests are computed once sufficient data points are available.
	4.	Results are pushed to the frontend for near real-time visualization and alerting.
	5.	Users can export processed datasets for further research or backtesting.

 
  Technologies Used
	•	Python (backend analytics and data processing)
	•	WebSockets (real-time data ingestion)
	•	Plotly / Streamlit (interactive visualization)
	•	NumPy, Pandas, Statsmodels (quantitative analytics)
