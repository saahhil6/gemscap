import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from typing import Dict, Tuple, Optional
from sklearn.linear_model import HuberRegressor, TheilSenRegressor

from utils.logger import setup_logger

logger = setup_logger("analytics")

class PairsAnalytics:
    """
    Statistical arbitrage analytics for pairs trading
    
    Core Metrics:
    - Hedge Ratio: OLS regression coefficient
    - Spread: Price differential adjusted by hedge ratio
    - Z-Score: Standardized spread for mean reversion signals
    - ADF Test: Stationarity test
    - Rolling Correlation: Relationship strength
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        logger.info(f"PairsAnalytics initialized with window={window}")
    
    def calculate_hedge_ratio(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        method: str = 'ols'
    ) -> Tuple[float, float, float]:
        """
        Calculate optimal hedge ratio via regression
        
        Model: asset1 = beta * asset2 + alpha + error
        
        Args:
            asset1: Dependent variable (Y)
            asset2: Independent variable (X)
            method: 'ols', 'huber', or 'theil_sen'
        
        Returns:
            (beta, alpha, r_squared)
        """
        # Align series and drop NaN
        df = pd.DataFrame({'y': asset1, 'x': asset2}).dropna()
        
        if len(df) < 2:
            logger.warning("Insufficient data for hedge ratio calculation")
            return 0.0, 0.0, 0.0
        
        X = df['x'].values.reshape(-1, 1)
        y = df['y'].values
        
        try:
            if method == 'ols':
                # Ordinary Least Squares
                X_with_const = sm.add_constant(X)
                model = OLS(y, X_with_const).fit()
                alpha = float(model.params[0])
                beta = float(model.params[1])
                r_squared = float(model.rsquared)
                
            elif method == 'huber':
                # Robust regression (Huber loss - less sensitive to outliers)
                model = HuberRegressor()
                model.fit(X, y)
                beta = float(model.coef_[0])
                alpha = float(model.intercept_)
                
                # Calculate R² manually
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
                
            elif method == 'theil_sen':
                # Theil-Sen estimator (median-based, very robust)
                model = TheilSenRegressor(random_state=42)
                model.fit(X, y)
                beta = float(model.coef_[0])
                alpha = float(model.intercept_)
                
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
            else:
                raise ValueError(f"Unknown regression method: {method}")
            
            logger.debug(f"Hedge ratio: β={beta:.4f}, α={alpha:.4f}, R²={r_squared:.4f}")
            return beta, alpha, r_squared
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate spread = asset1 - beta * asset2
        
        Interpretation:
        - Positive spread: asset1 is relatively expensive
        - Negative spread: asset1 is relatively cheap
        """
        return asset1 - hedge_ratio * asset2
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling Z-score of spread
        
        Z = (spread - rolling_mean) / rolling_std
        
        Trading Signal:
        - Z > 2: Spread is 2 std above mean → SHORT asset1, LONG asset2
        - Z < -2: Spread is 2 std below mean → LONG asset1, SHORT asset2
        - |Z| < 1: Neutral zone
        """
        if window is None:
            window = self.window
        
        rolling_mean = spread.rolling(window=window, min_periods=1).mean()
        rolling_std = spread.rolling(window=window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def rolling_correlation(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling Pearson correlation
        
        Interpretation:
        - corr > 0.7: Strong positive relationship
        - 0.3 < corr < 0.7: Moderate relationship
        - corr < 0.3: Weak relationship
        """
        if window is None:
            window = self.window
        
        return asset1.rolling(window=window, min_periods=1).corr(asset2)
    
    def adf_test(self, spread: pd.Series) -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Null Hypothesis: Spread has unit root (non-stationary/trending)
        Alternative: Spread is stationary (mean-reverting)
        
        Returns:
            Dict with test results
        """
        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 10:
                return {
                    'error': 'Insufficient data for ADF test',
                    'is_stationary': False
                }
            
            result = adfuller(spread_clean, autolag='AIC')
            
            is_stationary = result[1] < 0.05
            
            return {
                'statistic': float(result[0]),
                'pvalue': float(result[1]),
                'used_lag': int(result[2]),
                'n_obs': int(result[3]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'is_stationary': is_stationary,
                'interpretation': (
                    'Spread is stationary (mean-reverting) - Good for pairs trading' 
                    if is_stationary
                    else 'Spread is non-stationary (trending) - Not suitable for pairs trading'
                )
            }
        except Exception as e:
            logger.error(f"ADF test error: {e}")
            return {
                'error': str(e),
                'is_stationary': False
            }
    
    def analyze_pair(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        regression_method: str = 'ols',
        run_adf: bool = True
    ) -> Dict:
        """
        Complete pairs analysis pipeline
        
        Args:
            df1, df2: DataFrames with 'close' column and datetime index
            regression_method: 'ols', 'huber', 'theil_sen'
            run_adf: Whether to run ADF test (computationally expensive)
        
        Returns:
            Comprehensive analytics dictionary
        """
        try:
            # Align dataframes by timestamp
            df = pd.DataFrame({
                'asset1': df1['close'],
                'asset2': df2['close']
            }).dropna()
            
            if len(df) < self.window:
                return {
                    'error': f'Insufficient data. Need at least {self.window} data points, got {len(df)}',
                    'data_points': len(df)
                }
            
            # Calculate hedge ratio
            beta, alpha, r2 = self.calculate_hedge_ratio(
                df['asset1'],
                df['asset2'],
                method=regression_method
            )
            
            # Calculate spread
            spread = self.calculate_spread(df['asset1'], df['asset2'], beta)
            
            # Calculate z-score
            zscore = self.calculate_zscore(spread)
            
            # Rolling correlation
            correlation = self.rolling_correlation(df['asset1'], df['asset2'])
            
            # Current values
            current_zscore = float(zscore.iloc[-1]) if not zscore.empty and not pd.isna(zscore.iloc[-1]) else 0.0
            current_correlation = float(correlation.iloc[-1]) if not correlation.empty and not pd.isna(correlation.iloc[-1]) else 0.0
            current_spread = float(spread.iloc[-1]) if not spread.empty else 0.0
            
            # ADF test (optional, can be slow)
            adf_result = self.adf_test(spread) if run_adf else {'skipped': True}
            
            # Generate trading signal
            signal = self._generate_signal(current_zscore, current_correlation)
            
            # Return last 100 data points for plotting
            n_points = min(100, len(df))
            
            result = {
                'hedge_ratio': {
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r2,
                    'method': regression_method
                },
                'current_values': {
                    'zscore': current_zscore,
                    'correlation': current_correlation,
                    'spread': current_spread,
                    'asset1_price': float(df['asset1'].iloc[-1]),
                    'asset2_price': float(df['asset2'].iloc[-1])
                },
                'time_series': {
                    'timestamps': [ts.isoformat() for ts in df.index[-n_points:]],
                    'spread': spread.iloc[-n_points:].tolist(),
                    'zscore': zscore.iloc[-n_points:].tolist(),
                    'correlation': correlation.iloc[-n_points:].tolist(),
                    'asset1_prices': df['asset1'].iloc[-n_points:].tolist(),
                    'asset2_prices': df['asset2'].iloc[-n_points:].tolist()
                },
                'adf_test': adf_result,
                'signal': signal,
                'data_points': len(df),
                'window_size': self.window
            }
            
            logger.info(f"Pairs analysis complete: Z={current_zscore:.2f}, Signal={signal['action']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in pairs analysis: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _generate_signal(self, zscore: float, correlation: float) -> Dict:
        """
        Generate trading signal from z-score and correlation
        
        Returns:
            Dict with action, strength, and description
        """
        # Check correlation quality - if correlation is too low we avoid trading
        if correlation < 0.5:
            return {
                'action': 'NO_TRADE',
                'strength': 0,
                'description': f'Correlation too low ({correlation:.2f}). Pairs have diverged.',
                'color': 'gray'
            }

        # Z-score based signals
        if zscore > 2:
            strength = min(100, (zscore - 2) * 25)
            return {
                'action': 'SHORT',
                'strength': int(strength),
                'description': f'Spread overextended (+{zscore:.2f}σ). Short asset1, long asset2.',
                'color': 'red'
            }
        elif zscore < -2:
            strength = min(100, (abs(zscore) - 2) * 25)
            return {
                'action': 'LONG',
                'strength': int(strength),
                'description': f'Spread underextended ({zscore:.2f}σ). Long asset1, short asset2.',
                'color': 'green'
            }
        elif abs(zscore) < 0.5:
            return {
                'action': 'EXIT',
                'strength': 0,
                'description': 'Spread near mean. Close positions for profit.',
                'color': 'blue'
            }
        else:
            return {
                'action': 'HOLD',
                'strength': 0,
                'description': f'Z-score at {zscore:.2f}. Wait for clearer signal.',
                'color': 'yellow'
            }

    def calculate_volatility(self, prices: pd.Series, window: Optional[int] = None) -> pd.Series:
        """Calculate rolling volatility (standard deviation of returns, annualized)

        Returns a Series of annualized volatility values.
        """
        if window is None:
            window = self.window

        returns = prices.pct_change()
        vol = returns.rolling(window=window, min_periods=1).std()
        return vol * np.sqrt(252)

    def calculate_sharpe_ratio(self, returns: pd.Series, window: Optional[int] = None,
                               risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate rolling Sharpe ratio (annualized)

        Expects `returns` to be period returns (e.g., daily). Returns a series of
        rolling Sharpe values; handles division-by-zero safely by producing NaN.
        """
        if window is None:
            window = self.window

        mean_return = returns.rolling(window=window, min_periods=1).mean() * 252
        std_return = returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        std_return = std_return.replace(0, np.nan)

        return (mean_return - risk_free_rate) / std_return
