# backend/data/resampler.py
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class TickResampler:
    """
    Convert tick data to OHLCV bars
    
    Design:
    - Streaming resampler (updates as ticks arrive)
    - Multiple timeframes simultaneously
    - Handles irregular tick intervals
    """
    
    def __init__(self, timeframes: List[str] = ['1S', '1min', '5min']):
        self.timeframes = timeframes
        # In-progress bars keyed by bar start timestamp
        self.current_bars: Dict[datetime, Dict] = {}
        
    def resample_ticks(self, ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample tick data to OHLCV
            
        Args:
            ticks: DataFrame with columns [timestamp, price, quantity]
            timeframe: Pandas frequency string ('1S', '1min', '5min')
            
        Returns:
            OHLCV DataFrame
        """
        if ticks.empty:
            return pd.DataFrame()
        
        # Ensure timestamp index
        if 'timestamp' in ticks.columns:
            ticks = ticks.set_index('timestamp')
            # Ensure index is datetime-like
            ticks.index = pd.to_datetime(ticks.index)
        
        # Resample price to OHLC
        ohlc = ticks['price'].resample(timeframe).ohlc()
        
        # Volume = sum of quantities
        volume = ticks['quantity'].resample(timeframe).sum()
        
        # Trade count
        trade_count = ticks['price'].resample(timeframe).count()
        
        # Combine
        bars = pd.DataFrame({
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': volume,
            'trade_count': trade_count
        })
        
        # Drop incomplete bars (last bar might be partial)
        bars = bars.dropna()
        
        return bars
    
    def update_streaming_bar(self, tick: Dict, timeframe: str) -> Optional[Dict]:
        """
        Update current bar with new tick, emit when complete
        
        Returns:
            Completed bar dict if period ended, else None
        """
        ts = tick['timestamp']
        price = tick['price']
        qty = tick['quantity']
        
        # Determine bar period
        bar_key = self._get_bar_key(ts, timeframe)
        
        if bar_key not in self.current_bars:
            # Start new bar
            self.current_bars[bar_key] = {
                'timestamp': bar_key,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': qty,
                'trade_count': 1
            }
            return None
        
        # Update existing bar
        bar = self.current_bars[bar_key]
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += qty
        bar['trade_count'] += 1
        
        # Check if bar period has ended
        current_bar_key = self._get_bar_key(datetime.now(), timeframe)
        if current_bar_key != bar_key:
            # Period ended, emit completed bar
            completed_bar = self.current_bars.pop(bar_key)
            return completed_bar
        
        return None
    
    def _get_bar_key(self, timestamp: datetime, timeframe: str) -> datetime:
        """Round timestamp to bar period start"""
        if timeframe == '1S':
            return timestamp.replace(microsecond=0)
        elif timeframe == '1min':
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == '5min':
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

