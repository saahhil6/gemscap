# backend/data/ingestion.py
import asyncio
import json
import websockets
from typing import Dict, List, Callable
from datetime import datetime
from collections import deque

class BinanceIngester:
    """
    Real-time tick ingestion from Binance WebSocket
    
    Design Decisions:
    - Async/await for non-blocking I/O
    - Per-symbol connections (Binance allows combined streams)
    - Ring buffer for tick batching
    - Graceful reconnection with exponential backoff
    """
    
    def __init__(self, symbols: List[str], on_tick_callback: Callable):
        self.symbols = [s.lower() for s in symbols]
        self.callback = on_tick_callback
        self.buffers = {sym: deque(maxlen=1000) for sym in self.symbols}
        self.ws_url = self._build_url()
        
    def _build_url(self) -> str:
        # Binance combined stream format
        streams = "/".join([f"{sym}@trade" for sym in self.symbols])
        return f"wss://stream.binance.com:9443/stream?streams={streams}"
    
    async def connect(self):
        """Maintain persistent connection with auto-reconnect"""
        retry_delay = 1
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print(f"✓ Connected to Binance: {self.symbols}")
                    retry_delay = 1  # Reset on success
                    
                    async for message in ws:
                        await self._process_message(message)
                        
            except Exception as e:
                print(f"✗ Connection error: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Exponential backoff
    
    async def _process_message(self, raw_message: str):
        """Parse and buffer incoming ticks"""
        data = json.loads(raw_message)
        
        if 'data' not in data:
            return
            
        tick_data = data['data']
        symbol = tick_data['s'].lower()
        
        tick = {
            'timestamp': datetime.fromtimestamp(tick_data['T'] / 1000),
            'symbol': symbol,
            'price': float(tick_data['p']),
            'quantity': float(tick_data['q']),
            'is_buyer_maker': tick_data['m']
        }
        
        # Buffer for batch processing
        self.buffers[symbol].append(tick)
        
        # Callback for real-time updates
        await self.callback(tick)
    
    def get_buffered_ticks(self, symbol: str) -> List[Dict]:
        """Retrieve buffered ticks for batch insertion"""
        ticks = list(self.buffers[symbol])
        self.buffers[symbol].clear()
        return ticks