# src/trading/__init__.py
"""
Fenix Trading Module.

This module contains the refactored core of the trading system,
organized in a modular and maintainable way.

Main components:
- TradingEngine: Main trading engine
- OrderExecutor: Order executor to Binance
- MarketDataManager: Real-time market data management
- PositionManager: Open positions management
"""

from .engine import TradingEngine
from .executor import OrderExecutor
from .market_data import MarketDataManager

__all__ = [
    "TradingEngine",
    "OrderExecutor",
    "MarketDataManager",
]
