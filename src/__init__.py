"""
Pairs Trading Strategy - Moduli principali
"""

from .data_fetcher import DataFetcher
from .pairs_strategy import PairsStrategy
from .backtester import Backtester

# IBConnector importato separatamente per evitare problemi con asyncio
# from .ib_connector import IBConnector

__all__ = ["DataFetcher", "PairsStrategy", "Backtester"]
