"""
Modulo per il recupero dei dati storici delle azioni
Utilizza yfinance per dati gratuiti e Interactive Brokers per dati real-time
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


class DataFetcher:
    """
    Classe per il recupero e la gestione dei dati storici delle azioni
    """
    
    def __init__(self, data_folder: str = None):
        """
        Inizializza il DataFetcher
        
        Args:
            data_folder: Cartella per salvare i dati in cache
        """
        self.data_folder = data_folder or DATA_CONFIG["data_folder"]
        os.makedirs(self.data_folder, exist_ok=True)
        
    def fetch_stock_data(
        self, 
        symbol: str, 
        start_date: datetime = None, 
        end_date: datetime = None,
        years: int = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Scarica i dati storici di un'azione
        
        Args:
            symbol: Ticker dell'azione (es. "AAPL")
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni di dati (alternativo a start_date)
            use_cache: Se True, utilizza dati in cache se disponibili
            
        Returns:
            DataFrame con colonne: Open, High, Low, Close, Volume, Adj Close
        """
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            if years is None:
                years = DATA_CONFIG["default_years"]
            start_date = end_date - timedelta(days=years * 365)
        
        cache_file = os.path.join(
            self.data_folder, 
            f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        
        # Prova a caricare dalla cache
        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        # Scarica da yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"Nessun dato trovato per {symbol}")
            
            # Salva in cache
            df.to_csv(cache_file)
            
            return df
            
        except Exception as e:
            raise Exception(f"Errore nel recupero dati per {symbol}: {str(e)}")
    
    def fetch_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scarica i dati per una coppia di azioni
        
        Args:
            symbol1: Ticker della prima azione
            symbol2: Ticker della seconda azione
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni di dati
            use_cache: Se True, utilizza dati in cache
            
        Returns:
            Tupla di DataFrame per le due azioni
        """
        df1 = self.fetch_stock_data(symbol1, start_date, end_date, years, use_cache)
        df2 = self.fetch_stock_data(symbol2, start_date, end_date, years, use_cache)
        
        # Allinea le date
        common_dates = df1.index.intersection(df2.index)
        df1 = df1.loc[common_dates]
        df2 = df2.loc[common_dates]
        
        return df1, df2
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Scarica i dati per multiple azioni
        
        Args:
            symbols: Lista di ticker
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni di dati
            use_cache: Se True, utilizza dati in cache
            
        Returns:
            Dizionario {symbol: DataFrame}
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_stock_data(
                    symbol, start_date, end_date, years, use_cache
                )
            except Exception as e:
                print(f"Warning: Impossibile scaricare dati per {symbol}: {e}")
        
        return data
    
    def get_aligned_prices(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Ottiene i prezzi allineati per una coppia di azioni
        
        Args:
            symbol1: Ticker della prima azione
            symbol2: Ticker della seconda azione
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni
            price_col: Colonna da utilizzare (default: "Close")
            
        Returns:
            DataFrame con colonne per entrambe le azioni
        """
        df1, df2 = self.fetch_pair_data(symbol1, symbol2, start_date, end_date, years)
        
        prices = pd.DataFrame({
            symbol1: df1[price_col],
            symbol2: df2[price_col]
        })
        
        return prices.dropna()
    
    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None,
        window: int = None
    ) -> pd.DataFrame:
        """
        Calcola la correlazione tra due azioni
        
        Args:
            symbol1: Ticker della prima azione
            symbol2: Ticker della seconda azione
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni
            window: Finestra per correlazione rolling (None per correlazione totale)
            
        Returns:
            DataFrame con correlazione (rolling se window specificato)
        """
        prices = self.get_aligned_prices(symbol1, symbol2, start_date, end_date, years)
        
        # Calcola i rendimenti
        returns = prices.pct_change().dropna()
        
        if window is None:
            # Correlazione totale
            corr = returns[symbol1].corr(returns[symbol2])
            return pd.DataFrame({"correlation": [corr]}, index=["total"])
        else:
            # Correlazione rolling
            rolling_corr = returns[symbol1].rolling(window=window).corr(returns[symbol2])
            return pd.DataFrame({"correlation": rolling_corr})
    
    def get_benchmark_data(
        self,
        benchmark: str = "SPY",
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None
    ) -> pd.DataFrame:
        """
        Scarica i dati di un benchmark (es. SPY per S&P 500)
        
        Args:
            benchmark: Ticker del benchmark
            start_date: Data di inizio
            end_date: Data di fine
            years: Numero di anni
            
        Returns:
            DataFrame con i dati del benchmark
        """
        return self.fetch_stock_data(benchmark, start_date, end_date, years)


def test_data_fetcher():
    """Test del DataFetcher"""
    fetcher = DataFetcher()
    
    # Test singola azione
    print("Scaricando dati AAPL...")
    aapl = fetcher.fetch_stock_data("AAPL", years=2)
    print(f"AAPL: {len(aapl)} righe")
    print(aapl.tail())
    
    # Test coppia
    print("\nScaricando dati KO-PEP...")
    prices = fetcher.get_aligned_prices("KO", "PEP", years=2)
    print(f"Coppia KO-PEP: {len(prices)} righe")
    print(prices.tail())
    
    # Test correlazione
    print("\nCalcolando correlazione...")
    corr = fetcher.calculate_correlation("KO", "PEP", years=2)
    print(f"Correlazione totale: {corr['correlation'].values[0]:.4f}")


if __name__ == "__main__":
    test_data_fetcher()
