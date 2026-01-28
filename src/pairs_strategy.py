"""
Modulo per la strategia di Pairs Trading
Implementa la logica di mean reversion per coppie di azioni correlate
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGY_CONFIG


class PositionType(Enum):
    """Tipo di posizione nel pairs trade"""
    NONE = "none"
    LONG_FIRST_SHORT_SECOND = "long_first_short_second"  # Long symbol1, Short symbol2
    SHORT_FIRST_LONG_SECOND = "short_first_long_second"  # Short symbol1, Long symbol2


@dataclass
class TradeSignal:
    """Segnale di trading generato dalla strategia"""
    timestamp: datetime
    signal_type: str  # "OPEN", "CLOSE", "STOP_LOSS"
    position_type: PositionType
    zscore: float
    spread: float
    symbol1_price: float
    symbol2_price: float
    hedge_ratio: float


@dataclass
class Position:
    """Posizione aperta"""
    entry_date: datetime
    position_type: PositionType
    entry_zscore: float
    entry_spread: float
    symbol1_quantity: int
    symbol2_quantity: int
    symbol1_entry_price: float
    symbol2_entry_price: float


class PairsStrategy:
    """
    Strategia di Pairs Trading basata su Z-score dello spread
    
    La strategia:
    1. Calcola lo spread tra due azioni usando un hedge ratio (dalla regressione)
    2. Normalizza lo spread con Z-score
    3. Apre posizioni quando lo Z-score supera la soglia di entry
    4. Chiude quando torna verso la media (soglia exit)
    5. Stop loss se divergenza continua oltre il limite
    """
    
    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        entry_zscore: float = None,
        exit_zscore: float = None,
        stop_loss_zscore: float = None,
        lookback_period: int = None
    ):
        """
        Inizializza la strategia
        
        Args:
            symbol1: Ticker della prima azione
            symbol2: Ticker della seconda azione
            entry_zscore: Soglia Z-score per aprire posizioni
            exit_zscore: Soglia Z-score per chiudere posizioni
            stop_loss_zscore: Soglia Z-score per stop loss
            lookback_period: Finestra per calcolo media/std
        """
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        
        # Parametri strategia
        self.entry_zscore = entry_zscore or STRATEGY_CONFIG["entry_zscore"]
        self.exit_zscore = exit_zscore or STRATEGY_CONFIG["exit_zscore"]
        self.stop_loss_zscore = stop_loss_zscore or STRATEGY_CONFIG["stop_loss_zscore"]
        self.lookback_period = lookback_period or STRATEGY_CONFIG["lookback_period"]
        
        # Stato
        self.current_position: Optional[Position] = None
        self.hedge_ratio = 1.0
        self.spread_mean = 0.0
        self.spread_std = 1.0
        
        # Dati storici
        self.prices_df: Optional[pd.DataFrame] = None
        self.spread_series: Optional[pd.Series] = None
        self.zscore_series: Optional[pd.Series] = None
    
    def calculate_hedge_ratio(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """
        Calcola l'hedge ratio usando la regressione lineare (OLS)
        
        L'hedge ratio indica quante unità di symbol2 servono per ogni unità di symbol1
        per creare uno spread stazionario.
        
        Args:
            prices1: Serie prezzi prima azione
            prices2: Serie prezzi seconda azione
            
        Returns:
            Hedge ratio
        """
        # Regressione: prices1 = beta * prices2 + alpha + epsilon
        # Usiamo numpy per semplicità
        X = prices2.values.reshape(-1, 1)
        y = prices1.values
        
        # Aggiungi intercetta
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            return beta[1]  # Coefficiente di prices2
        except:
            return 1.0
    
    def calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: float = None
    ) -> pd.Series:
        """
        Calcola lo spread tra le due azioni
        
        Spread = prices1 - hedge_ratio * prices2
        
        Args:
            prices1: Serie prezzi prima azione
            prices2: Serie prezzi seconda azione
            hedge_ratio: Rapporto di copertura (se None, calcolato automaticamente)
            
        Returns:
            Serie dello spread
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
        
        self.hedge_ratio = hedge_ratio
        return prices1 - hedge_ratio * prices2
    
    def calculate_zscore(self, spread: pd.Series, lookback: int = None) -> pd.Series:
        """
        Calcola lo Z-score dello spread usando una finestra rolling
        
        Z-score = (spread - media_rolling) / std_rolling
        
        Args:
            spread: Serie dello spread
            lookback: Finestra per calcolo rolling
            
        Returns:
            Serie Z-score
        """
        if lookback is None:
            lookback = self.lookback_period
        
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()
        
        # Evita divisione per zero
        spread_std = spread_std.replace(0, np.nan)
        
        zscore = (spread - spread_mean) / spread_std
        
        # Salva per uso successivo
        self.spread_mean = spread_mean.iloc[-1] if len(spread_mean) > 0 else 0
        self.spread_std = spread_std.iloc[-1] if len(spread_std) > 0 else 1
        
        return zscore
    
    def fit(self, prices_df: pd.DataFrame):
        """
        Addestra la strategia sui dati storici
        
        Args:
            prices_df: DataFrame con colonne [symbol1, symbol2]
        """
        self.prices_df = prices_df
        
        prices1 = prices_df[self.symbol1]
        prices2 = prices_df[self.symbol2]
        
        # Calcola hedge ratio su tutto il periodo
        self.hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
        
        # Calcola spread e z-score
        self.spread_series = self.calculate_spread(prices1, prices2, self.hedge_ratio)
        self.zscore_series = self.calculate_zscore(self.spread_series)
    
    def generate_signal(
        self,
        current_zscore: float,
        price1: float,
        price2: float,
        timestamp: datetime = None
    ) -> Optional[TradeSignal]:
        """
        Genera un segnale di trading basato sullo Z-score corrente
        
        Args:
            current_zscore: Z-score corrente dello spread
            price1: Prezzo corrente di symbol1
            price2: Prezzo corrente di symbol2
            timestamp: Timestamp del segnale
            
        Returns:
            TradeSignal se c'è un segnale, None altrimenti
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        spread = price1 - self.hedge_ratio * price2
        
        # Se non abbiamo posizione aperta
        if self.current_position is None:
            # Z-score alto: spread troppo ampio verso l'alto
            # symbol1 è troppo caro rispetto a symbol2
            # -> Short symbol1, Long symbol2
            if current_zscore > self.entry_zscore:
                return TradeSignal(
                    timestamp=timestamp,
                    signal_type="OPEN",
                    position_type=PositionType.SHORT_FIRST_LONG_SECOND,
                    zscore=current_zscore,
                    spread=spread,
                    symbol1_price=price1,
                    symbol2_price=price2,
                    hedge_ratio=self.hedge_ratio
                )
            
            # Z-score basso: spread troppo ampio verso il basso
            # symbol1 è troppo economico rispetto a symbol2
            # -> Long symbol1, Short symbol2
            elif current_zscore < -self.entry_zscore:
                return TradeSignal(
                    timestamp=timestamp,
                    signal_type="OPEN",
                    position_type=PositionType.LONG_FIRST_SHORT_SECOND,
                    zscore=current_zscore,
                    spread=spread,
                    symbol1_price=price1,
                    symbol2_price=price2,
                    hedge_ratio=self.hedge_ratio
                )
        
        # Se abbiamo posizione aperta
        else:
            # Check stop loss
            if abs(current_zscore) > self.stop_loss_zscore:
                return TradeSignal(
                    timestamp=timestamp,
                    signal_type="STOP_LOSS",
                    position_type=self.current_position.position_type,
                    zscore=current_zscore,
                    spread=spread,
                    symbol1_price=price1,
                    symbol2_price=price2,
                    hedge_ratio=self.hedge_ratio
                )
            
            # Check exit (ritorno alla media)
            if self.current_position.position_type == PositionType.SHORT_FIRST_LONG_SECOND:
                # Chiudi se z-score scende sotto soglia exit (spread si è ristretto)
                if current_zscore < self.exit_zscore:
                    return TradeSignal(
                        timestamp=timestamp,
                        signal_type="CLOSE",
                        position_type=self.current_position.position_type,
                        zscore=current_zscore,
                        spread=spread,
                        symbol1_price=price1,
                        symbol2_price=price2,
                        hedge_ratio=self.hedge_ratio
                    )
            
            elif self.current_position.position_type == PositionType.LONG_FIRST_SHORT_SECOND:
                # Chiudi se z-score sale sopra -soglia exit (spread si è ristretto)
                if current_zscore > -self.exit_zscore:
                    return TradeSignal(
                        timestamp=timestamp,
                        signal_type="CLOSE",
                        position_type=self.current_position.position_type,
                        zscore=current_zscore,
                        spread=spread,
                        symbol1_price=price1,
                        symbol2_price=price2,
                        hedge_ratio=self.hedge_ratio
                    )
        
        return None
    
    def open_position(
        self,
        signal: TradeSignal,
        capital: float,
        position_size_pct: float = None
    ) -> Position:
        """
        Apre una nuova posizione basata sul segnale
        
        Args:
            signal: Segnale di apertura
            capital: Capitale disponibile
            position_size_pct: Percentuale del capitale da usare
            
        Returns:
            Nuova posizione
        """
        if position_size_pct is None:
            position_size_pct = STRATEGY_CONFIG["position_size_pct"]
        
        position_value = capital * position_size_pct
        
        # Calcola quantità per ogni leg
        # Dividiamo equamente il capitale tra le due posizioni
        half_value = position_value / 2
        
        qty1 = int(half_value / signal.symbol1_price)
        qty2 = int(half_value / signal.symbol2_price)
        
        # Aggiusta qty2 per hedge ratio se necessario
        # qty2 = int(qty1 * signal.hedge_ratio)
        
        self.current_position = Position(
            entry_date=signal.timestamp,
            position_type=signal.position_type,
            entry_zscore=signal.zscore,
            entry_spread=signal.spread,
            symbol1_quantity=qty1,
            symbol2_quantity=qty2,
            symbol1_entry_price=signal.symbol1_price,
            symbol2_entry_price=signal.symbol2_price
        )
        
        return self.current_position
    
    def close_position(self, signal: TradeSignal) -> Dict:
        """
        Chiude la posizione corrente
        
        Args:
            signal: Segnale di chiusura
            
        Returns:
            Dizionario con risultati del trade
        """
        if self.current_position is None:
            return {}
        
        pos = self.current_position
        
        # Calcola P&L
        if pos.position_type == PositionType.LONG_FIRST_SHORT_SECOND:
            # Long su symbol1, Short su symbol2
            pnl_symbol1 = (signal.symbol1_price - pos.symbol1_entry_price) * pos.symbol1_quantity
            pnl_symbol2 = (pos.symbol2_entry_price - signal.symbol2_price) * pos.symbol2_quantity
        else:
            # Short su symbol1, Long su symbol2
            pnl_symbol1 = (pos.symbol1_entry_price - signal.symbol1_price) * pos.symbol1_quantity
            pnl_symbol2 = (signal.symbol2_price - pos.symbol2_entry_price) * pos.symbol2_quantity
        
        total_pnl = pnl_symbol1 + pnl_symbol2
        
        # Calcola holding period
        holding_days = (signal.timestamp - pos.entry_date).days
        
        result = {
            "entry_date": pos.entry_date,
            "exit_date": signal.timestamp,
            "holding_days": holding_days,
            "position_type": pos.position_type.value,
            "entry_zscore": pos.entry_zscore,
            "exit_zscore": signal.zscore,
            "exit_type": signal.signal_type,
            "symbol1_pnl": pnl_symbol1,
            "symbol2_pnl": pnl_symbol2,
            "total_pnl": total_pnl,
            "symbol1_entry": pos.symbol1_entry_price,
            "symbol1_exit": signal.symbol1_price,
            "symbol2_entry": pos.symbol2_entry_price,
            "symbol2_exit": signal.symbol2_price,
            "symbol1_qty": pos.symbol1_quantity,
            "symbol2_qty": pos.symbol2_quantity
        }
        
        self.current_position = None
        
        return result
    
    def get_analytics(self, prices_df: pd.DataFrame = None) -> Dict:
        """
        Calcola analytics sulla coppia
        
        Args:
            prices_df: DataFrame con prezzi (opzionale, usa dati interni se disponibili)
            
        Returns:
            Dizionario con statistiche
        """
        if prices_df is None:
            prices_df = self.prices_df
        
        if prices_df is None:
            return {}
        
        prices1 = prices_df[self.symbol1]
        prices2 = prices_df[self.symbol2]
        
        # Correlazione
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        correlation = returns1.corr(returns2)
        
        # Statistiche spread
        spread = self.calculate_spread(prices1, prices2)
        zscore = self.calculate_zscore(spread)
        
        # Cointegration test (semplificato - Augmented Dickey-Fuller)
        try:
            from scipy import stats
            adf_stat, adf_pvalue = self._simple_adf_test(spread.dropna())
        except:
            adf_stat, adf_pvalue = None, None
        
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "correlation": correlation,
            "hedge_ratio": self.hedge_ratio,
            "current_zscore": zscore.iloc[-1] if len(zscore) > 0 else None,
            "spread_mean": spread.mean(),
            "spread_std": spread.std(),
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_pvalue,
            "is_cointegrated": adf_pvalue < 0.05 if adf_pvalue else None
        }
    
    def _simple_adf_test(self, series: pd.Series) -> Tuple[float, float]:
        """
        Test ADF semplificato per verificare stazionarietà
        
        Returns:
            (statistica, p-value)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, autolag='AIC')
            return result[0], result[1]
        except:
            # Fallback se statsmodels non disponibile
            return None, None


def test_strategy():
    """Test della strategia"""
    from data_fetcher import DataFetcher
    
    # Scarica dati
    fetcher = DataFetcher()
    prices = fetcher.get_aligned_prices("KO", "PEP", years=3)
    
    # Inizializza strategia
    strategy = PairsStrategy("KO", "PEP")
    strategy.fit(prices)
    
    # Analytics
    analytics = strategy.get_analytics()
    print("\n=== Analytics ===")
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Simula segnali sugli ultimi dati
    print("\n=== Ultimi Z-scores ===")
    print(strategy.zscore_series.tail(10))


if __name__ == "__main__":
    test_strategy()
