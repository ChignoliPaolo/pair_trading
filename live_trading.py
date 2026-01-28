"""
Script per il Trading Live con Interactive Brokers
Monitora le coppie e esegue automaticamente i trade basati sulla strategia
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import IB_CONFIG, STRATEGY_CONFIG, PAIRS_TO_MONITOR
from src.ib_connector import IBConnector, MockIBConnector
from src.pairs_strategy import PairsStrategy, PositionType
from src.data_fetcher import DataFetcher

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LivePairsTrader:
    """
    Trader live per la strategia Pairs Trading
    """
    
    def __init__(
        self,
        pairs: List[tuple],
        use_mock: bool = True,
        config: dict = None
    ):
        """
        Inizializza il trader
        
        Args:
            pairs: Lista di coppie da monitorare [(symbol1, symbol2), ...]
            use_mock: Se True, usa connettore simulato
            config: Configurazione strategia
        """
        self.pairs = pairs
        self.config = config or STRATEGY_CONFIG
        
        # Connettore IB
        if use_mock:
            self.ib = MockIBConnector()
            logger.info("Usando connettore IB simulato (paper trading)")
        else:
            self.ib = IBConnector()
            logger.info("Usando connettore IB reale")
        
        # Data fetcher per dati storici
        self.data_fetcher = DataFetcher()
        
        # Strategie per ogni coppia
        self.strategies: Dict[tuple, PairsStrategy] = {}
        
        # Posizioni aperte
        self.open_positions: Dict[tuple, dict] = {}
        
        # Stato
        self.running = False
        
    def initialize_strategies(self):
        """Inizializza le strategie per ogni coppia"""
        logger.info("Inizializzazione strategie...")
        
        for symbol1, symbol2 in self.pairs:
            try:
                # Carica dati storici per calibrazione
                prices = self.data_fetcher.get_aligned_prices(
                    symbol1, symbol2,
                    years=2  # 2 anni per calibrazione
                )
                
                # Crea e calibra strategia
                strategy = PairsStrategy(
                    symbol1, symbol2,
                    entry_zscore=self.config["entry_zscore"],
                    exit_zscore=self.config["exit_zscore"],
                    stop_loss_zscore=self.config["stop_loss_zscore"],
                    lookback_period=self.config["lookback_period"]
                )
                strategy.fit(prices)
                
                self.strategies[(symbol1, symbol2)] = strategy
                
                analytics = strategy.get_analytics()
                logger.info(f"Inizializzata {symbol1}/{symbol2}: "
                           f"corr={analytics['correlation']:.3f}, "
                           f"hedge_ratio={analytics['hedge_ratio']:.3f}")
                
            except Exception as e:
                logger.error(f"Errore inizializzazione {symbol1}/{symbol2}: {e}")
    
    def connect(self) -> bool:
        """Connette a Interactive Brokers"""
        return self.ib.connect()
    
    def disconnect(self):
        """Disconnette da Interactive Brokers"""
        self.ib.disconnect()
    
    def get_account_value(self) -> Optional[float]:
        """Ottiene il valore del conto"""
        try:
            summary = self.ib.get_account_summary()
            return float(summary.get("NetLiquidation", {}).get("value", 0))
        except:
            return None
    
    def calculate_position_size(
        self,
        capital: float,
        price1: float,
        price2: float
    ) -> tuple:
        """
        Calcola la dimensione delle posizioni
        
        Returns:
            (qty1, qty2)
        """
        position_value = capital * self.config["position_size_pct"]
        half_value = position_value / 2
        
        qty1 = int(half_value / price1)
        qty2 = int(half_value / price2)
        
        return qty1, qty2
    
    def update_zscore(
        self,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float
    ) -> float:
        """
        Aggiorna lo z-score con i prezzi correnti
        
        Returns:
            Z-score corrente
        """
        strategy = self.strategies.get((symbol1, symbol2))
        if strategy is None:
            return 0.0
        
        # Calcola spread corrente
        spread = price1 - strategy.hedge_ratio * price2
        
        # Calcola z-score usando statistiche storiche
        zscore = (spread - strategy.spread_mean) / strategy.spread_std
        
        return zscore
    
    def check_and_execute(self, symbol1: str, symbol2: str):
        """
        Controlla i segnali e esegue i trade per una coppia
        """
        pair = (symbol1, symbol2)
        strategy = self.strategies.get(pair)
        
        if strategy is None:
            return
        
        # Ottieni prezzi correnti
        price1, price2 = self.ib.get_pair_prices(symbol1, symbol2)
        
        if price1 is None or price2 is None:
            logger.warning(f"Prezzi non disponibili per {symbol1}/{symbol2}")
            return
        
        # Aggiorna e calcola z-score
        zscore = self.update_zscore(symbol1, symbol2, price1, price2)
        
        logger.info(f"{symbol1}/{symbol2}: Price1=${price1:.2f}, Price2=${price2:.2f}, Z={zscore:.2f}")
        
        # Genera segnale
        signal = strategy.generate_signal(zscore, price1, price2, datetime.now())
        
        if signal is None:
            return
        
        # Gestisci segnale
        if signal.signal_type == "OPEN" and pair not in self.open_positions:
            self._execute_open(pair, signal, price1, price2)
            
        elif signal.signal_type in ["CLOSE", "STOP_LOSS"] and pair in self.open_positions:
            self._execute_close(pair, signal, price1, price2)
    
    def _execute_open(self, pair: tuple, signal, price1: float, price2: float):
        """Esegue apertura posizione"""
        symbol1, symbol2 = pair
        
        # Ottieni capitale
        capital = self.get_account_value() or self.config["initial_capital"]
        
        # Calcola quantità
        qty1, qty2 = self.calculate_position_size(capital, price1, price2)
        
        if qty1 == 0 or qty2 == 0:
            logger.warning(f"Quantità insufficiente per {pair}")
            return
        
        # Determina direzione
        if signal.position_type == PositionType.LONG_FIRST_SHORT_SECOND:
            # Long symbol1, Short symbol2
            long_symbol, short_symbol = symbol1, symbol2
            long_qty, short_qty = qty1, qty2
        else:
            # Short symbol1, Long symbol2
            long_symbol, short_symbol = symbol2, symbol1
            long_qty, short_qty = qty2, qty1
        
        # Esegui trade
        logger.info(f"APERTURA: Long {long_qty} {long_symbol}, Short {short_qty} {short_symbol}")
        
        try:
            order_ids = self.ib.place_pairs_trade(
                long_symbol, short_symbol, long_qty, short_qty
            )
            
            # Registra posizione
            self.open_positions[pair] = {
                "open_time": datetime.now(),
                "position_type": signal.position_type,
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_qty": long_qty,
                "short_qty": short_qty,
                "entry_price1": price1,
                "entry_price2": price2,
                "entry_zscore": signal.zscore,
                "order_ids": order_ids
            }
            
            # Aggiorna stato strategia
            self.strategies[pair].open_position(signal, capital)
            
            logger.info(f"Posizione aperta: {pair}, Z-score={signal.zscore:.2f}")
            
        except Exception as e:
            logger.error(f"Errore apertura posizione {pair}: {e}")
    
    def _execute_close(self, pair: tuple, signal, price1: float, price2: float):
        """Esegue chiusura posizione"""
        position = self.open_positions.get(pair)
        
        if position is None:
            return
        
        logger.info(f"CHIUSURA ({signal.signal_type}): {pair}")
        
        try:
            order_ids = self.ib.close_pairs_position(
                position["long_symbol"],
                position["short_symbol"],
                position["long_qty"],
                position["short_qty"]
            )
            
            # Calcola P&L
            trade_result = self.strategies[pair].close_position(signal)
            
            logger.info(f"Posizione chiusa: {pair}, "
                       f"Entry Z={position['entry_zscore']:.2f}, "
                       f"Exit Z={signal.zscore:.2f}, "
                       f"P&L=${trade_result.get('total_pnl', 0):,.2f}")
            
            # Rimuovi posizione
            del self.open_positions[pair]
            
        except Exception as e:
            logger.error(f"Errore chiusura posizione {pair}: {e}")
    
    def run(self, interval_seconds: int = 60):
        """
        Avvia il loop di trading
        
        Args:
            interval_seconds: Intervallo tra i check (default: 60 secondi)
        """
        logger.info("Avvio trading loop...")
        self.running = True
        
        try:
            while self.running:
                logger.info("-" * 50)
                logger.info(f"Check: {datetime.now()}")
                
                for symbol1, symbol2 in self.pairs:
                    try:
                        self.check_and_execute(symbol1, symbol2)
                    except Exception as e:
                        logger.error(f"Errore check {symbol1}/{symbol2}: {e}")
                
                # Report posizioni aperte
                if self.open_positions:
                    logger.info(f"Posizioni aperte: {list(self.open_positions.keys())}")
                
                # Attendi
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Interruzione manuale")
            self.stop()
    
    def stop(self):
        """Ferma il trading loop"""
        logger.info("Arresto trading...")
        self.running = False
        
        # Chiudi posizioni aperte (opzionale)
        if self.open_positions:
            logger.warning(f"Posizioni ancora aperte: {list(self.open_positions.keys())}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pairs Trading Live")
    parser.add_argument("--live", action="store_true", help="Usa connessione IB reale")
    parser.add_argument("--interval", type=int, default=60, help="Intervallo check (secondi)")
    parser.add_argument("--pairs", nargs="+", help="Coppie da tradare (es: KO,PEP V,MA)")
    
    args = parser.parse_args()
    
    # Parse coppie
    if args.pairs:
        pairs = [tuple(p.split(",")) for p in args.pairs]
    else:
        pairs = PAIRS_TO_MONITOR[:3]  # Default: prime 3 coppie
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              PAIRS TRADING - LIVE TRADING                    ║
╠══════════════════════════════════════════════════════════════╣
║  Modalità: {'LIVE' if args.live else 'SIMULAZIONE (Paper)'}
║  Intervallo: {args.interval} secondi
║  Coppie: {pairs}
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.live:
        confirm = input("ATTENZIONE: Stai per avviare il trading LIVE. Confermi? (yes/no): ")
        if confirm.lower() != "yes":
            print("Operazione annullata.")
            return
    
    # Crea trader
    trader = LivePairsTrader(
        pairs=pairs,
        use_mock=not args.live
    )
    
    # Connetti
    if not trader.connect():
        print("Impossibile connettersi a IB. Assicurati che TWS sia in esecuzione.")
        return
    
    try:
        # Inizializza strategie
        trader.initialize_strategies()
        
        # Avvia trading
        trader.run(interval_seconds=args.interval)
        
    finally:
        trader.disconnect()


if __name__ == "__main__":
    main()
