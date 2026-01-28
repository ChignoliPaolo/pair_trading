"""
Modulo per la connessione a Interactive Brokers
Utilizza ib_insync per un'interfaccia Python-friendly
"""

import os
import sys
import asyncio
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IB_CONFIG

# Fix per Python 3.10+ asyncio event loop
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract, util
    IB_AVAILABLE = True
except ImportError as e:
    IB_AVAILABLE = False
    print(f"Warning: ib_insync non disponibile: {e}")


class IBConnector:
    """
    Classe per gestire la connessione e le operazioni con Interactive Brokers
    """
    
    def __init__(self, config: dict = None):
        """
        Inizializza il connettore IB
        
        Args:
            config: Configurazione di connessione (usa IB_CONFIG se non specificato)
        """
        if not IB_AVAILABLE:
            raise ImportError("ib_insync non è installato. Installa con: pip install ib_insync")
        
        self.config = config or IB_CONFIG
        self.ib = IB()
        self.connected = False
        self.positions = {}
        
    def connect(self) -> bool:
        """
        Stabilisce la connessione con TWS/IB Gateway
        
        Returns:
            True se connesso con successo
        """
        try:
            self.ib.connect(
                host=self.config["host"],
                port=self.config["port"],
                clientId=self.config["client_id"],
                timeout=self.config["timeout"]
            )
            self.connected = True
            print(f"Connesso a Interactive Brokers - Account: {self.ib.managedAccounts()}")
            return True
        except Exception as e:
            print(f"Errore di connessione a IB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Chiude la connessione"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("Disconnesso da Interactive Brokers")
    
    def _check_connection(self):
        """Verifica che la connessione sia attiva"""
        if not self.connected:
            raise ConnectionError("Non connesso a Interactive Brokers")
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
        """
        Crea un contratto per un'azione
        
        Args:
            symbol: Ticker dell'azione
            exchange: Exchange (default: SMART per routing intelligente)
            currency: Valuta (default: USD)
            
        Returns:
            Oggetto Stock
        """
        return Stock(symbol, exchange, currency)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Ottiene il prezzo corrente di un'azione
        
        Args:
            symbol: Ticker dell'azione
            
        Returns:
            Prezzo corrente o None se non disponibile
        """
        self._check_connection()
        
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)  # Attendi i dati
        
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        
        return price if price > 0 else None
    
    def get_pair_prices(self, symbol1: str, symbol2: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Ottiene i prezzi correnti per una coppia di azioni
        
        Args:
            symbol1: Ticker della prima azione
            symbol2: Ticker della seconda azione
            
        Returns:
            Tupla di prezzi (price1, price2)
        """
        self._check_connection()
        
        contract1 = self.create_stock_contract(symbol1)
        contract2 = self.create_stock_contract(symbol2)
        
        self.ib.qualifyContracts(contract1, contract2)
        
        ticker1 = self.ib.reqMktData(contract1)
        ticker2 = self.ib.reqMktData(contract2)
        
        self.ib.sleep(2)
        
        price1 = ticker1.marketPrice()
        price2 = ticker2.marketPrice()
        
        self.ib.cancelMktData(contract1)
        self.ib.cancelMktData(contract2)
        
        return (
            price1 if price1 > 0 else None,
            price2 if price2 > 0 else None
        )
    
    def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST"
    ) -> pd.DataFrame:
        """
        Ottiene i dati storici da IB
        
        Args:
            symbol: Ticker dell'azione
            duration: Durata (es. "1 Y", "6 M", "30 D")
            bar_size: Dimensione delle barre (es. "1 day", "1 hour")
            what_to_show: Tipo di dati (ADJUSTED_LAST, TRADES, MIDPOINT)
            
        Returns:
            DataFrame con dati storici
        """
        self._check_connection()
        
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            return pd.DataFrame()
        
        df = util.df(bars)
        df.set_index("date", inplace=True)
        return df
    
    def get_account_summary(self) -> Dict:
        """
        Ottiene il riepilogo del conto
        
        Returns:
            Dizionario con informazioni del conto
        """
        self._check_connection()
        
        account_values = self.ib.accountSummary()
        
        summary = {}
        for av in account_values:
            summary[av.tag] = {
                "value": av.value,
                "currency": av.currency
            }
        
        return summary
    
    def get_positions(self) -> List[Dict]:
        """
        Ottiene le posizioni aperte
        
        Returns:
            Lista di dizionari con le posizioni
        """
        self._check_connection()
        
        positions = self.ib.positions()
        
        return [
            {
                "symbol": pos.contract.symbol,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost
            }
            for pos in positions
        ]
    
    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str = "BUY"
    ) -> Optional[int]:
        """
        Piazza un ordine a mercato
        
        Args:
            symbol: Ticker dell'azione
            quantity: Quantità (positivo)
            action: "BUY" o "SELL"
            
        Returns:
            ID dell'ordine o None se fallito
        """
        self._check_connection()
        
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        
        self.ib.sleep(1)
        
        return trade.order.orderId
    
    def place_pairs_trade(
        self,
        symbol_long: str,
        symbol_short: str,
        quantity_long: int,
        quantity_short: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Piazza un trade di coppia (long su una, short sull'altra)
        
        Args:
            symbol_long: Azione da comprare
            symbol_short: Azione da vendere allo scoperto
            quantity_long: Quantità da comprare
            quantity_short: Quantità da vendere
            
        Returns:
            Tupla di order IDs (long_id, short_id)
        """
        self._check_connection()
        
        # Ordine long
        long_id = self.place_market_order(symbol_long, quantity_long, "BUY")
        
        # Ordine short
        short_id = self.place_market_order(symbol_short, quantity_short, "SELL")
        
        return long_id, short_id
    
    def close_pairs_position(
        self,
        symbol_long: str,
        symbol_short: str,
        quantity_long: int,
        quantity_short: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Chiude una posizione di coppia
        
        Args:
            symbol_long: Azione in posizione long
            symbol_short: Azione in posizione short
            quantity_long: Quantità long da chiudere
            quantity_short: Quantità short da chiudere
            
        Returns:
            Tupla di order IDs per la chiusura
        """
        self._check_connection()
        
        # Chiudi long (vendi)
        close_long_id = self.place_market_order(symbol_long, quantity_long, "SELL")
        
        # Chiudi short (compra per coprire)
        close_short_id = self.place_market_order(symbol_short, quantity_short, "BUY")
        
        return close_long_id, close_short_id
    
    def cancel_order(self, order_id: int):
        """
        Cancella un ordine
        
        Args:
            order_id: ID dell'ordine da cancellare
        """
        self._check_connection()
        
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                break
    
    def get_open_orders(self) -> List[Dict]:
        """
        Ottiene gli ordini aperti
        
        Returns:
            Lista di ordini aperti
        """
        self._check_connection()
        
        trades = self.ib.openTrades()
        
        return [
            {
                "order_id": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "status": trade.orderStatus.status
            }
            for trade in trades
        ]


class MockIBConnector:
    """
    Connettore IB simulato per testing senza connessione reale
    """
    
    def __init__(self, config: dict = None):
        self.config = config or IB_CONFIG
        self.connected = False
        self.positions = {}
        self.orders = []
        self.order_id_counter = 1
        
    def connect(self) -> bool:
        self.connected = True
        print("Mock IB: Connesso (simulazione)")
        return True
    
    def disconnect(self):
        self.connected = False
        print("Mock IB: Disconnesso")
    
    def get_current_price(self, symbol: str) -> float:
        # Restituisce prezzi simulati
        mock_prices = {
            "KO": 62.50,
            "PEP": 175.30,
            "V": 280.00,
            "MA": 450.00,
            "XOM": 110.00,
            "CVX": 155.00,
        }
        return mock_prices.get(symbol, 100.00)
    
    def get_pair_prices(self, symbol1: str, symbol2: str) -> Tuple[float, float]:
        return (self.get_current_price(symbol1), self.get_current_price(symbol2))
    
    def place_pairs_trade(
        self,
        symbol_long: str,
        symbol_short: str,
        quantity_long: int,
        quantity_short: int
    ) -> Tuple[int, int]:
        long_id = self.order_id_counter
        self.order_id_counter += 1
        short_id = self.order_id_counter
        self.order_id_counter += 1
        
        print(f"Mock IB: Trade piazzato - Long {quantity_long} {symbol_long}, Short {quantity_short} {symbol_short}")
        return long_id, short_id
    
    def close_pairs_position(
        self,
        symbol_long: str,
        symbol_short: str,
        quantity_long: int,
        quantity_short: int
    ) -> Tuple[int, int]:
        close_long_id = self.order_id_counter
        self.order_id_counter += 1
        close_short_id = self.order_id_counter
        self.order_id_counter += 1
        
        print(f"Mock IB: Posizione chiusa - {symbol_long}/{symbol_short}")
        return close_long_id, close_short_id


def test_connection():
    """Test della connessione (richiede TWS/IB Gateway attivo)"""
    try:
        ib = IBConnector()
        if ib.connect():
            print("Connessione riuscita!")
            
            # Test prezzo
            price = ib.get_current_price("AAPL")
            print(f"Prezzo AAPL: ${price:.2f}")
            
            # Test account
            summary = ib.get_account_summary()
            if "NetLiquidation" in summary:
                print(f"Valore netto: ${summary['NetLiquidation']['value']}")
            
            ib.disconnect()
        else:
            print("Connessione fallita")
    except Exception as e:
        print(f"Errore: {e}")
        print("Assicurati che TWS o IB Gateway sia in esecuzione")


if __name__ == "__main__":
    test_connection()
