"""
Configurazione per la strategia di Pairs Trading
"""

# ============================================
# CONFIGURAZIONE INTERACTIVE BROKERS
# ============================================
IB_CONFIG = {
    "host": "127.0.0.1",
    "port": 7497,  # 7497 per TWS Paper Trading, 7496 per TWS Live, 4001 per IB Gateway
    "client_id": 1,
    "timeout": 60,
}

# ============================================
# CONFIGURAZIONE STRATEGIA PAIRS TRADING
# ============================================
STRATEGY_CONFIG = {
    # Soglia Z-score per apertura posizioni (numero di deviazioni standard)
    "entry_zscore": 2.0,
    
    # Soglia Z-score per chiusura posizioni (ritorno alla media)
    "exit_zscore": 0.5,
    
    # Soglia Z-score per stop loss (uscita forzata se divergenza continua)
    "stop_loss_zscore": 4.0,
    
    # Finestra per calcolo media mobile e deviazione standard (giorni)
    "lookback_period": 60,
    
    # Capitale iniziale per backtest (USD)
    "initial_capital": 100000,
    
    # Percentuale del capitale da allocare per trade
    "position_size_pct": 0.25,
    
    # Commissioni per trade (percentuale)
    "commission_pct": 0.001,
    
    # Slippage stimato (percentuale)
    "slippage_pct": 0.0005,
}

# ============================================
# COPPIE DI AZIONI DA MONITORARE
# ============================================
# Esempi di coppie storicamente correlate
PAIRS_TO_MONITOR = [
    ("KO", "PEP"),      # Coca-Cola vs Pepsi
    ("V", "MA"),        # Visa vs Mastercard
    ("XOM", "CVX"),     # Exxon vs Chevron
    ("JPM", "BAC"),     # JPMorgan vs Bank of America
    ("GOOG", "META"),   # Google vs Meta
    ("HD", "LOW"),      # Home Depot vs Lowe's
    ("WMT", "TGT"),     # Walmart vs Target
    ("DIS", "CMCSA"),   # Disney vs Comcast
]

# ============================================
# BENCHMARK PER CONFRONTO
# ============================================
BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "VTI": "Total US Market",
    "VT": "Total World Market",
    "QQQ": "Nasdaq 100",
}

# ============================================
# CONFIGURAZIONE DATI
# ============================================
DATA_CONFIG = {
    # Periodo di default per backtest (anni)
    "default_years": 5,
    
    # Intervallo dati
    "interval": "1d",
    
    # Cartella per cache dati
    "data_folder": "data",
}
