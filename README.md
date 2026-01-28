# 📈 Pairs Trading Strategy

Sistema completo per il trading di coppie di azioni correlate (Pairs Trading / Statistical Arbitrage) con connessione a Interactive Brokers.

## 🎯 Cos'è il Pairs Trading?

Il **Pairs Trading** è una strategia di trading market-neutral che:

1. Identifica due azioni storicamente correlate (es. Coca-Cola e Pepsi)
2. Monitora lo "spread" tra i loro prezzi
3. Quando lo spread diverge significativamente dalla media:
   - **Vende allo scoperto** l'azione "sopravvalutata"
   - **Compra** l'azione "sottovalutata"
4. Chiude le posizioni quando lo spread ritorna verso la media

### Vantaggi
- ✅ Strategia market-neutral (profitto indipendente dalla direzione del mercato)
- ✅ Basata su mean reversion statistica
- ✅ Riduce l'esposizione al rischio di mercato

## 📁 Struttura del Progetto

```
strategy_trading/
├── app.py                 # Dashboard Streamlit
├── live_trading.py        # Script per trading live
├── config.py              # Configurazione parametri
├── requirements.txt       # Dipendenze Python
├── README.md              # Questo file
├── data/                  # Cache dati storici
└── src/
    ├── __init__.py
    ├── data_fetcher.py    # Download dati storici (yfinance)
    ├── pairs_strategy.py  # Logica strategia
    ├── backtester.py      # Backtesting
    └── ib_connector.py    # Connessione Interactive Brokers
```

## 🚀 Installazione

### 1. Clona o scarica il progetto

### 2. Crea un ambiente virtuale (consigliato)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. (Opzionale) Configura Interactive Brokers

Per il trading live, devi avere:
- Account Interactive Brokers (anche Paper Trading)
- TWS (Trader Workstation) o IB Gateway installato e in esecuzione
- API abilitate in TWS: `File > Global Configuration > API > Settings`
  - Abilita "Enable ActiveX and Socket Clients"
  - Porta Socket: 7497 (paper) o 7496 (live)

## 📊 Uso della Dashboard

### Avvia l'interfaccia grafica:

```bash
streamlit run app.py
```

La dashboard si aprirà nel browser (default: http://localhost:8501)

### Funzionalità:

1. **Selezione Coppia**: Scegli tra coppie predefinite o inserisci manualmente
2. **Parametri Strategia**: Configura Z-score entry/exit, lookback period
3. **Backtest**: Simula la strategia sui dati storici
4. **Grafici**:
   - Prezzi normalizzati delle due azioni
   - Spread e Z-score nel tempo
   - Equity curve
   - Drawdown
   - P&L per singolo trade
   - Rendimenti mensili (heatmap)
5. **Confronto Benchmark**: Compara con S&P 500, mercato globale, etc.

## 💹 Trading Live

### Modalità Simulazione (Paper Trading)

```bash
python live_trading.py
```

### Modalità Live (ATTENZIONE: denaro reale!)

```bash
python live_trading.py --live
```

### Opzioni disponibili:

```bash
python live_trading.py --help

Opzioni:
  --live              Usa connessione IB reale (default: simulazione)
  --interval N        Intervallo tra i check in secondi (default: 60)
  --pairs P1,P2 ...   Coppie da tradare (es: KO,PEP V,MA)
```

### Esempio:

```bash
# Trada KO/PEP e V/MA ogni 30 secondi in modalità paper
python live_trading.py --interval 30 --pairs KO,PEP V,MA
```

## ⚙️ Configurazione

Modifica `config.py` per personalizzare:

### Parametri Strategia

```python
STRATEGY_CONFIG = {
    "entry_zscore": 2.0,      # Apri posizione se |Z| > 2
    "exit_zscore": 0.5,       # Chiudi posizione se |Z| < 0.5
    "stop_loss_zscore": 4.0,  # Stop loss se |Z| > 4
    "lookback_period": 60,    # Giorni per calcolo media/std
    "initial_capital": 100000,
    "position_size_pct": 0.25,  # 25% del capitale per trade
}
```

### Coppie da Monitorare

```python
PAIRS_TO_MONITOR = [
    ("KO", "PEP"),      # Coca-Cola vs Pepsi
    ("V", "MA"),        # Visa vs Mastercard
    ("XOM", "CVX"),     # Exxon vs Chevron
    # Aggiungi le tue coppie...
]
```

### Configurazione IB

```python
IB_CONFIG = {
    "host": "127.0.0.1",
    "port": 7497,  # 7497 paper, 7496 live, 4001 IB Gateway
    "client_id": 1,
}
```

## 📈 Come Funziona la Strategia

### 1. Calcolo dello Spread

```
Spread = Prezzo_Azione1 - (Hedge_Ratio × Prezzo_Azione2)
```

L'Hedge Ratio viene calcolato tramite regressione lineare OLS.

### 2. Normalizzazione con Z-Score

```
Z-Score = (Spread_Corrente - Media_Spread) / Std_Spread
```

Calcolato su una finestra rolling (es. 60 giorni).

### 3. Segnali di Trading

| Condizione | Azione |
|------------|--------|
| Z > +entry_zscore | Short Azione1, Long Azione2 |
| Z < -entry_zscore | Long Azione1, Short Azione2 |
| \|Z\| < exit_zscore | Chiudi posizione |
| \|Z\| > stop_loss_zscore | Stop Loss |

### 4. Esempio Visivo

```
Z-Score
   ^
 4 |--------------------------- Stop Loss
   |
 2 |--------------------------- Entry (Short spread)
   |
 0 |--------------------------- Media
   |
-2 |--------------------------- Entry (Long spread)
   |
-4 |--------------------------- Stop Loss
   +--------------------------------> Tempo
```

## 📊 Metriche di Performance

La dashboard mostra:

- **Rendimento Totale/Annualizzato**
- **Sharpe Ratio**: Misura risk-adjusted return
- **Max Drawdown**: Massima perdita dal picco
- **Win Rate**: Percentuale trade vincenti
- **Profit Factor**: Profitti lordi / Perdite lorde
- **Durata Media Trade**

## ⚠️ Disclaimer

**ATTENZIONE**: Questo software è fornito a scopo educativo e di ricerca.

- Il trading comporta rischi significativi di perdita del capitale
- I risultati passati (backtest) NON garantiscono risultati futuri
- Il pairs trading richiede la possibilità di vendere allo scoperto
- Testa SEMPRE prima in paper trading
- Consulta un consulente finanziario prima di fare trading reale

## 🛠️ Troubleshooting

### "Impossibile connettersi a IB"
1. Verifica che TWS o IB Gateway sia in esecuzione
2. Controlla che le API siano abilitate
3. Verifica la porta (7497 per paper, 7496 per live)

### "Dati insufficienti"
- Alcune azioni potrebbero non avere dati sufficienti
- yfinance potrebbe avere limiti temporanei

### "Correlazione bassa"
- Non tutte le coppie sono adatte per pairs trading
- Cerca coppie con correlazione > 0.7 e cointegrazione

## 📚 Risorse

- [Pairs Trading su Wikipedia](https://en.wikipedia.org/wiki/Pairs_trade)
- [Interactive Brokers API](https://interactivebrokers.github.io/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)

## 📝 Licenza

MIT License - Usa liberamente ma a tuo rischio e pericolo.
