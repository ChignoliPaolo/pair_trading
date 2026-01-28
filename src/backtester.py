"""
Modulo per il backtesting della strategia di Pairs Trading
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGY_CONFIG, BENCHMARKS
from src.pairs_strategy import PairsStrategy, PositionType, TradeSignal
from src.data_fetcher import DataFetcher


@dataclass
class BacktestResult:
    """Risultati del backtest"""
    symbol1: str
    symbol2: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return: float
    avg_holding_days: float
    profit_factor: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Backtester per la strategia di Pairs Trading
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission_pct: float = None,
        slippage_pct: float = None
    ):
        """
        Inizializza il backtester
        
        Args:
            initial_capital: Capitale iniziale
            commission_pct: Commissioni percentuali
            slippage_pct: Slippage percentuale
        """
        self.initial_capital = initial_capital or STRATEGY_CONFIG["initial_capital"]
        self.commission_pct = commission_pct or STRATEGY_CONFIG["commission_pct"]
        self.slippage_pct = slippage_pct or STRATEGY_CONFIG["slippage_pct"]
        
        self.data_fetcher = DataFetcher()
    
    def run_backtest(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime = None,
        end_date: datetime = None,
        years: int = None,
        entry_zscore: float = None,
        exit_zscore: float = None,
        stop_loss_zscore: float = None,
        lookback_period: int = None,
        position_size_pct: float = None
    ) -> BacktestResult:
        """
        Esegue il backtest della strategia
        
        Args:
            symbol1: Prima azione
            symbol2: Seconda azione
            start_date: Data inizio
            end_date: Data fine
            years: Anni di dati (alternativo a date)
            entry_zscore: Soglia entry
            exit_zscore: Soglia exit
            stop_loss_zscore: Soglia stop loss
            lookback_period: Periodo lookback
            position_size_pct: Dimensione posizione
            
        Returns:
            BacktestResult con tutti i risultati
        """
        # Scarica dati
        prices = self.data_fetcher.get_aligned_prices(
            symbol1, symbol2, start_date, end_date, years
        )
        
        if len(prices) < 100:
            raise ValueError(f"Dati insufficienti: {len(prices)} righe")
        
        # Inizializza strategia
        strategy = PairsStrategy(
            symbol1, symbol2,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_loss_zscore=stop_loss_zscore,
            lookback_period=lookback_period
        )
        
        # Calcola spread e z-score
        strategy.fit(prices)
        
        # Variabili di stato
        capital = self.initial_capital
        position_size = position_size_pct or STRATEGY_CONFIG["position_size_pct"]
        
        trades = []
        equity_history = []
        
        # Skip il periodo di lookback iniziale
        lookback = lookback_period or STRATEGY_CONFIG["lookback_period"]
        
        # Itera sui dati
        for i in range(lookback, len(prices)):
            date = prices.index[i]
            price1 = prices[symbol1].iloc[i]
            price2 = prices[symbol2].iloc[i]
            zscore = strategy.zscore_series.iloc[i]
            
            if pd.isna(zscore):
                equity_history.append({"date": date, "equity": capital})
                continue
            
            # Genera segnale
            signal = strategy.generate_signal(zscore, price1, price2, date)
            
            if signal is not None:
                if signal.signal_type == "OPEN" and strategy.current_position is None:
                    # Apri posizione
                    position = strategy.open_position(signal, capital, position_size)
                    
                    # Calcola costi di transazione
                    trade_value = (position.symbol1_quantity * price1 + 
                                 position.symbol2_quantity * price2)
                    transaction_cost = trade_value * (self.commission_pct + self.slippage_pct)
                    capital -= transaction_cost
                    
                elif signal.signal_type in ["CLOSE", "STOP_LOSS"] and strategy.current_position is not None:
                    # Chiudi posizione
                    trade_result = strategy.close_position(signal)
                    
                    # Calcola costi di transazione
                    trade_value = (trade_result["symbol1_qty"] * price1 + 
                                 trade_result["symbol2_qty"] * price2)
                    transaction_cost = trade_value * (self.commission_pct + self.slippage_pct)
                    
                    # Aggiorna capitale
                    capital += trade_result["total_pnl"] - transaction_cost
                    trade_result["transaction_cost"] = transaction_cost
                    trade_result["net_pnl"] = trade_result["total_pnl"] - transaction_cost
                    
                    trades.append(trade_result)
            
            # Registra equity (incluso unrealized P&L)
            unrealized_pnl = 0
            if strategy.current_position is not None:
                pos = strategy.current_position
                if pos.position_type == PositionType.LONG_FIRST_SHORT_SECOND:
                    unrealized_pnl = (
                        (price1 - pos.symbol1_entry_price) * pos.symbol1_quantity +
                        (pos.symbol2_entry_price - price2) * pos.symbol2_quantity
                    )
                else:
                    unrealized_pnl = (
                        (pos.symbol1_entry_price - price1) * pos.symbol1_quantity +
                        (price2 - pos.symbol2_entry_price) * pos.symbol2_quantity
                    )
            
            equity_history.append({
                "date": date,
                "equity": capital + unrealized_pnl
            })
        
        # Chiudi posizione aperta alla fine
        if strategy.current_position is not None:
            final_price1 = prices[symbol1].iloc[-1]
            final_price2 = prices[symbol2].iloc[-1]
            final_zscore = strategy.zscore_series.iloc[-1]
            
            close_signal = TradeSignal(
                timestamp=prices.index[-1],
                signal_type="CLOSE",
                position_type=strategy.current_position.position_type,
                zscore=final_zscore,
                spread=final_price1 - strategy.hedge_ratio * final_price2,
                symbol1_price=final_price1,
                symbol2_price=final_price2,
                hedge_ratio=strategy.hedge_ratio
            )
            
            trade_result = strategy.close_position(close_signal)
            trade_value = (trade_result["symbol1_qty"] * final_price1 + 
                         trade_result["symbol2_qty"] * final_price2)
            transaction_cost = trade_value * (self.commission_pct + self.slippage_pct)
            
            capital += trade_result["total_pnl"] - transaction_cost
            trade_result["transaction_cost"] = transaction_cost
            trade_result["net_pnl"] = trade_result["total_pnl"] - transaction_cost
            trades.append(trade_result)
        
        # Crea equity curve
        equity_df = pd.DataFrame(equity_history)
        equity_df.set_index("date", inplace=True)
        equity_curve = equity_df["equity"]
        
        # Calcola metriche
        result = self._calculate_metrics(
            symbol1, symbol2, trades, equity_curve, 
            prices.index[0], prices.index[-1]
        )
        
        # Aggiungi confronto con benchmark
        result.benchmark_returns = self._calculate_benchmark_returns(
            prices.index[0], prices.index[-1]
        )
        
        return result
    
    def _calculate_metrics(
        self,
        symbol1: str,
        symbol2: str,
        trades: List[Dict],
        equity_curve: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calcola le metriche di performance"""
        
        final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else self.initial_capital
        
        # Rendimento totale
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Rendimento annualizzato
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Calcola rendimenti giornalieri
        daily_returns = equity_curve.pct_change().dropna()
        
        # Sharpe Ratio (assumendo risk-free rate = 0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Statistiche trades
        total_trades = len(trades)
        
        if total_trades > 0:
            pnls = [t.get("net_pnl", t.get("total_pnl", 0)) for t in trades]
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            losing_trades = sum(1 for pnl in pnls if pnl <= 0)
            win_rate = winning_trades / total_trades
            avg_trade_return = np.mean(pnls) / (self.initial_capital * STRATEGY_CONFIG["position_size_pct"])
            
            holding_days = [t.get("holding_days", 0) for t in trades]
            avg_holding_days = np.mean(holding_days) if holding_days else 0
            
            # Profit factor
            gross_profit = sum(pnl for pnl in pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_trade_return = avg_holding_days = 0
            profit_factor = 0
        
        return BacktestResult(
            symbol1=symbol1,
            symbol2=symbol2,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            avg_holding_days=avg_holding_days,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_series=drawdown
        )
    
    def _calculate_benchmark_returns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calcola i rendimenti dei benchmark nello stesso periodo"""
        
        benchmark_returns = {}
        
        for symbol, name in BENCHMARKS.items():
            try:
                data = self.data_fetcher.fetch_stock_data(
                    symbol, start_date, end_date
                )
                if len(data) > 0:
                    start_price = data["Close"].iloc[0]
                    end_price = data["Close"].iloc[-1]
                    returns = (end_price - start_price) / start_price
                    benchmark_returns[symbol] = {
                        "name": name,
                        "return": returns
                    }
            except Exception as e:
                print(f"Warning: Impossibile calcolare benchmark {symbol}: {e}")
        
        return benchmark_returns
    
    def run_multiple_backtests(
        self,
        pairs: List[Tuple[str, str]],
        years: int = None,
        **strategy_params
    ) -> List[BacktestResult]:
        """
        Esegue backtest su multiple coppie
        
        Args:
            pairs: Lista di tuple (symbol1, symbol2)
            years: Anni di dati
            **strategy_params: Parametri strategia
            
        Returns:
            Lista di BacktestResult
        """
        results = []
        
        for symbol1, symbol2 in pairs:
            try:
                print(f"Backtesting {symbol1}/{symbol2}...")
                result = self.run_backtest(
                    symbol1, symbol2, years=years, **strategy_params
                )
                results.append(result)
                print(f"  Rendimento: {result.total_return:.2%}, "
                      f"Sharpe: {result.sharpe_ratio:.2f}, "
                      f"Trades: {result.total_trades}")
            except Exception as e:
                print(f"  Errore: {e}")
        
        return results
    
    def optimize_parameters(
        self,
        symbol1: str,
        symbol2: str,
        years: int = None,
        entry_zscore_range: Tuple[float, float, float] = (1.5, 3.0, 0.5),
        exit_zscore_range: Tuple[float, float, float] = (0.0, 1.0, 0.25),
        lookback_range: Tuple[int, int, int] = (30, 90, 15)
    ) -> Dict:
        """
        Ottimizza i parametri della strategia
        
        Args:
            symbol1: Prima azione
            symbol2: Seconda azione
            years: Anni di dati
            entry_zscore_range: (min, max, step) per entry zscore
            exit_zscore_range: (min, max, step) per exit zscore
            lookback_range: (min, max, step) per lookback
            
        Returns:
            Dizionario con parametri ottimali e risultati
        """
        best_sharpe = -float('inf')
        best_params = {}
        best_result = None
        all_results = []
        
        entry_values = np.arange(*entry_zscore_range)
        exit_values = np.arange(*exit_zscore_range)
        lookback_values = range(*lookback_range)
        
        total_combinations = len(entry_values) * len(exit_values) * len(lookback_values)
        print(f"Ottimizzazione: {total_combinations} combinazioni da testare")
        
        count = 0
        for entry_z in entry_values:
            for exit_z in exit_values:
                if exit_z >= entry_z:  # Exit deve essere < entry
                    continue
                    
                for lookback in lookback_values:
                    count += 1
                    
                    try:
                        result = self.run_backtest(
                            symbol1, symbol2,
                            years=years,
                            entry_zscore=entry_z,
                            exit_zscore=exit_z,
                            lookback_period=lookback
                        )
                        
                        all_results.append({
                            "entry_zscore": entry_z,
                            "exit_zscore": exit_z,
                            "lookback_period": lookback,
                            "sharpe_ratio": result.sharpe_ratio,
                            "total_return": result.total_return,
                            "max_drawdown": result.max_drawdown,
                            "total_trades": result.total_trades
                        })
                        
                        if result.sharpe_ratio > best_sharpe and result.total_trades >= 5:
                            best_sharpe = result.sharpe_ratio
                            best_params = {
                                "entry_zscore": entry_z,
                                "exit_zscore": exit_z,
                                "lookback_period": lookback
                            }
                            best_result = result
                            
                    except Exception as e:
                        pass
                    
                    if count % 10 == 0:
                        print(f"  Progresso: {count}/{total_combinations}")
        
        return {
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "best_result": best_result,
            "all_results": pd.DataFrame(all_results)
        }
    
    def compare_with_benchmarks(self, result: BacktestResult) -> pd.DataFrame:
        """
        Crea una tabella di confronto con i benchmark
        
        Args:
            result: Risultato del backtest
            
        Returns:
            DataFrame con confronto
        """
        data = [{
            "Strategy": f"{result.symbol1}/{result.symbol2} Pairs",
            "Total Return": f"{result.total_return:.2%}",
            "Annualized Return": f"{result.annualized_return:.2%}",
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Max Drawdown": f"{result.max_drawdown:.2%}"
        }]
        
        for symbol, info in result.benchmark_returns.items():
            # Calcola metriche benchmark
            bench_data = self.data_fetcher.fetch_stock_data(
                symbol, result.start_date, result.end_date
            )
            daily_returns = bench_data["Close"].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            
            rolling_max = bench_data["Close"].expanding().max()
            drawdown = (bench_data["Close"] - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            days = (result.end_date - result.start_date).days
            years = days / 365.25
            ann_return = (1 + info["return"]) ** (1/years) - 1 if years > 0 else 0
            
            data.append({
                "Strategy": info["name"],
                "Total Return": f"{info['return']:.2%}",
                "Annualized Return": f"{ann_return:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.2%}"
            })
        
        return pd.DataFrame(data)


def test_backtest():
    """Test del backtester"""
    backtester = Backtester()
    
    print("Esecuzione backtest KO/PEP...")
    result = backtester.run_backtest("KO", "PEP", years=3)
    
    print("\n=== Risultati Backtest ===")
    print(f"Periodo: {result.start_date.date()} - {result.end_date.date()}")
    print(f"Capitale iniziale: ${result.initial_capital:,.2f}")
    print(f"Capitale finale: ${result.final_capital:,.2f}")
    print(f"Rendimento totale: {result.total_return:.2%}")
    print(f"Rendimento annualizzato: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Trades totali: {result.total_trades}")
    print(f"Win rate: {result.win_rate:.2%}")
    print(f"Profit factor: {result.profit_factor:.2f}")
    
    print("\n=== Confronto Benchmark ===")
    comparison = backtester.compare_with_benchmarks(result)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    test_backtest()
