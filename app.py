"""
Interfaccia Streamlit per la strategia di Pairs Trading
Dashboard con grafici, backtest e confronto con benchmark
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Aggiungi path per import locali
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PAIRS_TO_MONITOR, BENCHMARKS, STRATEGY_CONFIG
from src.data_fetcher import DataFetcher
from src.pairs_strategy import PairsStrategy
from src.backtester import Backtester

# Configurazione pagina
st.set_page_config(
    page_title="Pairs Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_pair_data(symbol1: str, symbol2: str, years: int):
    """Carica i dati di una coppia (con cache)"""
    fetcher = DataFetcher()
    return fetcher.get_aligned_prices(symbol1, symbol2, years=years)


@st.cache_data(ttl=3600)
def run_backtest_cached(symbol1: str, symbol2: str, years: int, 
                        entry_z: float, exit_z: float, stop_z: float, lookback: int):
    """Esegue backtest (con cache)"""
    backtester = Backtester()
    return backtester.run_backtest(
        symbol1, symbol2, years=years,
        entry_zscore=entry_z, exit_zscore=exit_z,
        stop_loss_zscore=stop_z, lookback_period=lookback
    )


@st.cache_data(ttl=3600)
def get_benchmark_data(symbol: str, years: int):
    """Carica dati benchmark (con cache)"""
    fetcher = DataFetcher()
    return fetcher.fetch_stock_data(symbol, years=years)


def create_price_chart(prices_df: pd.DataFrame, symbol1: str, symbol2: str) -> go.Figure:
    """Crea grafico prezzi normalizzati"""
    # Normalizza prezzi a 100 all'inizio
    normalized = prices_df / prices_df.iloc[0] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=normalized.index,
        y=normalized[symbol1],
        name=symbol1,
        line=dict(color='#2962FF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=normalized.index,
        y=normalized[symbol2],
        name=symbol2,
        line=dict(color='#FF6D00', width=2)
    ))
    
    fig.update_layout(
        title="Prezzi Normalizzati (Base 100)",
        xaxis_title="Data",
        yaxis_title="Prezzo Normalizzato",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_spread_chart(strategy: PairsStrategy) -> go.Figure:
    """Crea grafico spread e z-score"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Spread", "Z-Score"),
        row_heights=[0.5, 0.5]
    )
    
    # Spread
    fig.add_trace(go.Scatter(
        x=strategy.spread_series.index,
        y=strategy.spread_series,
        name='Spread',
        line=dict(color='#7C4DFF', width=1.5)
    ), row=1, col=1)
    
    # Media mobile spread
    spread_ma = strategy.spread_series.rolling(window=strategy.lookback_period).mean()
    fig.add_trace(go.Scatter(
        x=spread_ma.index,
        y=spread_ma,
        name='Media Mobile',
        line=dict(color='#FF4081', width=1, dash='dash')
    ), row=1, col=1)
    
    # Z-Score
    fig.add_trace(go.Scatter(
        x=strategy.zscore_series.index,
        y=strategy.zscore_series,
        name='Z-Score',
        line=dict(color='#00BFA5', width=1.5)
    ), row=2, col=1)
    
    # Linee soglia
    entry_z = strategy.entry_zscore
    exit_z = strategy.exit_zscore
    
    for threshold, color, name in [
        (entry_z, 'red', f'Entry (+{entry_z})'),
        (-entry_z, 'red', f'Entry (-{entry_z})'),
        (exit_z, 'green', f'Exit (+{exit_z})'),
        (-exit_z, 'green', f'Exit (-{exit_z})'),
        (0, 'gray', 'Zero')
    ]:
        fig.add_hline(
            y=threshold, row=2, col=1,
            line=dict(color=color, width=1, dash='dot'),
            annotation_text=name,
            annotation_position="right"
        )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_equity_chart(result, benchmark_data: dict) -> go.Figure:
    """Crea grafico equity curve vs benchmark"""
    fig = go.Figure()
    
    # Normalizza equity curve a 100
    equity_normalized = result.equity_curve / result.equity_curve.iloc[0] * 100
    
    fig.add_trace(go.Scatter(
        x=equity_normalized.index,
        y=equity_normalized,
        name=f'Pairs Strategy ({result.symbol1}/{result.symbol2})',
        line=dict(color='#2962FF', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(41, 98, 255, 0.1)'
    ))
    
    # Aggiungi benchmark
    colors = ['#FF6D00', '#00C853', '#AA00FF', '#00BCD4']
    for i, (symbol, data) in enumerate(benchmark_data.items()):
        if data is not None and len(data) > 0:
            # Allinea con le date della strategia
            aligned = data['Close'].reindex(equity_normalized.index, method='ffill')
            normalized = aligned / aligned.iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized,
                name=f'{BENCHMARKS.get(symbol, symbol)}',
                line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
            ))
    
    fig.update_layout(
        title="Performance: Strategia vs Benchmark",
        xaxis_title="Data",
        yaxis_title="Valore (Base 100)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_drawdown_chart(result) -> go.Figure:
    """Crea grafico drawdown"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=result.drawdown_series.index,
        y=result.drawdown_series * 100,
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.3)',
        line=dict(color='#FF1744', width=1),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Data",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        height=300,
        showlegend=False
    )
    
    fig.update_yaxes(ticksuffix='%')
    
    return fig


def create_trades_chart(result) -> go.Figure:
    """Crea grafico P&L dei singoli trade"""
    if not result.trades:
        return None
    
    trades_df = pd.DataFrame(result.trades)
    trades_df['trade_num'] = range(1, len(trades_df) + 1)
    trades_df['pnl'] = trades_df.get('net_pnl', trades_df['total_pnl'])
    
    colors = ['#00C853' if pnl > 0 else '#FF1744' for pnl in trades_df['pnl']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=trades_df['trade_num'],
        y=trades_df['pnl'],
        marker_color=colors,
        name='P&L per Trade'
    ))
    
    fig.update_layout(
        title="P&L per Trade",
        xaxis_title="Trade #",
        yaxis_title="P&L ($)",
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


def create_correlation_heatmap(pairs: list) -> go.Figure:
    """Crea heatmap delle correlazioni"""
    fetcher = DataFetcher()
    
    # Raccogli tutti i simboli unici
    symbols = list(set([s for pair in pairs for s in pair]))
    
    # Scarica dati
    data = {}
    for symbol in symbols:
        try:
            df = fetcher.fetch_stock_data(symbol, years=2)
            data[symbol] = df['Close'].pct_change()
        except:
            pass
    
    if len(data) < 2:
        return None
    
    # Crea DataFrame dei rendimenti
    returns_df = pd.DataFrame(data).dropna()
    
    # Calcola matrice correlazione
    corr_matrix = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matrice Correlazioni",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_monthly_returns_heatmap(result) -> go.Figure:
    """Crea heatmap rendimenti mensili"""
    equity = result.equity_curve.copy()
    
    # Converti indice a DatetimeIndex senza timezone
    try:
        if isinstance(equity.index, pd.DatetimeIndex):
            if equity.index.tz is not None:
                equity.index = equity.index.tz_localize(None)
        else:
            # Converti manualmente gestendo timezone
            new_index = []
            for dt in equity.index:
                if hasattr(dt, 'replace'):
                    # È un datetime object, rimuovi tzinfo
                    new_index.append(dt.replace(tzinfo=None))
                else:
                    new_index.append(dt)
            equity.index = pd.DatetimeIndex(new_index)
    except Exception:
        # Fallback: usa l'indice come stringa e riconverti
        equity.index = pd.to_datetime([str(d)[:10] for d in equity.index])
    
    # Calcola rendimenti mensili
    monthly_returns = equity.resample('ME').last().pct_change() * 100
    
    # Pivot per anno/mese
    monthly_returns = monthly_returns.dropna()
    
    if len(monthly_returns) == 0:
        # Ritorna grafico vuoto se non ci sono dati
        fig = go.Figure()
        fig.update_layout(
            title="Rendimenti Mensili (%) - Dati insufficienti",
            height=250,
            template='plotly_white'
        )
        return fig
    
    df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    pivot = df.pivot(index='year', columns='month', values='return')
    
    # Mappa nomi mesi
    month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
                   'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 9},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Rendimenti Mensili (%)",
        height=250,
        template='plotly_white'
    )
    
    return fig


def display_metrics(result, col):
    """Mostra metriche in formato card"""
    with col:
        # Rendimento
        return_color = "positive" if result.total_return > 0 else "negative"
        st.metric(
            "Rendimento Totale",
            f"{result.total_return:.2%}",
            delta=f"{result.annualized_return:.2%} ann."
        )
        
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}"
        )
        
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown:.2%}"
        )
        
        st.metric(
            "Trade Totali",
            f"{result.total_trades}",
            delta=f"Win rate: {result.win_rate:.1%}"
        )


def main():
    """Main function"""
    st.title("📈 Pairs Trading Strategy Dashboard")
    st.markdown("---")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Selezione coppia
        st.subheader("Selezione Coppia")
        
        pair_options = [f"{p[0]} / {p[1]}" for p in PAIRS_TO_MONITOR]
        selected_pair = st.selectbox(
            "Coppia predefinita",
            options=pair_options,
            index=0
        )
        
        # O input manuale
        st.markdown("**Oppure inserisci manualmente:**")
        col1, col2 = st.columns(2)
        with col1:
            custom_symbol1 = st.text_input("Simbolo 1", value="")
        with col2:
            custom_symbol2 = st.text_input("Simbolo 2", value="")
        
        # Usa custom se inseriti, altrimenti predefinita
        if custom_symbol1 and custom_symbol2:
            symbol1, symbol2 = custom_symbol1.upper(), custom_symbol2.upper()
        else:
            idx = pair_options.index(selected_pair)
            symbol1, symbol2 = PAIRS_TO_MONITOR[idx]
        
        st.markdown("---")
        
        # Parametri temporali
        st.subheader("Periodo")
        years = st.slider("Anni di dati", 1, 10, 5)
        
        st.markdown("---")
        
        # Parametri strategia
        st.subheader("Parametri Strategia")
        
        entry_zscore = st.slider(
            "Entry Z-Score",
            1.0, 4.0, float(STRATEGY_CONFIG["entry_zscore"]), 0.1,
            help="Soglia per aprire posizioni"
        )
        
        exit_zscore = st.slider(
            "Exit Z-Score",
            0.0, 2.0, float(STRATEGY_CONFIG["exit_zscore"]), 0.1,
            help="Soglia per chiudere posizioni"
        )
        
        stop_loss_zscore = st.slider(
            "Stop Loss Z-Score",
            2.0, 6.0, float(STRATEGY_CONFIG["stop_loss_zscore"]), 0.5,
            help="Soglia per stop loss"
        )
        
        lookback_period = st.slider(
            "Lookback Period (giorni)",
            20, 120, STRATEGY_CONFIG["lookback_period"], 5,
            help="Finestra per calcolo media/std"
        )
        
        st.markdown("---")
        
        # Benchmark selection
        st.subheader("Benchmark")
        selected_benchmarks = st.multiselect(
            "Seleziona benchmark",
            options=list(BENCHMARKS.keys()),
            default=["SPY", "VT"],
            format_func=lambda x: f"{x} - {BENCHMARKS[x]}"
        )
        
        st.markdown("---")
        
        run_button = st.button("🚀 Esegui Backtest", type="primary", use_container_width=True)
    
    # Main content
    if run_button or 'result' not in st.session_state:
        with st.spinner(f"Caricamento dati {symbol1}/{symbol2}..."):
            try:
                # Carica dati
                prices_df = load_pair_data(symbol1, symbol2, years)
                
                # Inizializza strategia per analisi
                strategy = PairsStrategy(
                    symbol1, symbol2,
                    entry_zscore=entry_zscore,
                    exit_zscore=exit_zscore,
                    stop_loss_zscore=stop_loss_zscore,
                    lookback_period=lookback_period
                )
                strategy.fit(prices_df)
                
                # Esegui backtest
                result = run_backtest_cached(
                    symbol1, symbol2, years,
                    entry_zscore, exit_zscore, stop_loss_zscore, lookback_period
                )
                
                # Carica benchmark
                benchmark_data = {}
                for bench in selected_benchmarks:
                    try:
                        benchmark_data[bench] = get_benchmark_data(bench, years)
                    except:
                        benchmark_data[bench] = None
                
                # Salva in session state
                st.session_state['prices_df'] = prices_df
                st.session_state['strategy'] = strategy
                st.session_state['result'] = result
                st.session_state['benchmark_data'] = benchmark_data
                st.session_state['symbol1'] = symbol1
                st.session_state['symbol2'] = symbol2
                
            except Exception as e:
                st.error(f"Errore: {str(e)}")
                return
    
    # Recupera dati da session state
    if 'result' in st.session_state:
        prices_df = st.session_state['prices_df']
        strategy = st.session_state['strategy']
        result = st.session_state['result']
        benchmark_data = st.session_state['benchmark_data']
        symbol1 = st.session_state['symbol1']
        symbol2 = st.session_state['symbol2']
        
        # Header con metriche principali
        st.header(f"📊 {symbol1} / {symbol2}")
        
        # Analytics coppia
        analytics = strategy.get_analytics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Correlazione", f"{analytics['correlation']:.3f}")
        with col2:
            st.metric("Hedge Ratio", f"{analytics['hedge_ratio']:.3f}")
        with col3:
            coint_status = "✅ Sì" if analytics.get('is_cointegrated') else "❌ No"
            st.metric("Cointegrata?", coint_status)
        with col4:
            current_z = analytics.get('current_zscore', 0)
            st.metric("Z-Score Attuale", f"{current_z:.2f}" if current_z else "N/A")
        with col5:
            st.metric("Periodo", f"{years} anni")
        
        st.markdown("---")
        
        # Tabs per diverse visualizzazioni
        tabs = st.tabs([
            "📈 Performance",
            "🔍 Analisi Spread",
            "💰 Trades",
            "📊 Statistiche",
            "🏆 Confronto Benchmark"
        ])
        
        # Tab 1: Performance
        with tabs[0]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Equity curve
                equity_fig = create_equity_chart(result, benchmark_data)
                st.plotly_chart(equity_fig, use_container_width=True)
                
                # Drawdown
                dd_fig = create_drawdown_chart(result)
                st.plotly_chart(dd_fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Metriche")
                
                st.metric("Capitale Iniziale", f"${result.initial_capital:,.0f}")
                st.metric("Capitale Finale", f"${result.final_capital:,.0f}")
                
                delta_color = "normal" if result.total_return >= 0 else "inverse"
                st.metric(
                    "Rendimento Totale",
                    f"{result.total_return:.2%}",
                    delta=f"${result.final_capital - result.initial_capital:,.0f}",
                    delta_color=delta_color
                )
                
                st.metric("Rendimento Annualizzato", f"{result.annualized_return:.2%}")
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                st.metric("Profit Factor", f"{result.profit_factor:.2f}")
        
        # Tab 2: Analisi Spread
        with tabs[1]:
            # Grafico prezzi
            price_fig = create_price_chart(prices_df, symbol1, symbol2)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Grafico spread e z-score
            spread_fig = create_spread_chart(strategy)
            st.plotly_chart(spread_fig, use_container_width=True)
            
            # Statistiche spread
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Media Spread", f"{strategy.spread_series.mean():.4f}")
            with col2:
                st.metric("Std Spread", f"{strategy.spread_series.std():.4f}")
            with col3:
                st.metric("Z-Score Attuale", f"{strategy.zscore_series.iloc[-1]:.2f}")
        
        # Tab 3: Trades
        with tabs[2]:
            if result.trades:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Grafico P&L trades
                    trades_fig = create_trades_chart(result)
                    if trades_fig:
                        st.plotly_chart(trades_fig, use_container_width=True)
                    
                    # Tabella trades
                    st.subheader("📋 Dettaglio Trades")
                    trades_df = pd.DataFrame(result.trades)
                    
                    # Formatta colonne
                    display_cols = ['entry_date', 'exit_date', 'holding_days', 
                                   'position_type', 'exit_type', 'total_pnl']
                    if 'net_pnl' in trades_df.columns:
                        display_cols.append('net_pnl')
                    
                    trades_display = trades_df[display_cols].copy()
                    # Formatta date (gestisce sia naive che timezone-aware)
                    trades_display['entry_date'] = trades_display['entry_date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10]
                    )
                    trades_display['exit_date'] = trades_display['exit_date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10]
                    )
                    trades_display['total_pnl'] = trades_display['total_pnl'].apply(lambda x: f"${x:,.2f}")
                    if 'net_pnl' in trades_display.columns:
                        trades_display['net_pnl'] = trades_display['net_pnl'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(trades_display, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Statistiche Trade")
                    st.metric("Trades Totali", result.total_trades)
                    st.metric("Trades Vincenti", result.winning_trades)
                    st.metric("Trades Perdenti", result.losing_trades)
                    st.metric("Win Rate", f"{result.win_rate:.1%}")
                    st.metric("Durata Media", f"{result.avg_holding_days:.1f} giorni")
                    
                    if result.trades:
                        pnls = [t.get('net_pnl', t['total_pnl']) for t in result.trades]
                        st.metric("Miglior Trade", f"${max(pnls):,.2f}")
                        st.metric("Peggior Trade", f"${min(pnls):,.2f}")
            else:
                st.info("Nessun trade eseguito nel periodo selezionato.")
        
        # Tab 4: Statistiche
        with tabs[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rendimenti mensili
                monthly_fig = create_monthly_returns_heatmap(result)
                st.plotly_chart(monthly_fig, use_container_width=True)
                
                # Distribuzione rendimenti
                daily_returns = result.equity_curve.pct_change().dropna() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=daily_returns,
                    nbinsx=50,
                    name='Rendimenti Giornalieri',
                    marker_color='#2962FF'
                ))
                fig.update_layout(
                    title="Distribuzione Rendimenti Giornalieri",
                    xaxis_title="Rendimento (%)",
                    yaxis_title="Frequenza",
                    template='plotly_white',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistiche dettagliate
                st.subheader("📈 Statistiche Dettagliate")
                
                stats_data = {
                    "Metrica": [
                        "Rendimento Totale",
                        "Rendimento Annualizzato",
                        "Volatilità Annualizzata",
                        "Sharpe Ratio",
                        "Max Drawdown",
                        "Rendimento/Drawdown",
                        "Trade Totali",
                        "Win Rate",
                        "Profit Factor",
                        "Durata Media Trade"
                    ],
                    "Valore": [
                        f"{result.total_return:.2%}",
                        f"{result.annualized_return:.2%}",
                        f"{daily_returns.std() * np.sqrt(252):.2%}",
                        f"{result.sharpe_ratio:.2f}",
                        f"{result.max_drawdown:.2%}",
                        f"{abs(result.total_return / result.max_drawdown):.2f}" if result.max_drawdown != 0 else "N/A",
                        f"{result.total_trades}",
                        f"{result.win_rate:.1%}",
                        f"{result.profit_factor:.2f}",
                        f"{result.avg_holding_days:.1f} giorni"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                # Correlazione matrice se multiple coppie
                st.subheader("🔗 Correlazioni")
                corr_fig = create_correlation_heatmap(PAIRS_TO_MONITOR[:4])
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
        
        # Tab 5: Confronto Benchmark
        with tabs[4]:
            st.subheader("🏆 Confronto con Benchmark")
            
            # Tabella confronto
            comparison_data = {
                "Strategia/Benchmark": [f"Pairs {symbol1}/{symbol2}"],
                "Rendimento Totale": [f"{result.total_return:.2%}"],
                "Rend. Annualizzato": [f"{result.annualized_return:.2%}"],
                "Sharpe Ratio": [f"{result.sharpe_ratio:.2f}"],
                "Max Drawdown": [f"{result.max_drawdown:.2%}"]
            }
            
            for bench_symbol, bench_df in benchmark_data.items():
                if bench_df is not None and len(bench_df) > 0:
                    bench_returns = bench_df['Close'].pct_change().dropna()
                    total_ret = (bench_df['Close'].iloc[-1] / bench_df['Close'].iloc[0]) - 1
                    
                    days = (bench_df.index[-1] - bench_df.index[0]).days
                    ann_ret = (1 + total_ret) ** (365.25/days) - 1 if days > 0 else 0
                    
                    sharpe = (bench_returns.mean() / bench_returns.std()) * np.sqrt(252) if bench_returns.std() > 0 else 0
                    
                    rolling_max = bench_df['Close'].expanding().max()
                    drawdown = (bench_df['Close'] - rolling_max) / rolling_max
                    max_dd = drawdown.min()
                    
                    comparison_data["Strategia/Benchmark"].append(BENCHMARKS.get(bench_symbol, bench_symbol))
                    comparison_data["Rendimento Totale"].append(f"{total_ret:.2%}")
                    comparison_data["Rend. Annualizzato"].append(f"{ann_ret:.2%}")
                    comparison_data["Sharpe Ratio"].append(f"{sharpe:.2f}")
                    comparison_data["Max Drawdown"].append(f"{max_dd:.2%}")
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Evidenzia il migliore
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Grafici comparativi
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart rendimenti
                returns_data = []
                returns_data.append({
                    "name": f"Pairs Strategy",
                    "return": result.total_return * 100
                })
                
                for bench_symbol, bench_df in benchmark_data.items():
                    if bench_df is not None and len(bench_df) > 0:
                        total_ret = (bench_df['Close'].iloc[-1] / bench_df['Close'].iloc[0]) - 1
                        returns_data.append({
                            "name": BENCHMARKS.get(bench_symbol, bench_symbol),
                            "return": total_ret * 100
                        })
                
                returns_df = pd.DataFrame(returns_data)
                
                colors = ['#2962FF'] + ['#BDBDBD'] * (len(returns_df) - 1)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=returns_df['name'],
                    y=returns_df['return'],
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in returns_df['return']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Rendimento Totale (%)",
                    yaxis_title="Rendimento %",
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart Sharpe
                sharpe_data = []
                sharpe_data.append({
                    "name": f"Pairs Strategy",
                    "sharpe": result.sharpe_ratio
                })
                
                for bench_symbol, bench_df in benchmark_data.items():
                    if bench_df is not None and len(bench_df) > 0:
                        bench_returns = bench_df['Close'].pct_change().dropna()
                        sharpe = (bench_returns.mean() / bench_returns.std()) * np.sqrt(252) if bench_returns.std() > 0 else 0
                        sharpe_data.append({
                            "name": BENCHMARKS.get(bench_symbol, bench_symbol),
                            "sharpe": sharpe
                        })
                
                sharpe_df = pd.DataFrame(sharpe_data)
                
                colors = ['#2962FF'] + ['#BDBDBD'] * (len(sharpe_df) - 1)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sharpe_df['name'],
                    y=sharpe_df['sharpe'],
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in sharpe_df['sharpe']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Sharpe Ratio",
                    yaxis_title="Sharpe",
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Conclusioni
            st.markdown("---")
            st.subheader("📝 Conclusioni")
            
            # Calcola outperformance
            if benchmark_data:
                spy_data = benchmark_data.get('SPY')
                if spy_data is not None and len(spy_data) > 0:
                    spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1
                    outperformance = result.total_return - spy_return
                    
                    if outperformance > 0:
                        st.success(f"""
                        ✅ **La strategia Pairs Trading ha sovraperformato l'S&P 500 di {outperformance:.2%}**
                        
                        - Rendimento Strategia: {result.total_return:.2%}
                        - Rendimento S&P 500: {spy_return:.2%}
                        - Sharpe Ratio: {result.sharpe_ratio:.2f}
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **La strategia Pairs Trading ha sottoperformato l'S&P 500 di {abs(outperformance):.2%}**
                        
                        - Rendimento Strategia: {result.total_return:.2%}
                        - Rendimento S&P 500: {spy_return:.2%}
                        
                        Considera di modificare i parametri o testare altre coppie.
                        """)


if __name__ == "__main__":
    main()
