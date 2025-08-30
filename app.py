import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for real-time refresh
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 60  # default 60 seconds

# Add helper function to determine currency symbol based on ticker
def get_currency_symbol(ticker):
    """Return the appropriate currency symbol based on ticker"""
    if ticker.endswith(('.NS', '.BO')):  # Indian stock exchanges (NSE, BSE)
        return '₹'
    return '$'  # Default to USD

class EnhancedStockPredictor:
    def __init__(self, ticker, start_date='2020-01-01', end_date=None):
        """Enhanced stock price predictor with multiple models and features"""
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = -np.inf
        self.real_time_price = None
        self.last_update_time = None
        
    def fetch_data(self):
        """Fetch comprehensive stock data"""
        try:
            # Fetch stock data
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                return None
                
            # Get additional info
            self.info = stock.info
            self.data.columns = [col.lower().replace(' ', '_') for col in self.data.columns]
            
            # Determine currency for display
            self.currency_symbol = get_currency_symbol(self.ticker)
            
            # Get real-time price data if available
            try:
                real_time_data = stock.history(period='1d', interval='1m').iloc[-1]
                if not real_time_data.empty:
                    self.real_time_price = real_time_data['Close']
                    self.last_update_time = datetime.now()
            except:
                pass
                
            return self.data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def advanced_feature_engineering(self):
        """Create comprehensive technical indicators"""
        if self.data is None or len(self.data) < 50:
            raise ValueError("Insufficient data for feature engineering")
        
        df = self.data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) > period:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            df['vwap'] = (df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum())
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Lagged features
        for lag in range(1, 8):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Seasonal features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        self.data = df.dropna()
        return self.data
    
    def prepare_features(self, target_col='close', look_ahead=1):
        """Prepare feature matrix and target variable"""
        # Select features (exclude target and non-predictive columns)
        exclude_cols = [target_col, 'open', 'high', 'low', 'dividends', 'stock_splits']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols].values
        y = self.data[target_col].shift(-look_ahead).dropna().values
        
        # Align X with shifted y
        X = X[:-look_ahead if look_ahead > 0 else len(X)]
        
        return X, y, feature_cols
    
    def train_multiple_models(self, X_train, y_train):
        """Train multiple ML models and select the best one"""
        models_config = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models_config.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                avg_score = np.mean(cv_scores)
                
                # Train on full training set
                model.fit(X_train, y_train)
                self.models[name] = model
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
                    self.best_model = model
                
                st.write(f"✅ {name}: CV R² = {avg_score:.3f} (±{np.std(cv_scores):.3f})")
                
            except Exception as e:
                st.write(f"❌ {name}: Training failed - {str(e)}")
        
        self.best_score = best_score
        st.success(f"🏆 Best model: {best_model_name} (R² = {best_score:.3f})")
        return best_model_name
    
    def make_predictions(self, X_test, days_ahead=30):
        """Make predictions and forecast future prices"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test)
                predictions[name] = pred
            except:
                continue
        
        # Future predictions (simplified approach)
        if self.best_model and len(self.data) > 0:
            last_features = X_test[-1:] if len(X_test) > 0 else self.data.iloc[-1:].drop(['close'], axis=1).values
            future_predictions = []
            
            for _ in range(days_ahead):
                pred = self.best_model.predict(last_features.reshape(1, -1))[0]
                future_predictions.append(pred)
                # Simple feature update (in practice, this would be more sophisticated)
                if len(last_features[0]) > 0:
                    last_features[0][-1] = pred
        
        return predictions, future_predictions if 'future_predictions' in locals() else []

def create_advanced_charts(data, predictions, ticker):
    """Create comprehensive visualization dashboard"""
    
    # Main price chart with predictions
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=('Stock Price & Predictions', 'Volume', 'RSI', 'MACD', 
                       'Bollinger Bands', 'Returns Distribution', 'Correlation Heatmap', 'Model Comparison'),
        specs=[[{"colspan": 2}, None],
               [{"colspan": 2}, None],
               [{}, {}],
               [{}, {}]],
        vertical_spacing=0.08
    )
    
    # Price and predictions
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Actual Price', 
                            line=dict(color='blue', width=2)), row=1, col=1)
    
    if 'Random Forest' in predictions:
        # Add prediction line (simplified)
        pred_index = data.index[-len(predictions['Random Forest']):]
        fig.add_trace(go.Scatter(x=pred_index, y=predictions['Random Forest'], 
                                name='RF Predictions', line=dict(color='red', width=2)), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', 
                        marker_color='lightblue'), row=2, col=1)
    
    # RSI
    if 'rsi' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name='RSI', 
                                line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD', 
                                line=dict(color='blue')), row=3, col=2)
        fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal', 
                                line=dict(color='red')), row=3, col=2)
    
    fig.update_layout(height=1200, title_text=f"{ticker} - Comprehensive Analysis Dashboard")
    return fig

def calculate_portfolio_metrics(data, ticker):
    """Calculate advanced portfolio metrics"""
    returns = data['returns'].dropna()
    
    # Determine currency symbol
    currency = get_currency_symbol(ticker)
    
    metrics = {
        'Total Return': f"{((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%",
        'Annualized Return': f"{(returns.mean() * 252) * 100:.2f}%",
        'Volatility (Annual)': f"{(returns.std() * np.sqrt(252)) * 100:.2f}%",
        'Sharpe Ratio': f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.3f}",
        'Max Drawdown': f"{((data['close'] / data['close'].cummax() - 1).min()) * 100:.2f}%",
        'Current Price': f"{currency}{data['close'].iloc[-1]:.2f}",
        'Volatility (30d)': f"{returns.tail(30).std() * np.sqrt(252) * 100:.2f}%"
    }
    
    return metrics

# Enhanced UI
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .real-time-badge {
        background-color: #22c55e;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.4rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced AI Stock Predictor</h1>', unsafe_allow_html=True)
    
    # Add real-time data indicator
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown("### *Powered by Multiple ML Models & Advanced Technical Analysis*")
    with col2:
        time_now = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="text-align: right;">
            <span style="color: #888;">Last update: {time_now}</span>
            <span class="real-time-badge">LIVE</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Real-time data settings
        st.markdown("### ⚡ Real-Time Settings")
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh data", value=st.session_state.auto_refresh)
        refresh_options = {
            "30 sec": 30, 
            "1 min": 60, 
            "2 min": 120, 
            "5 min": 300
        }
        selected_refresh = st.selectbox(
            "Refresh interval:", 
            list(refresh_options.keys()),
            index=1  # Default to 1 minute
        )
        st.session_state.refresh_interval = refresh_options[selected_refresh]
        
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh_time = datetime.now()
            st.rerun()
        
        # Stock selection with autocomplete-like interface
        st.markdown("### 📊 Stock Selection")
        
        # Popular categories
        categories = {
            "🔥 Trending": ["TSLA", "NVDA", "AMD", "PLTR", "GME"],
            "💼 Blue Chip": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "🏦 Financial": ["JPM", "BAC", "GS", "MS", "C"],
            "⚡ Tech": ["AAPL", "MSFT", "GOOGL", "META", "NFLX"],
            "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABT", "TMO"],
            "🇮🇳 Indian Markets": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        }
        
        selected_category = st.selectbox("Choose Category:", list(categories.keys()))
        
        col1, col2 = st.columns(2)
        for i, stock in enumerate(categories[selected_category]):
            if (i % 2 == 0 and col1.button(f"📈 {stock}", key=f"cat_{stock}")) or \
               (i % 2 == 1 and col2.button(f"📈 {stock}", key=f"cat_{stock}")):
                st.session_state.selected_ticker = stock
        
        # Manual input
        ticker = st.text_input("🎯 Or Enter Ticker:", 
                              value=getattr(st.session_state, 'selected_ticker', 'AAPL'),
                              placeholder="e.g., AAPL, TSLA, NVDA")
        
        st.markdown("### 📅 Analysis Period")
        
        # Preset time ranges
        time_ranges = {
            "📅 1 Year": 365,
            "📅 2 Years": 730,
            "📅 3 Years": 1095,
            "📅 5 Years": 1825
        }
        
        selected_range = st.selectbox("Quick Select:", list(time_ranges.keys()))
        days_back = time_ranges[selected_range]
        
        start_date = st.date_input("Start Date", 
                                  value=datetime.now() - timedelta(days=days_back))
        end_date = st.date_input("End Date", value=datetime.now())
        
        st.markdown("### 🤖 Model Configuration")
        enable_future_forecast = st.checkbox("🔮 Enable Future Forecasting", value=True)
        forecast_days = st.slider("Forecast Days:", 1, 60, 30) if enable_future_forecast else 0
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            scaling_method = st.selectbox("Scaling Method:", ["StandardScaler", "MinMaxScaler"])
            test_size = st.slider("Test Size (%):", 10, 40, 20) / 100
            cross_validation = st.checkbox("Enable Cross-Validation", value=True)
        
        # Analysis button
        analyze_btn = st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True)
    
    # Main content
    if not analyze_btn:
        # Welcome dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 AI-Powered Models</h3>
                <p>• Random Forest</p>
                <p>• Gradient Boosting</p>
                <p>• Support Vector Regression</p>
                <p>• Linear Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Advanced Features</h3>
                <p>• 25+ Technical Indicators</p>
                <p>• Portfolio Metrics</p>
                <p>• Risk Analysis</p>
                <p>• Future Forecasting</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Rich Visualizations</h3>
                <p>• Interactive Charts</p>
                <p>• Multiple Timeframes</p>
                <p>• Correlation Analysis</p>
                <p>• Performance Metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Market overview with real-time data
        st.markdown("### 📈 Real-Time Market Overview")

        # Show last refresh time and next refresh
        time_since_refresh = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        next_refresh = max(0, st.session_state.refresh_interval - time_since_refresh)

        refresh_col1, refresh_col2 = st.columns(2)
        with refresh_col1:
            st.markdown(f"Last updated: **{st.session_state.last_refresh_time.strftime('%H:%M:%S')}**")
        with refresh_col2:
            if st.session_state.auto_refresh:
                st.markdown(f"Next refresh in: **{int(next_refresh)}s**")

        # Define market indices to track (US + Indian)
        sample_tickers = ["^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN"]
        sample_names = ["S&P 500", "Dow Jones", "NASDAQ", "Nifty 50", "Sensex"]

        try:
            market_data = []
            for ticker_sym, name in zip(sample_tickers, sample_names):
                # Use interval='1m' for most recent data when markets are open
                data = yf.download(ticker_sym, period="2d", interval="1m")
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[0]  # Previous day close
                    change = ((current - prev_close) / prev_close) * 100
                    market_time = data.index[-1].strftime("%H:%M:%S")
                    market_data.append({
                        "Index": name, 
                        "Price": f"{current:.2f}", 
                        "Change": f"{change:+.2f}%",
                        "Last Trade": market_time
                    })
            
            if market_data:
                df_market = pd.DataFrame(market_data)
                
                # Apply conditional formatting
                def color_change(val):
                    if '+' in val:
                        return 'color: green; font-weight: bold'
                    else:
                        return 'color: red; font-weight: bold'
                    
                # Display with formatting
                st.dataframe(
                    df_market.style.applymap(color_change, subset=['Change']), 
                    use_container_width=True,
                    height=220
                )
        except Exception as e:
            st.info(f"Market data temporarily unavailable: {e}")

        # Add real-time stock quotes for selected tickers
        if not analyze_btn:
            st.markdown("### 💹 Real-Time Stock Tracker")
            
            # Tracking popular stocks from all categories
            track_tickers = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
            track_cols = st.columns(3)
            
            for i, ticker_sym in enumerate(track_tickers):
                col_idx = i % 3
                with track_cols[col_idx]:
                    try:
                        stock = yf.Ticker(ticker_sym)
                        # Get intraday data
                        live_data = stock.history(period='1d', interval='1m')
                        if not live_data.empty:
                            current = live_data['Close'].iloc[-1]
                            open_price = live_data['Open'].iloc[0]
                            change = ((current - open_price) / open_price) * 100
                            volume = live_data['Volume'].sum()
                            
                            # Format color based on price movement
                            color = "green" if change >= 0 else "red"
                            arrow = "▲" if change >= 0 else "▼"
                            
                            # Get correct currency symbol
                            currency = get_currency_symbol(ticker_sym)
                            
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                                <h3 style="margin: 0; color: #333;">{ticker_sym}</h3>
                                <h2 style="margin: 5px 0; color: {color};">{currency}{current:.2f} {arrow}</h2>
                                <p style="margin: 0; color: {color};">{change:+.2f}% today</p>
                                <p style="margin: 0; color: #666; font-size: 0.8rem;">Vol: {volume:,.0f}</p>
                                <p style="margin: 0; color: #888; font-size: 0.7rem;">Last: {live_data.index[-1].strftime('%H:%M:%S')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{ticker_sym}**: Data unavailable")
                    except Exception as e:
                        st.markdown(f"**{ticker_sym}**: Data unavailable")

        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time_elapsed = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
            if time_elapsed >= st.session_state.refresh_interval:
                st.session_state.last_refresh_time = datetime.now()
                st.rerun()
    
    else:
        # Run analysis
        predictor = EnhancedStockPredictor(ticker, 
                                         start_date=start_date.strftime('%Y-%m-%d'),
                                         end_date=end_date.strftime('%Y-%m-%d'))
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Data fetching
            status_text.text("📡 Fetching comprehensive stock data...")
            progress_bar.progress(15)
            
            data = predictor.fetch_data()
            if data is None:
                st.error("❌ Failed to fetch data. Please check the ticker symbol.")
                return
            
            # Step 2: Feature engineering
            status_text.text("🔧 Creating advanced technical indicators...")
            progress_bar.progress(30)
            predictor.advanced_feature_engineering()
            
            # Step 3: Data preparation
            status_text.text("📊 Preparing machine learning features...")
            progress_bar.progress(45)
            X, y, feature_names = predictor.prepare_features()
            
            # Scaling
            scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False
            )
            
            # Step 4: Model training
            status_text.text("🤖 Training multiple AI models...")
            progress_bar.progress(70)
            best_model_name = predictor.train_multiple_models(X_train, y_train)
            
            # Step 5: Predictions
            status_text.text("🎯 Generating predictions and forecasts...")
            progress_bar.progress(90)
            predictions, future_forecast = predictor.make_predictions(X_test, forecast_days)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress
            progress_container.empty()
            
            # Results dashboard
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎉 Analysis Complete for {ticker.upper()}</h2>
                <p><strong>Best Model:</strong> {best_model_name} | <strong>Accuracy:</strong> {predictor.best_score:.3f}</p>
                <p><strong>Data Points:</strong> {len(data):,} | <strong>Features:</strong> {len(feature_names)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.markdown("### 📊 Portfolio Analytics")
            metrics = calculate_portfolio_metrics(predictor.data, ticker)
            
            cols = st.columns(4)
            metric_items = list(metrics.items())
            for i, (key, value) in enumerate(metric_items[:4]):
                cols[i].metric(key, value)
            
            cols2 = st.columns(3)
            for i, (key, value) in enumerate(metric_items[4:]):
                if i < 3:
                    cols2[i].metric(key, value)
            
            # Model performance comparison
            st.markdown("### 🏆 Model Performance Comparison")
            performance_data = []
            for name, model in predictor.models.items():
                try:
                    pred = model.predict(X_test)
                    r2 = r2_score(y_test, pred)
                    mse = mean_squared_error(y_test, pred)
                    mae = mean_absolute_error(y_test, pred)
                    performance_data.append({
                        "Model": name,
                        "R² Score": f"{r2:.3f}",
                        "MSE": f"{mse:.2f}",
                        "MAE": f"${mae:.2f}",
                        "Status": "🏆 Best" if name == best_model_name else "✅ Good"
                    })
                except:
                    continue
            
            if performance_data:
                st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
            
            # Advanced visualizations
            st.markdown("### 📈 Advanced Analytics Dashboard")
            
            # Create comprehensive charts
            try:
                chart_fig = create_advanced_charts(predictor.data, predictions, ticker)
                st.plotly_chart(chart_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Advanced charts unavailable: {e}")
            
            # Future forecast
            if enable_future_forecast and future_forecast:
                st.markdown("### 🔮 Future Price Forecast")
                
                forecast_dates = pd.date_range(start=predictor.data.index[-1] + timedelta(days=1), 
                                             periods=len(future_forecast))
                
                fig_forecast = go.Figure()
                
                # Historical prices
                fig_forecast.add_trace(go.Scatter(
                    x=predictor.data.index[-100:], 
                    y=predictor.data['close'].iloc[-100:],
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_dates, 
                    y=future_forecast,
                    name=f'{forecast_days}-Day Forecast',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                currency = predictor.currency_symbol  # Get the currency symbol
                
                fig_forecast.update_layout(
                    title=f"{ticker} - Price Forecast",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency})",
                    height=400
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast summary with correct currency
                current_price = predictor.data['close'].iloc[-1]
                forecast_price = future_forecast[-1]
                price_change = ((forecast_price - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"{currency}{current_price:.2f}")
                col2.metric(f"Forecasted Price ({forecast_days}d)", f"{currency}{forecast_price:.2f}")
                col3.metric("Expected Change", f"{price_change:+.2f}%")
            
            # Feature importance (if available)
            if hasattr(predictor.best_model, 'feature_importances_'):
                st.markdown("### 🎯 Feature Importance Analysis")
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': predictor.best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 15 Most Important Features"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Data explorer
            with st.expander("🔍 Data Explorer"):
                st.markdown("#### Recent Data")
                st.dataframe(predictor.data.tail(20), use_container_width=True)
                
                st.markdown("#### Feature Statistics")
                st.dataframe(predictor.data.describe(), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_container.empty()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>⚠️ <strong>Investment Disclaimer:</strong> This tool is for educational and research purposes only. 
        Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.</p>
        <p>🚀 Built with advanced machine learning algorithms and comprehensive technical analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()