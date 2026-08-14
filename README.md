# AI/ML Stock Market Screening and Signal Analysis System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost Engine](https://img.shields.io/badge/ML%20Engine-XGBoost-111111?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An enterprise-grade quantitative stock market screening, live tick ingestion, and AI-driven signal analysis platform built in Python. Designed for real-time NSE stock screening, Smoothed Moving Average (SMMA) crossover detection, rolling Exchange Traded Quantity (ETQ) volume calculation, market depth tracking, and ML signal validation with human-readable explainability.

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [System Architecture](#-system-architecture)
- [Core Functional Specifications](#-core-functional-specifications)
- [Prerequisites & System Requirements](#-prerequisites--system-requirements)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Environment Configuration (.env)](#-environment-configuration-env)
- [Running the Application](#-running-the-application)
- [Broker Ingestion Setup](#-broker-ingestion-setup)
- [AI/ML Model & Quantitative Feature Engineering](#-aiml-model--quantitative-feature-engineering)
- [Automated Testing & Build Scripts](#-automated-testing--build-scripts)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)

---

## 📌 Executive Summary

The **AI/ML Stock Screening System** continuously monitors real-time NSE tick data feeds, filters out illiquid stocks, computes rolling indicators, and evaluates technical crossover signals. Signals are passed to a trained **XGBoost machine learning classifier** that evaluates market micro-structure (e.g., LTQ ratios, order book bid/ask imbalance, price momentum) to classify signals as `ACCEPT BUY`, `ACCEPT SELL`, or `AVOID / REJECT`.

---

## 📂 System Architecture

```
Stock Screening System/
├── .env.example                # Template for environment variables and API credentials
├── .gitignore                  # Git exclusion rules
├── Dockerfile                  # Production container definition
├── pyproject.toml              # Project dependencies and tool configurations
├── requirements.txt            # Python package requirements
├── StockScreeningSystem.spec   # PyInstaller build specification
│
├── config/
│   └── settings.py             # System thresholds (Price, Liquidity, Indicator Periods)
│
├── src/
│   ├── dashboard/
│   │   ├── app.py              # Streamlit dashboard interface
│   │   └── style.css           # Dark glassmorphism UI stylesheet
│   ├── data/
│   │   ├── broker_adapters.py  # Connectors for Angel One, Fyers API v3, WebSocket, & CSV
│   │   ├── data_manager.py     # Thread-safe rolling memory buffer for time-series ticks
│   │   └── sample_generator.py # Real-time tick stream emulator for offline testing
│   ├── indicators/
│   │   ├── screener.py         # Universe filter (Price ₹30–₹500 & Bid/Ask Qty > 10L)
│   │   └── technical.py        # Technical formulas: SMMA(20), SMMA(120), ETQ, Avg Price
│   └── ml/
│       ├── feature_extractor.py# Feature engineering (LTQ 2m/5m ratio, momentum, spread)
│       ├── predictor.py        # Inference engine with confidence scoring & reasoning
│       └── trainer.py          # ML pipeline trainer for model retraining
│
├── models/
│   ├── smma_crossover_model.pkl# Pre-trained XGBoost classification model
│   └── feature_importance.json # Model feature importance weights
│
├── data/
│   ├── historical/             # Parquet datasets for historical analysis & training
│   ├── instruments/            # NSE symbol master definition files
│   └── sample/                 # Offline sample tick datasets (.csv)
│
├── scripts/
│   ├── build_exe.py            # PyInstaller binary compiler script
│   ├── download_instruments.py # Utility to sync symbol master from broker API
│   ├── generate_sample_csv.py  # Utility to generate test tick data
│   └── train_model.py          # CLI script to train/retrain ML model
│
├── tests/
│   └── test_screener_indicators.py # Automated unit test suite
│
├── run_dashboard.bat           # 1-Click launcher for Streamlit Dashboard
├── run_train_model.bat         # 1-Click launcher for ML Model Retraining
└── run_tests.bat               # 1-Click launcher for Unit Tests
```

---

## 🎯 Core Functional Specifications

| Specification | Operational Rule / Formula | Status |
| :--- | :--- | :---: |
| **Price Screener** | Filters universe to stocks with Last Traded Price (LTP) between **₹30.00** and **₹500.00**. | ✅ Active |
| **Liquidity Screener** | Requires **Total Bid Quantity > 10,00,000** AND **Total Ask Quantity > 10,00,000**. | ✅ Active |
| **Fast Moving Average** | **SMMA (20)** — Smoothed Moving Average over 20 rolling tick periods. | ✅ Active |
| **Slow Moving Average** | **SMMA (120)** — Smoothed Moving Average over 120 rolling tick periods. | ✅ Active |
| **ETQ Volume** | Aggregates Exchange Traded Quantity over rolling windows: **5m**, **20m**, and **60m**. | ✅ Active |
| **Average LTP** | Calculates average LTP over rolling windows: **20m** and **60m**. | ✅ Active |
| **Order Book Depth** | Real-time 5-level Bid/Ask price and quantity depth breakdown. | ✅ Active |
| **ML Signal Classifier** | Predicts crossover validity (`ACCEPT BUY`, `ACCEPT SELL`, `AVOID`) with confidence score & explainability. | ✅ Active |

---

## 💻 Prerequisites & System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 12+
- **Python**: Version `3.11` or higher
- **RAM**: 4 GB minimum (8 GB recommended for streaming live feeds)
- **Disk Space**: 500 MB for repository and environment

---

## 📥 Installation & Environment Setup

### Step 1: Clone or Open Project
Navigate to the root directory of the project:
```bash
cd "Stock Screening System"
```

### Step 2: Create Virtual Environment
It is strongly recommended to use an isolated Python virtual environment:

**Windows (PowerShell/CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
*(Or install via `pyproject.toml` using `pip install -e .`)*

---

## 🔑 Environment Configuration (.env)

Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to configure your API credentials and preferences:

```ini
# Broker API Credentials (Optional - Required only for live WebSocket feeds)
ANGEL_API_KEY=your_angel_one_api_key
ANGEL_CLIENT_CODE=your_client_code
ANGEL_PASSWORD=your_pin_or_password
ANGEL_TOTP_KEY=your_totp_secret_key

FYERS_APP_ID=your_fyers_app_id
FYERS_ACCESS_TOKEN=your_fyers_access_token

# System Environment Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DASHBOARD_PORT=8501
```

---

## 🚀 Running the Application

### Method A: One-Click Windows Launchers (Recommended for Windows)
Double-click any of the provided `.bat` scripts in the root directory:
- **`run_dashboard.bat`**: Launches the Streamlit Dashboard web app.
- **`run_train_model.bat`**: Retrains the XGBoost ML model on current historical data.
- **`run_tests.bat`**: Executes the automated test suite.

### Method B: Command Line Interface (CLI)

#### 1. Start Dashboard
```bash
streamlit run src/dashboard/app.py
```
The application will start and automatically open in your browser at `http://localhost:8501`.

#### 2. Retrain ML Model
```bash
python scripts/train_model.py
```

#### 3. Run Unit Tests
```bash
python -m unittest tests/test_screener_indicators.py
```

---

## 📡 Broker Ingestion Setup

The platform supports multiple data feed sources via `src/data/broker_adapters.py`:

1. **Simulated Live Stream (Default)**: Generates realistic NSE tick data for offline development and UI testing.
2. **Angel One SmartAPI**: Streams live ticks via WebSocket (`SmartWebSocketV2`). Requires `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_PASSWORD`, and `ANGEL_TOTP_KEY`.
3. **Fyers API v3**: Connects to Fyers Data Socket for real-time tick streaming.
4. **CSV Tick Upload**: Allows uploading offline tick CSV logs for backtesting and replay.

Select your desired feed source from the **"Data & Feed Configuration"** sidebar inside the Streamlit dashboard.

---

## 🤖 AI/ML Model & Quantitative Feature Engineering

### Signal Logic
- **BUY Signal**: SMMA(20) crosses **above** SMMA(120).
- **SELL Signal**: SMMA(20) crosses **below** SMMA(120).

### Feature Set
The XGBoost model processes the following features to validate whether a signal should be accepted or rejected:

1. **LTQ Ratio ($2m / 5m$)**: Ratio of average Last Traded Quantity over 2 minutes vs 5 minutes:
   $$\text{LTQ Ratio} = \frac{\text{Mean}(\text{LTQ}_{2m})}{\text{Mean}(\text{LTQ}_{5m})}$$
   *A sudden spike ($> 1.2$) signals aggressive market participant entry.*
2. **Order Book Imbalance**: Ratio of total bid depth vs ask depth:
   $$\text{Imbalance} = \frac{\text{Bid Quantity}}{\text{Bid Quantity} + \text{Ask Quantity}}$$
3. **Price Momentum (20m)**: Percentage deviation of LTP relative to 20-minute average:
   $$\text{Momentum}_{20m} = \frac{\text{LTP} - \text{Avg LTP}_{20m}}{\text{Avg LTP}_{20m}}$$
4. **SMMA Spread %**: Distance between fast and slow moving averages:
   $$\text{Spread \%} = \frac{\text{SMMA}_{20} - \text{SMMA}_{120}}{\text{SMMA}_{120}} \times 100$$

### Model Output & Explainability
- **Decision**: `ACCEPT BUY`, `ACCEPT SELL`, or `AVOID / REJECT`.
- **Confidence**: Model probability score (e.g., `88.5%`).
- **Reasoning**: Human-readable explanation of key factors driving the model decision (e.g., *"Strong LTQ surge (+42% 2m vs 5m), Bullish momentum (+0.85%), Buyer order book dominance (62% bids)"*).

---

## 🧪 Automated Testing & Build Scripts

### Running Tests
Execute the unit test suite to verify indicator calculations, universe screener logic, and model inference:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Packaging Standalone Executable (.exe)
To compile a standalone Windows `.exe` application without needing Python installed on the target machine:
```bash
python scripts/build_exe.py
```
The compiled application will be generated in `dist/StockScreeningSystem.exe`.

---

## 🐳 Docker Deployment

To build and run the application in a Docker container:

### 1. Build Image
```bash
docker build -t stock-screening-system .
```

### 2. Run Container
```bash
docker run -d -p 8501:8501 --env-file .env --name stock-screener stock-screening-system
```
Access the dashboard at `http://localhost:8501`.

---

## 🛠️ Troubleshooting

- **Missing Module `dotenv`**: Run `pip install python-dotenv` or ensure `.venv` is activated.
- **Port 8501 Already in Use**: Specify a custom port when running Streamlit:
  ```bash
  streamlit run src/dashboard/app.py --server.port 8502
  ```
- **Broker Live Stream Fails**: Verify your TOTP key and API credentials in `.env`. Ensure market hours (09:15 to 15:30 IST) are active for live feeds.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
