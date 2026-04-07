"""
ValueBetting Configuration
---------------------------
Pydantic-based settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings


import pathlib

_env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = {"env_file": str(_env_file), "env_file_encoding": "utf-8"}

    # === Trading Mode ===
    LIVE_MODE: bool = False

    # === Polymarket ===
    POLYMARKET_CLOB_URL: str = "https://clob.polymarket.com"
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_API_SECRET: str = ""
    POLYMARKET_PASSPHRASE: str = ""
    POLYMARKET_PRIVATE_KEY: str = ""
    POLYMARKET_PROXY_WALLET: str = ""
    CHAIN_ID: int = 137  # Polygon

    # === External APIs ===
    ODDS_API_KEY: str = ""
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
    ANTHROPIC_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    CRYPTOPANIC_API_KEY: str = ""
    LUNARCRUSH_API_KEY: str = ""
    COINGECKO_API_KEY: str = ""
    KALSHI_API_KEY_ID: str = ""
    KALSHI_PRIVATE_KEY_PATH: str = ""

    # === ACLED (Armed Conflict Data) ===
    ACLED_API_KEY: str = ""            # Free at https://acleddata.com/
    ACLED_EMAIL: str = ""

    # === Edge & Entry ===
    MIN_EDGE_PCT: float = 0.01         # 1% min edge — ultra-aggressive for paper training
    EXIT_EDGE_PCT: float = 0.005       # Exit at 0.5% edge — take profits fast
    MIN_CONSENSUS_CONFIDENCE: float = 0.10  # Very low for paper — let trades flow
    MIN_SOURCES: int = 1               # 1 source OK for paper training
    MAX_SPREAD: float = 0.10           # Relaxed for paper training
    MIN_ORDERBOOK_DEPTH: float = 0.0   # Ignore depth for paper
    SLIPPAGE_BUDGET: float = 0.00      # Zero for paper training
    FEE_RATE: float = 0.00             # Zero for paper training (restore to 0.015 for live)

    # === Position Sizing (Corrected Kelly) ===
    INITIAL_BANKROLL: float = 500.0
    KELLY_FRACTION: float = 0.25       # Quarter-Kelly
    MAX_BET_PCT: float = 0.01          # $5 max bet (1% of $500)
    MIN_POSITION_USD: float = 2.0      # Low minimum for paper

    # === Risk Management ===
    MAX_CONCURRENT_POSITIONS: int = 20  # More positions for training data
    MAX_TRADES_PER_SESSION: int = 100   # High cap for training
    MAX_CATEGORY_EXPOSURE_PCT: float = 0.40  # No single category > 40%
    DAILY_LOSS_LIMIT_PCT: float = 0.10
    STOP_LOSS_PER_CONTRACT: float = 0.10
    MAX_HOLD_SECONDS: int = 86400      # 24h default time stop
    CONSECUTIVE_LOSS_PAUSE: int = 3
    PAUSE_DURATION_SECONDS: int = 1800  # 30 min

    # === Heat System ===
    HEAT_YELLOW: float = 0.05
    HEAT_ORANGE: float = 0.10
    HEAT_RED: float = 0.15
    BALANCE_FLOOR_PCT: float = 0.70

    # === Scan Intervals ===
    SCAN_INTERVAL_SECONDS: int = 10    # Ultra-fast for paper training
    POSITION_MONITOR_SECONDS: int = 10
    ODDS_REFRESH_SECONDS: int = 120
    STALE_ODDS_SECONDS: int = 300

    # === AI ===
    AI_MODEL: str = "claude-sonnet-4-20250514"
    AI_DAILY_BUDGET_USD: float = 2.0
    AI_ENABLED: bool = True

    # === Sportsbooks ===
    SPORTS: list[str] = [
        "basketball_nba", "basketball_ncaab",
        "americanfootball_nfl", "americanfootball_ncaaf",
        "baseball_mlb", "icehockey_nhl",
        "soccer_epl", "soccer_usa_mls",
        "mma_mixed_martial_arts",
    ]
    BOOKMAKERS: list[str] = [
        "draftkings", "fanduel", "betmgm", "bovada", "betonlineag",
    ]

    # === Logging ===
    LOG_FILE: str = "valuebetting.log"
    TRADE_LOG_FILE: str = "trades.jsonl"


settings = Settings()
