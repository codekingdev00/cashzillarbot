"""Absolute safety ceilings.

These are NOT strategy parameters. Every tunable value lives in the YAML config.
These constants are the outer bounds that neither the config file nor a Telegram
/settings command can exceed, so a typo such as "risk 10" instead of "risk 1.0"
is rejected instead of trading. Change them only by deliberately editing code.
"""

MAX_RISK_PER_TRADE_PCT = 2.0      # per-trade risk can never exceed 2% of equity
MAX_LEVERAGE = 10.0               # leverage cap can never be configured above 10x
MAX_DAILY_LOSS_PCT = 10.0         # daily loss halt must trigger at or before -10%
MAX_DRAWDOWN_KILL_PCT = 50.0      # drawdown kill switch must trigger at or before -50%
MIN_REWARD_RISK = 1.0             # minimum R:R filter can never be set below 1:1
MAX_OPEN_POSITIONS = 10
MIN_SIGNAL_CATEGORIES = 3         # confluence: at least 3 uncorrelated signal categories
MIN_CONFIDENCE = 3                # a signal needs at least 3 of 5 checks
