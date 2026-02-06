"""
Trading Environment - сердце RL системы.

Это "игра", в которой агент учится торговать.

Состояние (observation): технические индикаторы + информация о позиции
Действия (actions): 0=HOLD, 1=BUY, 2=SELL
Награда (reward): изменение P&L с штрафами
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from indicators import add_indicators, get_feature_columns
from config import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    POINT_VALUE,
    COMMISSION,
    MAX_POSITION,
    TRADE_PENALTY,
    PROFIT_BONUS,
    LOSS_PENALTY,
    MAX_DRAWDOWN_KILL,
    EPISODE_MAX_STEPS,
)


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for NQ futures trading.

    The agent observes a window of market features and decides to BUY, SELL, or HOLD.
    """

    metadata = {"render_modes": ["human"]}

    # Actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        point_value: float = POINT_VALUE,
        commission: float = COMMISSION,
        max_position: int = MAX_POSITION,
        eval_mode: bool = False,
        render_mode=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.eval_mode = eval_mode  # In eval mode: no early stop, run all data

        # Store raw data
        self.raw_df = df.copy()

        # Add technical indicators
        self.df = add_indicators(self.raw_df)
        self.feature_columns = get_feature_columns()
        self.n_features = len(self.feature_columns)

        # Extract feature matrix (for speed)
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.close_prices = self.df["close"].values.astype(np.float64)

        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.point_value = point_value
        self.commission = commission
        self.max_position = max_position

        # Total steps available
        self.max_steps = len(self.df) - self.window_size - 1

        # ==========================================
        # GYMNASIUM SPACES
        # ==========================================

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # - window_size * n_features (market data)
        # - 3 extra: position, unrealized_pnl_normalized, balance_change_normalized
        obs_size = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # State tracking (initialized in reset)
        self._reset_state()

    def _reset_state(self):
        """Reset all internal state variables."""
        self.current_step = 0
        self.episode_step = 0  # Steps within this episode
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance  # For reward calculation
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.trade_log = []
        self.returns_history = []  # For Sharpe-based reward

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self._reset_state()

        # Random start position for training diversity
        if options and options.get("random_start", True):
            max_start = max(0, self.max_steps - 5000)  # At least 5000 steps
            if max_start > 0:
                self.current_step = self.np_random.integers(0, max_start)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector that the agent sees.

        Contains:
        - Flattened window of technical indicators (normalized)
        - Current position (-1, 0, 1)
        - Unrealized PnL (normalized)
        - Balance change from initial (normalized)
        """
        start = self.current_step
        end = start + self.window_size

        # Market features: window_size rows x n_features columns -> flatten
        market_data = self.features[start:end].flatten()

        # Replace any NaN/Inf with 0
        market_data = np.nan_to_num(market_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Position info
        current_price = self.close_prices[end]

        # Unrealized PnL normalized by initial balance
        if self.position != 0:
            unrealized = (
                (current_price - self.entry_price)
                * self.position
                * self.point_value
            )
        else:
            unrealized = 0.0
        unrealized_norm = unrealized / self.initial_balance

        # Balance change normalized
        balance_change = (self.balance - self.initial_balance) / self.initial_balance

        # Combine all features
        extra = np.array(
            [float(self.position), unrealized_norm, balance_change],
            dtype=np.float32,
        )

        obs = np.concatenate([market_data, extra])
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        """Return info dictionary."""
        idx = self.current_step + self.window_size
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "price": self.close_prices[idx] if idx < len(self.close_prices) else 0,
            "max_balance": self.max_balance,
        }

    def step(self, action: int):
        """
        Execute one step in the environment.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        idx = self.current_step + self.window_size
        current_price = self.close_prices[idx]
        reward = 0.0
        trade_made = False

        # ==========================================
        # EXECUTE ACTION
        # ==========================================

        if action == self.BUY:
            if self.position == 0:
                # Open long position
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.commission
                trade_made = True
            elif self.position == -1:
                # Close short position
                pnl = (self.entry_price - current_price) * self.point_value
                self.balance += pnl - self.commission
                self.total_pnl += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                    reward += PROFIT_BONUS * (pnl / self.initial_balance)
                self.trade_log.append(
                    {
                        "type": "close_short",
                        "price": current_price,
                        "pnl": pnl,
                        "balance": self.balance,
                    }
                )
                # Then open long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.commission
                trade_made = True

        elif action == self.SELL:
            if self.position == 0:
                # Open short position
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.commission
                trade_made = True
            elif self.position == 1:
                # Close long position
                pnl = (current_price - self.entry_price) * self.point_value
                self.balance += pnl - self.commission
                self.total_pnl += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                    reward += PROFIT_BONUS * (pnl / self.initial_balance)
                self.trade_log.append(
                    {
                        "type": "close_long",
                        "price": current_price,
                        "pnl": pnl,
                        "balance": self.balance,
                    }
                )
                # Then open short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.commission
                trade_made = True

        # HOLD action = do nothing

        # ==========================================
        # COMPUTE REWARD (improved v2)
        # ==========================================

        # 1) Portfolio value BEFORE and AFTER this step
        #    This is the core signal: did the agent's decision make money?
        next_idx = min(idx + 1, len(self.close_prices) - 1)
        next_price = self.close_prices[next_idx]

        # Current portfolio value (balance + unrealized PnL)
        if self.position != 0:
            unrealized = (next_price - self.entry_price) * self.position * self.point_value
        else:
            unrealized = 0.0
        current_portfolio = self.balance + unrealized

        # Step return (normalized by initial balance)
        step_return = (current_portfolio - self.prev_balance) / self.initial_balance
        self.returns_history.append(step_return)
        self.prev_balance = current_portfolio

        # 2) Base reward = step return (the most important signal)
        reward += step_return * 100  # Scale up for better gradients

        # 3) Trade penalty (discourages overtrading / churning)
        if trade_made:
            reward += TRADE_PENALTY

        # 4) Reward/penalty on trade close (realized PnL feedback)
        #    Already partially handled above via balance change,
        #    but we add explicit bonus/penalty for closed trades
        if trade_made and self.trade_log:
            last_trade = self.trade_log[-1]
            pnl = last_trade["pnl"]
            if pnl > 0:
                reward += PROFIT_BONUS * (pnl / self.initial_balance)
            else:
                reward -= LOSS_PENALTY * abs(pnl / self.initial_balance)

        # 5) Track max balance for drawdown
        self.max_balance = max(self.max_balance, current_portfolio)

        # ==========================================
        # ADVANCE STEP
        # ==========================================
        self.current_step += 1
        self.episode_step += 1

        # Check if episode is done
        terminated = False
        truncated = False

        # Terminate if drawdown exceeds limit (only during training)
        drawdown = (self.max_balance - current_portfolio) / max(self.max_balance, 1)
        if not self.eval_mode and drawdown > MAX_DRAWDOWN_KILL:
            terminated = True
            reward -= 2.0  # Penalty for blowing up

        # Truncate if episode too long (only during training — shorter = faster learning)
        if not self.eval_mode and self.episode_step >= EPISODE_MAX_STEPS:
            truncated = True

        # Truncate if we've reached the end of data
        if self.current_step >= self.max_steps:
            truncated = True

        # Force close any open position on episode end
        if (terminated or truncated) and self.position != 0:
            close_idx = min(self.current_step + self.window_size, len(self.close_prices) - 1)
            close_price = self.close_prices[close_idx]
            if self.position == 1:
                pnl = (close_price - self.entry_price) * self.point_value
            else:
                pnl = (self.entry_price - close_price) * self.point_value
            self.balance += pnl - self.commission
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.position = 0

        obs = self._get_observation() if not (terminated or truncated) else self._get_terminal_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_terminal_obs(self) -> np.ndarray:
        """Return a valid observation for terminal state."""
        obs_size = self.window_size * self.n_features + 3
        return np.zeros(obs_size, dtype=np.float32)

    def get_episode_stats(self) -> dict:
        """Get statistics for the completed episode."""
        win_rate = (
            self.winning_trades / max(self.total_trades, 1) * 100
        )
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        max_dd = (self.max_balance - self.balance) / max(self.max_balance, 1) * 100

        return {
            "total_return_pct": total_return,
            "total_pnl": self.total_pnl,
            "final_balance": self.balance,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_dd,
        }

    def render(self):
        """Print current state (for debugging)."""
        if self.render_mode == "human":
            info = self._get_info()
            pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
            print(
                f"Step {info['step']:6d} | "
                f"Price: {info['price']:10.2f} | "
                f"Pos: {pos_str[info['position']]:5s} | "
                f"Balance: ${info['balance']:12.2f} | "
                f"Trades: {info['total_trades']:4d} | "
                f"PnL: ${info['total_pnl']:10.2f}"
            )
