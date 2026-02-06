"""
Evaluate - бэктест обученного агента на тестовых данных

Запуск:
    python evaluate.py

Это:
1. Загрузит обученную модель
2. Прогонит её на тестовых данных (2022-2025)
3. Покажет статистику и графики
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from data_loader import prepare_data
from trading_env import TradingEnv
from config import MODELS_DIR, RESULTS_DIR, RL_ALGORITHM, WINDOW_SIZE, INITIAL_BALANCE


def load_model(model_path=None):
    """Load a trained model."""
    if model_path is None:
        # Try best model first, then final
        best_path = os.path.join(MODELS_DIR, "best_model.zip")
        final_path = os.path.join(MODELS_DIR, "rl_trader_final.zip")

        if os.path.exists(best_path):
            model_path = best_path
            print(f"Loading best model: {best_path}")
        elif os.path.exists(final_path):
            model_path = final_path
            print(f"Loading final model: {final_path}")
        else:
            raise FileNotFoundError(
                f"No trained model found in {MODELS_DIR}. Run train.py first!"
            )

    algo = RL_ALGORITHM.upper()
    if algo == "PPO":
        model = PPO.load(model_path)
    elif algo == "A2C":
        model = A2C.load(model_path)
    elif algo == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return model


def run_backtest(model, df, deterministic=True):
    """
    Run the model on data and collect results.

    Returns dict with equity curve, trades, stats.
    """
    env = TradingEnv(df, window_size=WINDOW_SIZE, eval_mode=True)
    obs, info = env.reset(options={"random_start": False})

    # Track everything
    equity_curve = [INITIAL_BALANCE]
    positions = [0]
    prices = [info["price"]]
    actions_taken = []
    steps = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        equity_curve.append(info["balance"])
        positions.append(info["position"])
        prices.append(info["price"])
        actions_taken.append(int(action))
        steps += 1

    # Episode stats
    stats = env.get_episode_stats()
    stats["steps"] = steps
    stats["actions"] = actions_taken
    stats["equity_curve"] = equity_curve
    stats["prices"] = prices
    stats["trade_log"] = env.trade_log

    return stats


def calculate_sharpe(equity_curve, periods_per_year=252 * 78):
    """
    Calculate annualized Sharpe ratio.
    For 5-min bars: ~78 bars per trading day * 252 days.
    """
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown percentage."""
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    return max_dd * 100


def plot_results(stats, save_path=None):
    """Generate performance plots."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("RL Trading Agent - Backtest Results", fontsize=16, fontweight="bold")

    equity = stats["equity_curve"]
    prices = stats["prices"]

    # Plot 1: Equity Curve
    ax1 = axes[0]
    ax1.plot(equity, color="green", linewidth=1.5, label="Portfolio Value")
    ax1.axhline(y=INITIAL_BALANCE, color="gray", linestyle="--", alpha=0.5, label="Initial Balance")
    ax1.fill_between(
        range(len(equity)),
        INITIAL_BALANCE,
        equity,
        where=[e > INITIAL_BALANCE for e in equity],
        color="green",
        alpha=0.1,
    )
    ax1.fill_between(
        range(len(equity)),
        INITIAL_BALANCE,
        equity,
        where=[e < INITIAL_BALANCE for e in equity],
        color="red",
        alpha=0.1,
    )
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Price + Actions
    ax2 = axes[1]
    ax2.plot(prices, color="blue", linewidth=0.5, alpha=0.7, label="NQ Price")

    # Mark buy/sell actions
    actions = stats["actions"]
    buy_steps = [i for i, a in enumerate(actions) if a == 1]
    sell_steps = [i for i, a in enumerate(actions) if a == 2]

    if buy_steps:
        ax2.scatter(
            buy_steps,
            [prices[i + 1] for i in buy_steps if i + 1 < len(prices)],
            color="green",
            marker="^",
            s=10,
            alpha=0.6,
            label=f"BUY ({len(buy_steps)})",
        )
    if sell_steps:
        ax2.scatter(
            sell_steps,
            [prices[i + 1] for i in sell_steps if i + 1 < len(prices)],
            color="red",
            marker="v",
            s=10,
            alpha=0.6,
            label=f"SELL ({len(sell_steps)})",
        )

    ax2.set_title("Price & Trading Actions")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Drawdown
    ax3 = axes[2]
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    ax3.fill_between(range(len(drawdown)), 0, drawdown, color="red", alpha=0.3)
    ax3.plot(drawdown, color="red", linewidth=0.5)
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("Drawdown %")
    ax3.set_xlabel("Step")
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def print_stats(stats):
    """Print formatted backtest statistics."""
    equity = stats["equity_curve"]
    sharpe = calculate_sharpe(equity)
    max_dd = calculate_max_drawdown(equity)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Balance:     ${INITIAL_BALANCE:>12,.2f}")
    print(f"  Final Balance:       ${stats['final_balance']:>12,.2f}")
    print(f"  Total Return:        {stats['total_return_pct']:>11.2f}%")
    print(f"  Total PnL:           ${stats['total_pnl']:>12,.2f}")
    print(f"  Sharpe Ratio:        {sharpe:>11.2f}")
    print(f"  Max Drawdown:        {max_dd:>11.2f}%")
    print(f"  Total Trades:        {stats['total_trades']:>11d}")
    print(f"  Winning Trades:      {stats['winning_trades']:>11d}")
    print(f"  Win Rate:            {stats['win_rate_pct']:>11.2f}%")
    print(f"  Steps Simulated:     {stats['steps']:>11,d}")
    print("=" * 60)

    # Verdict
    if stats["total_return_pct"] > 0 and sharpe > 1.0:
        print("  VERDICT: PROMISING - Consider paper trading")
    elif stats["total_return_pct"] > 0:
        print("  VERDICT: MARGINALLY POSITIVE - Needs more tuning")
    else:
        print("  VERDICT: NEEDS WORK - Adjust reward/features/training")
    print("=" * 60)


def evaluate():
    """Main evaluation function."""
    print("=" * 60)
    print("  RL TRADING BOT - EVALUATION")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading test data...")
    _, test_df = prepare_data()

    # Load model
    print("\n[2/3] Loading trained model...")
    model = load_model()

    # Run backtest
    print("\n[3/3] Running backtest on test data...")
    stats = run_backtest(model, test_df, deterministic=True)

    # Print results
    print_stats(stats)

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "backtest_results.png")
    plot_results(stats, save_path=plot_path)

    # Save detailed results
    results_path = os.path.join(RESULTS_DIR, "backtest_stats.csv")
    pd.DataFrame([stats]).drop(
        columns=["actions", "equity_curve", "prices", "trade_log"], errors="ignore"
    ).to_csv(results_path, index=False)
    print(f"\nDetailed stats saved to: {results_path}")

    return stats


if __name__ == "__main__":
    evaluate()
