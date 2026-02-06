"""
Train - обучение RL агента на исторических данных NQ

Запуск:
    python train.py

Это запустит обучение PPO агента на минутных данных.
Модель сохранится в папку models/
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from data_loader import prepare_data
from trading_env import TradingEnv
from config import (
    MODELS_DIR,
    RESULTS_DIR,
    RL_ALGORITHM,
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    NET_ARCH,
    BATCH_SIZE,
    N_STEPS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    ENT_COEF,
    N_EPOCHS,
    WINDOW_SIZE,
)


class TradingCallback(BaseCallback):
    """
    Custom callback to log training progress.
    """

    def __init__(self, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # Log info from finished episodes
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                if rewards:
                    print(
                        f"\n[Step {self.n_calls:,}] "
                        f"Mean reward: {np.mean(rewards):.4f} | "
                        f"Mean length: {np.mean(lengths):.0f} | "
                        f"Episodes: {len(rewards)}"
                    )
        return True


def make_env(df, seed=0):
    """Create a wrapped TradingEnv."""

    def _init():
        env = TradingEnv(df, window_size=WINDOW_SIZE)
        env = Monitor(env)
        return env

    return _init


def create_model(env):
    """Create the RL model based on config."""
    algo = RL_ALGORITHM.upper()

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            policy_kwargs={"net_arch": NET_ARCH},
            verbose=1,
            tensorboard_log=os.path.join(RESULTS_DIR, "tensorboard"),
        )
    elif algo == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            policy_kwargs={"net_arch": NET_ARCH},
            verbose=1,
            tensorboard_log=os.path.join(RESULTS_DIR, "tensorboard"),
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            policy_kwargs={"net_arch": NET_ARCH},
            verbose=1,
            tensorboard_log=os.path.join(RESULTS_DIR, "tensorboard"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return model


def train():
    """Main training function."""
    print("=" * 60)
    print("  RL TRADING BOT - TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading and preprocessing data...")
    train_df, test_df = prepare_data()

    # 2. Create environments
    print("\n[2/4] Creating trading environments...")
    train_env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(test_df)])

    print(f"  Train env ready: {len(train_df)} bars")
    print(f"  Eval env ready:  {len(test_df)} bars")

    # 3. Create model
    print(f"\n[3/4] Creating {RL_ALGORITHM} model...")
    model = create_model(train_env)
    print(f"  Architecture: {NET_ARCH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=MODELS_DIR,
        name_prefix="rl_trader",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=RESULTS_DIR,
        eval_freq=25000,
        n_eval_episodes=3,
        deterministic=True,
    )

    trading_callback = TradingCallback(eval_freq=10000)

    # 4. Train!
    print(f"\n[4/4] Starting training...")
    print(f"  This will take a while. Progress:")
    print("-" * 60)

    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, trading_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    # Save final model
    final_path = os.path.join(MODELS_DIR, "rl_trader_final")
    model.save(final_path)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE!")
    print(f"  Time: {hours}h {minutes}m {seconds}s")
    print(f"  Model saved to: {final_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train()
