"""
Training script for RL-based Dynamic Fusion.
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# RL imports
try:
    import torch.nn as nn
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv, DummyVecEnv, VecNormalize,
    )
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

try:
    from sb3_contrib import TRPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

# Environment
from dynamic_fusion_env import create_env


class DynamicFusionTrainer:
    """Trainer for the RL-based Dynamic Fusion agent."""

    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.vecnorm_path = os.path.join(self.output_dir, "vecnormalize.pkl")
        print(f"Results will be saved to: {self.output_dir}")

    def make_env(self, rank, mode):
        """Factory returning a thunk that builds a single environment instance."""
        def _init():
            return create_env(
                mode=mode,
                dataset_size=self.args.dataset_size,
                task=self.args.task,
                action_type="weights",  # continuous weights for PPO/SAC/TRPO
                split_id=self.args.split_id,
                classifier_type=self.args.classifier,
                out_csv_name=f"{self.output_dir}/rewards{rank}",
            )
        return _init

    def create_train_env(self):
        """Create the vectorized TRAIN environment with obs/reward normalization."""
        print(f"Creating TRAIN environment: {self.args.dataset_size} dataset, "
              f"{self.args.task} task")

        # FIX (leakage): train on the TRAIN split, never on the test outlets.
        env = SubprocVecEnv(
            [self.make_env(i, mode="train") for i in range(self.args.num_envs)]
        )
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def train_agent(self, env):
        algo_name = self.args.algo.upper()

        algo_map = {
            'ppo': PPO,
            'sac': SAC,
            'trpo': TRPO if SB3_CONTRIB_AVAILABLE else None,
        }
        AlgoClass = algo_map.get(self.args.algo.lower())
        if AlgoClass is None:
            print(f"Error: Algorithm '{self.args.algo}' not available. "
                  f"Install sb3-contrib for TRPO.")
            return None

        policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.Tanh)

        if self.args.algo.lower() == 'sac':
            model = AlgoClass(
                "MlpPolicy", env,
                learning_rate=self.args.learning_rate,
                batch_size=self.args.batch_size,
                gamma=self.args.gamma,
                policy_kwargs=policy_kwargs,
                seed=self.args.seed,
                verbose=0,
            )
        else:  # PPO, TRPO
            model = AlgoClass(
                "MlpPolicy", env,
                learning_rate=self.args.learning_rate,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                n_epochs=self.args.n_epochs,
                gamma=self.args.gamma,
                policy_kwargs=policy_kwargs,
                seed=self.args.seed,
                verbose=0,
            )

        model.learn(total_timesteps=self.args.total_timesteps, progress_bar=True)

        model_path = f"{self.output_dir}/{self.args.algo}_dynamic_fusion_model"
        model.save(model_path)
        env.save(self.vecnorm_path)
        return model

    def evaluate_agent(self, model):
        """Evaluate the trained agent on the held-out TEST split."""
        if model is None:
            return {}

        algo_name = self.args.algo.upper()

        eval_env = DummyVecEnv([self.make_env(0, mode="test")])
        eval_env = VecNormalize.load(self.vecnorm_path, eval_env)
        eval_env.training = False       # freeze running statistics
        eval_env.norm_reward = False    # report raw (un-normalized) rewards

        obs = eval_env.reset()
        for _ in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, dones, infos = eval_env.step(action)
            if dones[0]:
                info = infos[0]
                results = {
                    'accuracy': info.get('accuracy', 0),
                    'precision': info.get('precision', 0),
                    'recall': info.get('recall', 0),
                    'f1_score': info.get('f1_score', 0),
                    'mae': info.get('mae', 0),
                    'total_outlets': info.get('total_outlets', 0),
                }   

                return results
        return {}

    def save_results(self, results):
        """Persist the RL results and the experiment configuration."""
        if results:
            row = {
                'Model': self.args.algo.upper(),
                'Task': self.args.task,
                'Dataset': self.args.dataset_size,
                'Split': self.args.split_id,
                'Accuracy': round(results.get('accuracy', 0), 4),
                'Precision': round(results.get('precision', 0), 4),
                'Recall': round(results.get('recall', 0), 4),
                'F1-Score': round(results.get('f1_score', 0), 4),
                'MAE': round(results.get('mae', 0), 4),
            }
            csv_path = f"{self.output_dir}/results_{self.args.algo}_" \
                       f"{self.args.dataset_size}_{self.args.task}.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

        config = {
            'timestamp': self.timestamp,
            'algo': self.args.algo,
            'dataset_size': self.args.dataset_size,
            'task': self.args.task,
            'split_id': self.args.split_id,
            'classifier': self.args.classifier,
            'total_timesteps': self.args.total_timesteps,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size,
            'n_steps': self.args.n_steps,
            'n_epochs': self.args.n_epochs,
            'gamma': self.args.gamma,
            'num_envs': self.args.num_envs,
            'seed': self.args.seed,
        }
        with open(f"{self.output_dir}/experiment_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Experiment config saved to: {self.output_dir}/experiment_config.json")

    def run(self):
        # run the full training and evaluation pipeline
        env = self.create_train_env()
        model = self.train_agent(env)
        results = self.evaluate_agent(model)
        self.save_results(results)



def main():
    parser = argparse.ArgumentParser(
        description='Train the RL-based Dynamic Fusion agent (contextual bandit).'
    )

    # Algorithm
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'trpo', 'sac'],
                        help='RL algorithm (paper uses PPO).')

    # Environment
    parser.add_argument('--datasetsize', dest='dataset_size', type=str,
                        default='small', choices=['small', 'large'],
                        help='Dataset: small (ACL-2020) or large (MBFC-2025).')
    parser.add_argument('--task', type=str, default='bias',
                        choices=['bias', 'fact'],
                        help='Classification task: bias or factuality.')
    parser.add_argument('--split-id', type=int, default=0, choices=range(5),
                        help='Cross-validation split id (0-4).')
    parser.add_argument('--classifier', type=str, default='rf',
                        choices=['rf', 'svm'],
                        help='Fixed reward classifier: rf (RandomForest) or svm.')

    # Training hyper-parameters (defaults follow the paper)
    parser.add_argument('--total-timesteps', type=int, default=500000,
                        help='Total training timesteps.')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-4,
                        help='Learning rate (paper: 1e-4).')
    parser.add_argument('--st', dest='n_steps', type=int, default=1024,
                        help='Rollout size / steps per update (paper: 1024).')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size (paper: 256).')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Epochs per policy update.')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Discount factor. Contextual bandit => 0 '
                             '(immediate reward only).')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Number of parallel training environments.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    args = parser.parse_args()
    np.random.seed(args.seed)

    if not STABLE_BASELINES_AVAILABLE:
        print("Error: stable-baselines3 is required.")
        print("Install with: pip install -r requirements.txt")
        return

    DynamicFusionTrainer(args).run()


if __name__ == "__main__":
    main()
