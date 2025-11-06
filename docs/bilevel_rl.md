# Bi-level Reinforcement Learning Implementation

このドキュメントでは、`blackrl`に実装されたBi-level強化学習の使用方法を説明します。

## 概要

Bi-level強化学習は、リーダー・フォロワー間の相互作用を2人マルコフゲームとしてモデル化し、以下の最適化問題を解きます：

```
max_{θ_L} J_L(f_{θ_L}, g^*)
subject to g^* ∈ argmax_g J_F(f_{θ_L}, g)
```

ここで：
- `J_L`: リーダーの目的関数（割引累積報酬）
- `J_F`: フォロワーの目的関数（Max-Ent RL）
- `f_{θ_L}`: リーダーの方策
- `g^*`: フォロワーの最適反応方策

## 実装コンポーネント

### 1. 環境 (Environment)

2人マルコフゲーム環境は `blackrl.src.envs.base.Environment` を継承して実装します。

```python
from blackrl.envs import Environment, GlobalEnvSpec, EnvStep, StepType

class MyBilevelEnv(Environment):
    def __init__(self):
        # 環境の初期化
        self._spec = GlobalEnvSpec(
            observation_space=...,
            action_space=...,  # フォロワーの行動空間
            leader_action_space=...,  # リーダーの行動空間
            max_episode_length=100,
        )
    
    @property
    def spec(self):
        return self._spec
    
    def reset(self, init_state=None):
        # 環境をリセット
        obs = ...
        episode_info = {}
        return obs, episode_info
    
    def step(self, leader_action, action):
        # リーダー行動とフォロワー行動で環境を更新
        next_obs = ...
        reward = ...
        env_info = {'leader_reward': ...}
        step_type = StepType.MID
        
        return EnvStep(
            env_spec=self.spec,
            action=action,
            reward=reward,
            observation=next_obs,
            env_info=env_info,
            step_type=step_type,
        )
```

### 2. MDCE IRL

フォロワーの報酬パラメータを推定するには、`MDCEIRL`を使用します。

```python
from blackrl.agents.follower import MDCEIRL

# 特徴量関数を定義
def feature_fn(state, leader_action, follower_action):
    # φ(s, a, b) を返す
    return np.concatenate([state, leader_action, follower_action])

# MDCE IRLを初期化
mdce_irl = MDCEIRL(
    feature_fn=feature_fn,
    discount=0.99,
    learning_rate=0.01,
    max_iterations=1000,
)

# 専門家軌跡から報酬パラメータを推定
trajectories = [
    {
        'observations': [...],
        'leader_actions': [...],
        'follower_actions': [...],
        'rewards': [...],
    },
    # ... より多くの軌跡
]

w = mdce_irl.fit(trajectories, policy_fn_factory, leader_policy, env)
```

### 3. Soft Q-Learning

フォロワーの最適方策を導出するには、`SoftQLearning`を使用します。

```python
from blackrl.agents.follower import SoftQLearning

# 報酬関数を定義（MDCE IRLで推定されたパラメータを使用）
def reward_fn(state, leader_action, follower_action):
    phi = feature_fn(state, leader_action, follower_action)
    return np.dot(w, phi)

# Soft Q-Learningを初期化
soft_q = SoftQLearning(
    env_spec=env.spec,
    reward_fn=reward_fn,
    leader_policy=leader_policy,
    discount=0.99,
    learning_rate=1e-3,
    temperature=1.0,
)

# Q関数を学習
for iteration in range(1000):
    obs, _ = env.reset()
    while True:
        leader_act = leader_policy(obs)
        follower_act = soft_q.sample_action(obs, leader_act)
        env_step = env.step(leader_act, follower_act)
        
        soft_q.update(
            obs, leader_act, follower_act,
            env_step.reward, env_step.observation, env_step.last,
        )
        
        if env_step.last:
            break
```

### 4. Bi-level RLアルゴリズム

完全なBi-level RLアルゴリズムは `BilevelRL` クラスで実装されています。

```python
from blackrl.algos import BilevelRL

# リーダー方策を定義
def leader_policy(observation, deterministic=False):
    # リーダーの方策実装
    return leader_action

# Bi-level RLアルゴリズムを初期化
algo = BilevelRL(
    env_spec=env.spec,
    leader_policy=leader_policy,
    discount_leader=0.99,
    discount_follower=0.99,
    learning_rate_leader=1e-3,
    learning_rate_follower=1e-3,
)

# 学習
stats = algo.train(
    env=env,
    expert_trajectories=trajectories,
    n_leader_iterations=1000,
    n_follower_iterations=1000,
    verbose=True,
)
```

## 使用例

完全な使用例は `notebooks/` ディレクトリを参照してください。

## 参考文献

- MDCE IRL: Maximum Discounted Causal Entropy IRL
- Soft Q-Learning: エントロピー正則化されたQ-Learning
- Bi-level Optimization: 二層最適化問題

