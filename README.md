# Strategic-Decision-Optimization-in-MOBA-Games-using-Deep-Reinforcement-Learning

## **1. Project Proposal**

### 1.1 Project Topic
**Strategic Decision Optimization in Multiplayer Online Battle Arena Games Using Deep Reinforcement Learning**

### 1.2 Problem Statement

Strategic decision making in MOBA games faces several challenges:
* Complex game states make it difficult for players and scripted agents to judge team power, map control, and objective pressure.
* Limited human attention prevents players from processing many features at once, so cues about economy, vision, and fight readiness are often ignored.
* Rule based bots rely on manually designed heuristics that cannot cover the large variety of match situations.
* Early mistakes snowball into large disadvantages, and there is little systematic guidance to help players recover or protect a lead.

### 1.3 Proposed Solution
We propose using Deep Reinforcement Learning to build an agent that:
1. Evaluates game states using features such as economy, experience, objectives, vision, and combat strength
2. Learns action values through a Deep Q Network trained with experience replay
3. DRL models operate on six macro actions; the tabular baseline uses a simplified subset
4. Produces stable strategy recommendations across more than five hundred observed states

### 1.4 Data Description
We use a comprehensive public dataset from Kaggle containing:

- **9,879 match records** from high ranked Diamond tier League of Legends games
- **40 features** describing economy, combat, objective control, vision, and jungle pressure
- Real world competitive patterns reflected in early economic and map control differences

The dataset represents ten minute snapshots of ranked matches and includes key team and map control statistics such as gold, experience, kills, objectives, and vision. The target variable indicates whether the blue team won. These early game features capture meaningful competitive patterns and provide a realistic basis for modeling macro strategic decisions.

### 1.5 Expected Outcomes
- Increase average episode reward by threefold to fivefold when moving from Q learning to the Deep Q Network
- Reduce training loss by more than ten percent as the model converges
- Lower exploration rate from about eighty percent to below thirty percent during training
- Produce consistent recommended actions across over five hundred unique game states

### 1.6 Business Value
This solution targets the **$185 billion global gaming industry**, with applications in:
- Game agent development and intelligent non player character behavior
- Player training platforms that provide data driven strategic recommendations
- Game balancing and design tools informed by large scale strategic evaluation
- Esports analytics and coaching systems focused on macro decision patterns
## **2. Abstract**

**Why Important:** Early game macro decisions strongly affect MOBA win rates. These decisions are hard to model because game states are high dimensional and rewards are noisy. Using real match data helps create more reliable and interpretable strategy models.


**Techniques Used:** We implement and compare three reinforcement learning approaches:
1. **Q-Learning (Tabular)** - Uses a 36-state discretized model to provide a simple baseline.
2. **Deep Q-Network (DQN)** - Learns from a 22-dimensional one-hot state vector using a neural network.
3. **Double Deep Q-Network (DDQN)** - Uses two networks and a 6-dimensional continuous state vector to reduce overestimation.

**Results:** Our DDQN agent achieved:
- Approximately 400–500% improvement in final reward compared to the Q-Learning baseline
- About 130–140% increase in final reward compared to the DQN model

**Conclusion:** Deep RL can model early game macro decisions effectively. DDQN performs best and provides stable and interpretable strategy recommendations. This makes it suitable for analytics and coaching applications.

## **3. Introduction**

### 3.1 Background
MOBA games require players to make rapid macro decisions involving economy, map control, and objective pressure. Early game advantages strongly influence match outcomes, and ten minute game states are widely used to evaluate win probability. Recent analytic show that features such as gold difference, vision score, and objective control are highly correlated with competitive success. However, traditional rule based agents face difficulty adapting to the complexity and variability of real matches.

Key limitations include:

*   Difficulty evaluating many interacting features at once
*   Fixed heuristics that cannot generalize across diverse game states
*   Inconsistent strategic decision making under dynamic conditions

These challenges create an opportunity for reinforcement learning approaches that learn decision patterns directly from competitive match data.

### 3.2 Problem Significance
Consider these compelling statistics about early game impact in MOBA environments:

- Industry analytics report that teams with a **10-minute gold lead win 70%–80%** of matches
- Securing the first dragon increases win probability by **15%–20%**
- Taking the first Rift Herald increases early tower pressure by **58%–70%**
- Prior MOBA analyses indicate teams with a vision score advantage at 10 minutes gain **10%–15%** higher objective control success

### 3.3 Our Solution
We propose an RL powered system that learns optimal macro strategies in MOBA games by interacting with a structured environment built from real match data. Unlike rule based systems, our reinforcement learning approach:

- Adapts to diverse game states by evaluating economy, experience, objectives, vision, and combat strength
- Discovers non obvious strategic actions across more than five hundred unique early game situations
- Improves continuously through experience replay and iterative value updates

### 3.4 Business Value Proposition
**For Investors:** This solution addresses the $185 billion global gaming market with:
- Improved strategic consistency that can raise engagement and retention for competitive players
- Scalable integration into game analytics tools, training platforms, and AI assisted coaching systems
- Enhanced game balancing insights by revealing high value macro decisions across hundreds of states

This framework offers value for game developers, esports teams, and analytics platforms by enabling data grounded decision evaluation at scale.

### 3.5 Contributions
1. Built a structured MOBA decision-making environment using real match data

2. Implemented and compared Q-Learning, DQN, and DDQN for early-game macro strategy optimization

3. Demonstrated measurable improvements in reward stability and exploration efficiency

4. Produced interpretable action policies for 500+ unique early-game states

## **4. Literature Review**

### 4.1 Related Work
**Paper 1: "Towards Playing Full MOBA Games with Deep Reinforcement Learning" (Ye et al., 2020)**

- Developed a large-scale DRL system to play full MOBA games using distributed training and multi-stage policy learning.
- Achieved near human-expert performance in complex, long-horizon MOBA environments.
- **Gap:** Did not study early-game macro decisions or compare lightweight RL algorithms for interpretable strategy learning.

**Paper 2: "Dota 2 with Large Scale Deep Reinforcement Learning" (Christopher Berner, Greg Brockman, 2021)**

- Built a large-scale self-play reinforcement learning system to learn complex, long-horizon strategies in Dota 2.
- Achieved superhuman performance, defeating the world champion team OG after 10 months of continual self-play training.
- **Gap:** Focused on full-game multi-agent coordination with massive compute; did not study lightweight RL baselines or interpretable early-game macro decisions.

**Paper 3: "Mastering Complex Control in MOBA Games with Deep Reinforcement Learning" (Ye et al., 2020)**

- Proposed a scalable DRL framework using techniques like action masking, control decoupling, and dual-clip PPO to master MOBA 1v1 control tasks.
- Their agent “Tencent Solo” achieved human-professional performance in Honor of Kings 1v1 matches.
- **Gap:** Focused on fine-grained action control in 1v1 settings rather than early-game macro strategy; did not evaluate simpler RL algorithms or interpretable policy learning on structured MOBA state representations.

### 4.2 Our Contribution
Our work addresses these gaps by:
1. Modeling interpretable early-game macro decisions using structured 10-minute MOBA states derived from real competitive match data.
2. Comparing Q-Learning, DQN, and DDQN on related state abstractions constructed from the same MOBA dataset to evaluate stability, learning efficiency, and policy quality.
3. Producing interpretable action recommendations for over 500 unique early-game states using our deep RL models.


### 4.3 Theoretical Foundation
The Bellman equation forms the foundation of our approach:

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

Where:
- $Q(s,a)$ = Expected cumulative reward for taking action $a$ in state $s$
- $r$ = Immediate reward
- $\gamma$ = Discount factor (0.95)
- $s'$ = Next state after action

**Double DQN Innovation:**
$$Q(s,a) = r + \gamma Q_{target}(s', \arg\max_{a'} Q_{online}(s', a'))$$

This decouples action selection from value estimation, reducing overestimation bias.

## **5. Problem Description**

### 5.1 Problem Formulation

We formulate the MOBA strategic decision-making problem as a **Markov Decision Process (MDP)**:

**States (S):** Each state represents the team’s condition at the 10-minute game snapshot, including:

- Economic condition: gold difference (Behind / Even / Ahead)
- Objective control: tower and dragon advantage
- Combat readiness: kill difference (Losing / Even / Winning)
- Vision, map pressure, and XP status: encoded as categorical or continuous features depending on the model, DQN uses a 22 dimensional one-hot encoding of these features, while DDQN compresses them into a 6 dimensional continuous state vector.

**Actions (A):** The agent selects from six strategic actions:

- **Tower_Push** – Apply lane pressure and threaten structures
- **Safe_Farm** – Prioritize gold/XP efficiency and reduce risk
- **Dragon_Control** – Contest or secure dragon objectives
- **Vision_Control** – Improve map visibility and deny enemy wards
- **Group_Attack** – Group to force coordinated engagements
- **Baron_Control** – Pressure or secure major neutral objectives

For the Q-Learning baseline, we restrict the action space to a simplified subset of three actions (Tower_Push, Safe_Farm, Dragon_Control) for interpretability and data efficiency.

**Rewards (R):**
The reward function balances multiple objectives:
```
reward = win_loss + economy_signal + objective_signal + action_specific_shaping
```
Reward components include:
- +1 / –1 for win or loss
- Small bonus for favorable gold or dragon differences
- Action-dependent shaping to encourage appropriate macro decisions

**Transitions (T):**
State transitions follow the order of match records:
- States evolve according to real game progression
- Actions do not alter future states
- Stochasticity arises from variability across recorded matches

### 5.2 Key Challenges

1. **High-dimensional state space** — Macro features span economy, objectives, vision, XP, and combat.
2. **Weak reward signals** — Early-game advantages translate inconsistently into final outcomes.
3. **Offline RL constraint** — Actions cannot affect future states, limiting exploration.
4. **Different model abstractions** — Tabular, one-hot, and continuous states reduce cross-model comparability.

### 5.3 Solution Approach

**Step 1: Data Preparation**
- Use 10-minute high-diamond match data and engineer macro-game features.

**Step 2: Q-Learning Baseline**
- Discretize states into 36 categories and train a 36×3 Q-table with shaped rewards.

**Step 3: DQN Implementation**
- Encode states as a 22-dimensional one-hot vector and learn Q-values for six actions via a neural network and replay buffer.

**Step 4: DDQN Enhancement**
- Use a 6-dimensional continuous state vector and a dual-network setup to reduce overestimation.

## **6. Model Description**

### 6.1 Methods Overview

We implement three progressively sophisticated methods:

| Method | State Space | Function Approximation | Key Feature |
|--------|-------------|----------------------|-------------|
| Q-Learning | Discrete ([36] states) | Tabular | Exact values, interpretable |
| DQN | One-hot encoded([22] features) | Neural Network | Generalization |
| DDQN | Continuous ([6] features) | Dual Neural Networks | Reduced overestimation |

### 6.2 Q-Learning Architecture

**Q-Table Structure:**
- 36 states
  - Economy: Behind / Even / Ahead (3 levels)
  - Objective control: HardLosingObj / LosingObj / EvenObj / LeadingObj (4 levels)
  - Fight readiness: LosingFight / EvenFight / WinningFight (3 levels)
  - Total states = 3 × 4 × 3 = 36

- 3 actions:
  1. Tower_Push
  2. Safe_Farm
  3. Dragon_Control
- Total entries: 36 × 3 = 108 entries

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**Hyperparameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration (ε): 1.0 → 0.01 (decay: 0.995)

### 6.3 DQN Architecture

**Neural Network:**
```
Input Layer: 22 features
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 128 neurons (ReLU)
Hidden Layer 3: 64 neurons (ReLU)
Output Layer: 6 Q-values
```

**Key Components:**
- **Experience Replay Buffer:** 20,000 transitions
- **Batch Size:** 64
- **Optimizer:** Adam (lr= 0.001)
- **Loss Function:** Mean Squared Error

### 6.4 Double DQN Architecture

**Innovation:** Uses two networks to decouple action selection from value estimation

**Online Network:** Selects best action
$$a^* = \arg\max_{a'} Q_{online}(s', a')$$

**Target Network:** Evaluates selected action
$$Q_{target}(s', a^*)$$

**Update Frequency:** Target network updated every 5 episodes

### 6.5 Why These Methods?

1. **Q-Learning:** Provides an interpretable baseline to evaluate whether simple discrete macro-state representations are sufficient for strategic decision-making.
2. **DQN:** Handles high-dimensional one-hot state vectors and learns generalized value estimates beyond tabular limitations.
3. **DDQN:** Reduces overestimation and improves stability when learning from continuous state features and noisy reward signals.

### 6.6 Limitations and Future Improvements

**Current Limitations:**
- Weak reward signals make it difficult for DQN and DDQN to learn stable value estimates.
- State representations lack temporal context, preventing the models from capturing evolving game momentum.
- Action space mismatch across methods limits the fairness and comparability of performance results.

**Future Improvements:**
- Design richer and more consistent reward shaping to give the agent clearer learning signals.
- Introduce sequential features or recurrent architectures to model multi-step strategic dependencies.
- Unify and refine the action space and state representation to ensure fair cross-model comparisons and stronger generalization.
