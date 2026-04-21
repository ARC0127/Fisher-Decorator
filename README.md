**Fisher Decorator: Refining Flow Policy via A Local Transport Map**



## Acknowledgements

The research presented in this paper is led by Xiaoyuan Cheng.  
The core codebase is primarily developed by Xiaoyuan Cheng and Haoyu Wang.

This repository partially builds upon ideas and implementations from the paper  
"DeFlow: Decoupling Manifold Modeling and Value Maximization for Offline Policy Extraction" (https://arxiv.org/abs/2601.10471).


## Abstract

Recent advances in flow-based offline reinforcement learning (RL) have achieved strong performance by parameterizing policies via flow matching. However, they still face critical trade-offs among expressiveness, optimality, and efficiency. In particular, existing flow policies interpret the $L_2$ regularization as an upper bound of the 2-Wasserstein distance ($W_2$), which can be problematic in offline settings. This issue stems from a fundamental geometric mismatch: the behavioral policy manifold is inherently anisotropic, whereas the $L_2$ (or upper bound of $W_2$) regularization is isotropic and density-insensitive, leading to systematically misaligned optimization directions. To address this, we revisit offline RL from a geometric perspective and show that policy refinement can be formulated as a local transport map—an initial flow policy augmented by a residual displacement. By analyzing the induced density transformation, we derive a local quadratic approximation of the KL-constrained objective governed by the Fisher information matrix, enabling a tractable anisotropic optimization formulation. By leveraging the score function embedded in the flow velocity, we obtain a corresponding quadratic constraint for efficient optimization. Our results reveal that the optimality gap in prior methods arises from their isotropic approximation. In contrast, our framework achieves a controllable approximation error within a provable neighborhood of the optimal solution. Extensive experiments demonstrate state-of-the-art performance across diverse offline RL benchmarks. 

paper link: https://arxiv.org/abs/2604.17919.

## Code Implementation

This repository provides the official implementation of **FiDec (Fisher Decorator)** built on top of a flow-based offline RL framework.---

### Running the Code

To train FiDec on a D4RL benchmark:

```bash
python main.py \
  --env_name=hopper-medium-v2 \
  --agent=agents/fidec.py \
  --offline_steps=1000000 \
  --seed=0
