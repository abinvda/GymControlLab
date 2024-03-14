# GymControlLab
This repository contains an implementation of the Maximum a Posteriori Policy Optimization (MPO) algorithm for reinforcement learning, along with support for several baseline algorithms from Stable Baselines3, including Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). The implementation allows training and evaluating policies on Gymnasium environments.

## Installation
Before running the script, ensure you have Python 3.6+ installed along with the following packages:

- `gymnasium`
- `torch`
- `stable_baselines3`
- `tensorboardX`
- `scipy`
- `tqdm`

You can install the required packages using pip:

```bash
pip install gymnasium torch stable_baselines3 tensorboardX scipy tqdm
```

## Usage

The script can be run from the command line with various arguments to configure the environment, the algorithm, and other training parameters.

### Basic Usage

To train a model using MPO on the HalfCheetah-v4 environment:

```bash
python main.py --env HalfCheetah-v4 --algo_name MPO
```

To use a baseline algorithm like SAC instead of MPO:

```bash
python main.py --env HalfCheetah-v4 --algo_name SAC
```

### Rendering

To visualize the trained model in action, use the `--render` flag along with `--model_name` to specify the path to the trained model:

```bash
python main.py --env HalfCheetah-v4 --algo_name MPO --render True --model_name MPO_/model_latest.pt
```

### Advanced Configuration

The script supports various hyperparameters and configurations. For example, to set a custom learning rate and batch size:

```bash
python main.py --env HalfCheetah-v4 --algo_name MPO --adam_learning_rate 0.0003 --batch_size 128
```
### View training charts

The script automatically creates directories for saving logs, which can be viewed using tensorboard:

```bash
tensorboard --logdir ./training_logs/HalfCheetah-v4/
```

## Hyperparameters

The script allows configuring multiple hyperparameters, including:
- `--dual_constraint`: KL divergence dual constraint in E-step.
- `--kl_mean_constraint`: KL divergence mean constraint in M-step.
- `--kl_var_constraint`: KL divergence covariance constraint in M-step.
- `--discount_factor`: Discount factor for future rewards.
- `--adam_learning_rate`: Learning rate for Adam optimizer.
- `--policyNet_size` and `--qNet_size`: Sizes of the policy and Q-networks.
- `--sample_episode_num`: Number of episodes to sample per training iteration.
- `--batch_size`: Batch size for training.
- `--iteration_num`: Total number of training iterations.
- Additional hyperparameters for evaluation and optimization.

For a full list of arguments and their default values, use the `-h` or `--help` flag:

```bash
python main.py --help
```

## Directories

The script automatically creates directories for saving models and logs:
- `saved_models`: Directory for saving trained models.
- `training_logs`: Directory for TensorBoard logs.
