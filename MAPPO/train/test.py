"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import setproctitle
from pathlib import Path
import torch
from utils.test_config import get_config
from environment2.env_wrappers import DummyVecEnv
from environment2.constant_muav import *
from environment2.env_discrete import DiscreteActionEnv


def make_train_env(all_args):
    def get_env_fn():
        def init_env():
            env = DiscreteActionEnv()
            return env
        return init_env

    return DummyVecEnv([get_env_fn() for _ in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn():
        def init_env():
            env = DiscreteActionEnv()
            return env

        return init_env

    return DummyVecEnv([get_env_fn() for _ in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser, eval_seed = 1):
    parser.add_argument('--num_agents', type=int, default=num_uav, help="number of agents")
    parser.add_argument('--seed', type=int, default=eval_seed+20, help="Random seed for numpy/torch")
    """定义agent数量"""

    all_args = parser.parse_known_args(args)[0]
    return all_args


def test(args):
    reward_rnn = []
    energy_rnn = []
    # 定义数据存放路径
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results/compare_fig5")
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # 定义数据
    phm_vec = []
    re_vec = []
    for i in range(10):
        parser = get_config()
        all_args = parse_args(args, parser, eval_seed=i)

        # cuda
        if all_args.cuda and torch.cuda.is_available():
            print("choose to use gpu...")
            device = torch.device("cuda:0")
            torch.set_num_threads(all_args.n_training_threads)
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            device = torch.device("cpu")
            torch.set_num_threads(all_args.n_training_threads)

        # 设置进程名
        setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

        # seed
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        np.random.seed(all_args.seed)

        # 生成环境
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None

        """定义用户个数"""
        num_agents = num_uav

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        # 定义runner
        if all_args.share_policy:
            from runner.shared.env_runner import EnvRunner as Runner
        else:
            from runner.separated.env_runner import EnvRunner as Runner
        runner = Runner(config)

        a,b,c = runner.single_eval()
        re_vec.append(a)
        phm_vec.append(np.sum(c))

        # reward_rnn.append(reward)
        # energy_rnn.append(energy)

    print('mean_phm = ' + str(sum(phm_vec)/10))

    # np.savetxt(run_dir / 're_vec.csv', np.array(re_vec), delimiter=',')
    # np.savetxt(run_dir / 'phm_vec.csv', np.array(phm_vec), delimiter=',')

if __name__ == "__main__":
    test(sys.argv[1:])
