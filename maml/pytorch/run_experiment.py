import torch
from gym_twoDNavigation.envs import TwoDNavigationEnv

from spinup.maml.pytorch.vpg_maml_point import vpg as vpg_all
from spinup.maml.utils.run_utils import ExperimentGrid
from spinup.user_config import DEFAULT_DATA_DIR

""" start of hyper-parameter change area """

discount = 0.99  # RL param
gae_lambda = 0.97  # RL param
epsilon = 0.2  # RL param in case PPO is used
d_target = 0.01  # RL param in case PPO is used

# values from MAML (paper https://arxiv.org/abs/1703.03400 and github https://github.com/cbfinn/maml_rl)
hidden_sizes = (100, 100)  # model parameters
hidden_activation = torch.nn.ReLU  # model parameters

horizon = 100  # trajectory length / horizon H
meta_batch_size = 20  # amount of tasks / meta batch size M
iterations = 100  # iterations / number of iterations for training
sample_batch_size = 20  # amount of trajectories / sample batch size K

eval_task_amount = 40  # amount of tasks for evaluation
eval_iterations = 3  # iterations / number of iterations for evaluation
eval_batch_size = 40  # amount of trajectories / sample batch size K for evaluation
# (on github they used eval_batch_size=4000 but there is not much difference to eval_batch_size=40)

# learning rates by tuning with hyperopt
pi_lr_inner = 0.0990  # alpha for actor
vf_lr_inner = 0.1200  # alpha for critic
pi_lr = 0.0028  # beta for actor
vf_lr = 0.0039  # beta for critic
pi_lr_eval = 0.0029  # learning rate for actor for evaluation
vf_lr_eval = 0.0036  # learning rate for critic for evaluation

# seeds
seed = 0  # training: initial model parameters, sampling of tasks during training and randomness in training
eval_seed = 1  # evaluation: sampling of tasks for evaluation (see below) and randomness in training
# if eval_seed == seed, we evaluate on the first 'eval_task_amount' sampled task from the training

""" end of hyper-parameter change area """

# tasks used for evaluation
evaluation_tasks = [tuple(goal) for goal in TwoDNavigationEnv.sample_goals(eval_task_amount + 1, seed=eval_seed)[1:]]


def create_ExperimentGrid(name, env_name, is_training, meta_learning=None, saved_model_path=None, use_ppo=False,
                          is_random=False):
    eg = ExperimentGrid(name=name)
    eg.add('env_name', env_name)
    eg.add('ac_kwargs:hidden_sizes', hidden_sizes)
    eg.add('ac_kwargs:activation', hidden_activation)
    eg.add('seed', seed if is_training else eval_seed)
    eg.add('steps_per_epoch', sample_batch_size * horizon if is_training else eval_batch_size * horizon)
    if is_random:
        eg.add('epochs', 0)
    else:
        eg.add('epochs', iterations if is_training else eval_iterations)
    eg.add('gamma', discount)
    eg.add('train_v_iters', 1)
    eg.add('lam', gae_lambda)
    eg.add('max_ep_len', horizon)
    eg.add('goal', None if is_training else evaluation_tasks)
    eg.add('saved_model_path', saved_model_path)
    eg.add('meta_batch_size', meta_batch_size if is_training else 0)
    eg.add('pi_lr', pi_lr if is_training else pi_lr_eval)
    eg.add('vf_lr', vf_lr if is_training else vf_lr_eval)
    eg.add('meta_learning', meta_learning)
    if meta_learning is not None:
        eg.add('pi_lr_inner', pi_lr_inner)
        eg.add('vf_lr_inner', vf_lr_inner)
    eg.add('use_ppo', use_ppo)
    if use_ppo:
        eg.add('clip_ratio', epsilon)
        eg.add('target_kl', d_target)
    return eg


def maml(saved_model_path=None, use_ppo=False):
    def train_maml():
        eg_maml_train = create_ExperimentGrid(name='train-maml',
                                              env_name='gym_twoDNavigation:twoDNavigation-v0',
                                              is_training=True, meta_learning='maml', use_ppo=use_ppo)
        eg_maml_train.run(vpg_all)

    if saved_model_path is None or not saved_model_path:
        if saved_model_path is None:
            train_maml()
        saved_model_path = DEFAULT_DATA_DIR + '/train-maml/train-maml_s' + str(seed) + '/pyt_save/model.pt'
    eg_maml = create_ExperimentGrid(name='evaluate-maml',
                                    env_name='gym_twoDNavigation:twoDNavigation-v0',
                                    is_training=False, saved_model_path=saved_model_path)
    eg_maml.run(vpg_all)


def fomaml(saved_model_path=None, use_ppo=False):
    def train_fomaml():
        eg_fomaml_train = create_ExperimentGrid(name='train-fomaml',
                                                env_name='gym_twoDNavigation:twoDNavigation-v0',
                                                is_training=True, meta_learning='fomaml', use_ppo=use_ppo)
        eg_fomaml_train.run(vpg_all)

    if saved_model_path is None or not saved_model_path:
        if saved_model_path is None:
            train_fomaml()
        saved_model_path = DEFAULT_DATA_DIR + '/train-fomaml/train-fomaml_s' + str(seed) + '/pyt_save/model.pt'
    eg_fomaml = create_ExperimentGrid(name='evaluate-fomaml',
                                      env_name='gym_twoDNavigation:twoDNavigation-v0',
                                      is_training=False, saved_model_path=saved_model_path)
    eg_fomaml.run(vpg_all)


def pretrained(saved_model_path=None):
    def train_pretrained():
        eg_pretrained_train = create_ExperimentGrid(name='train-pretrained',
                                                    env_name='gym_twoDNavigation:twoDNavigation-v0',
                                                    is_training=True)
        eg_pretrained_train.run(vpg_all)

    if saved_model_path is None or not saved_model_path:
        if saved_model_path is None:
            train_pretrained()
        saved_model_path = DEFAULT_DATA_DIR + '/train-pretrained/train-pretrained_s' + str(seed) + '/pyt_save/model.pt'
    eg_pretrained = create_ExperimentGrid(name='evaluate-pretrained',
                                          env_name='gym_twoDNavigation:twoDNavigation-v0',
                                          is_training=False, saved_model_path=saved_model_path)
    eg_pretrained.run(vpg_all)


def random():
    eg_random_train = create_ExperimentGrid(name='train-random', env_name='gym_twoDNavigation:twoDNavigation-v0',
                                            is_training=True, is_random=True)
    eg_random_train.run(vpg_all)
    saved_model_path = DEFAULT_DATA_DIR + '/train-random/train-random_s' + str(seed) + '/pyt_save/model.pt'
    eg_random = create_ExperimentGrid(name='evaluate-random', env_name='gym_twoDNavigation:twoDNavigation-v0',
                                      is_training=False, saved_model_path=saved_model_path)
    eg_random.run(vpg_all)


def oracle(saved_model_path=None):
    def train_oracle():
        eg_oracle_train = create_ExperimentGrid(name='train-oracle',
                                                env_name='gym_twoDNavigation:twoDNavigation-oracle-v0',
                                                is_training=True)
        eg_oracle_train.run(vpg_all)

    if saved_model_path is None or not saved_model_path:
        if saved_model_path is None:
            train_oracle()
        saved_model_path = DEFAULT_DATA_DIR + '/train-oracle/train-oracle_s' + str(seed) + '/pyt_save/model.pt'
    eg_oracle = create_ExperimentGrid(name='evaluate-oracle', env_name='gym_twoDNavigation:twoDNavigation-oracle-v0',
                                      is_training=False, saved_model_path=saved_model_path)
    eg_oracle.run(vpg_all)


def search_learning_rates(type, use_ppo=False):
    from hyperopt import fmin, tpe
    from hyperopt import hp
    import gym

    def train_pretrained_objective(args):
        print(args)
        pi_lr_grid = args['pi_lr']
        vf_lr_grid = args['vf_lr']
        rewards = vpg_all(env_fn=lambda: gym.make('gym_twoDNavigation:twoDNavigation-v0'),
                          ac_kwargs=dict(hidden_sizes=hidden_sizes, activation=hidden_activation),
                          steps_per_epoch=sample_batch_size * horizon,
                          epochs=10,
                          gamma=discount,
                          train_v_iters=1,
                          lam=gae_lambda,
                          max_ep_len=horizon,
                          goal=None,
                          meta_batch_size=10,
                          pi_lr=pi_lr_grid,
                          vf_lr=vf_lr_grid
                          )
        return - sum(rewards) / len(rewards)  # best average (over iterations) reward during training

    def train_maml_objective(args):
        print(args)
        pi_lr_inner_grid = args['pi_lr_inner']
        vf_lr_inner_grid = args['vf_lr_inner']
        rewards = vpg_all(env_fn=lambda: gym.make('gym_twoDNavigation:twoDNavigation-v0'),
                          ac_kwargs=dict(hidden_sizes=hidden_sizes, activation=hidden_activation),
                          steps_per_epoch=sample_batch_size * horizon,
                          epochs=10,
                          gamma=discount,
                          train_v_iters=1,
                          lam=gae_lambda,
                          max_ep_len=horizon,
                          goal=None,
                          meta_batch_size=10,
                          pi_lr=pi_lr,  # obtained from trained_pretrained_best
                          vf_lr=vf_lr,  # obtained from trained_pretrained_best
                          meta_learning='maml',
                          pi_lr_inner=pi_lr_inner_grid,
                          vf_lr_inner=vf_lr_inner_grid,
                          use_ppo=use_ppo,
                          clip_ratio=epsilon,
                          target_kl=d_target
                          )
        return - sum(rewards) / len(rewards)  # best average (over iterations) reward during training

    def evaluate_maml_objective(args):
        print(args)
        pi_lr_eval_grid = args['pi_lr_eval']
        vf_lr_eval_grid = args['vf_lr_eval']
        reward = 0
        for i in range(20):
            rewards = vpg_all(env_fn=lambda: gym.make('gym_twoDNavigation:twoDNavigation-v0'),
                              ac_kwargs=dict(hidden_sizes=hidden_sizes, activation=hidden_activation),
                              steps_per_epoch=sample_batch_size * horizon,
                              epochs=eval_iterations,
                              gamma=discount,
                              train_v_iters=1,
                              lam=gae_lambda,
                              max_ep_len=horizon,
                              goal=evaluation_tasks[i],
                              meta_batch_size=0,
                              pi_lr=pi_lr_eval_grid,
                              vf_lr=vf_lr_eval_grid,
                              saved_model_path=DEFAULT_DATA_DIR + '/train-maml/train-maml_s' + str(seed) + '/pyt_save/model.pt'
                              )
            reward += rewards[-1]
        return -reward / 20  # best average (over goals) final reward

    # define a search space: we constrain the search space by pi_lr_? < vf_lr_?
    scale = hp.uniform('scale', 0.5, 1)
    scale_inner = hp.uniform('scale_inner', 0.5, 1)
    scale_eval = hp.uniform('scale_eval', 0.5, 1)
    lr = hp.loguniform('lr', -10, -2)
    lr_inner = hp.loguniform('lr_inner', -8, 0)
    lr_eval = hp.loguniform('lr_eval', -10, -2)
    lr_space = {
        'pi_lr': scale * lr,
        'vf_lr': lr
    }
    lr_inner_space = {
        'pi_lr_inner': scale_inner * lr_inner,
        'vf_lr_inner': lr_inner
    }
    lr_eval_space = {
        'pi_lr_eval': scale_eval * lr_eval,
        'vf_lr_eval': lr_eval
    }

    train_pretrained_best = None
    train_maml_best = None
    evaluate_maml_best = None
    # minimize the objective over the space
    if type == 'lr':
        train_pretrained_best = fmin(train_pretrained_objective, lr_space, algo=tpe.suggest, max_evals=100)
        print('train pretrained')
        print(train_pretrained_best)
    if type == 'lr_inner':
        train_maml_best = fmin(train_maml_objective, lr_inner_space, algo=tpe.suggest, max_evals=100)
        print('train MAML')
        print(train_maml_best)
    if type == 'lr_eval':
        evaluate_maml_best = fmin(evaluate_maml_objective, lr_eval_space, algo=tpe.suggest, max_evals=50)
        print('evaluate MAML')
        print(evaluate_maml_best)

    print('train pretrained')
    print(train_pretrained_best)
    # default setup
    # {'lr': 0.003906901999623383, 'scale': 0.7290835445863331}    # best loss 53.8634841225364
    # vf_lr = 0.003906901999623383, pi_lr = 0.002848458

    print('train MAML')
    print(train_maml_best)
    # default setup
    # {'lr_inner': 0.10998566998676625, 'scale_inner': 0.8996942138954774}  # best loss 52.005310405384414
    # vf_lr_inner = 0.10998566998676625, pi_lr_inner = 0.098953471

    # default setup with PPO
    # {'lr_inner': 0.12147481214967237, 'scale_inner': 0.867154885498972}  # best loss 51.4439021023837
    # vf_lr_inner = 0.12147481214967237, pi_lr_inner = 0.105337477

    print('evaluate MAML')
    print(evaluate_maml_best)
    # default setup
    # {'lr_eval': 0.0035953217553703226, 'scale_eval': 0.8029936747726055}  # best loss 12.477317237854004
    # vf_lr_eval = 0.0035953217553703226, pi_lr_eval = 0.002887021

    # default setup with PPO
    # {'lr_eval': 0.0032858292245435045, 'scale_eval': 0.9193482695154358}  # best loss 13.47548770904541
    # vf_lr_eval = 0.0032858292245435045, pi_lr_eval = 0.003020821


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='If you want to change any hyper-parameters, do this directly in the run_experiment.py file in the top, please.')
    parser.add_argument('--search_learning_rate', '-search', choices=['lr', 'lr_inner', 'lr_eval'], default=None,
                        help='Start searching for learning rates (no other experiments will be run). '
                             '"lr" is beta, "lr_inner" is alpha and "lr_eval" is learning rate for evaluation')
    parser.add_argument('--random', '-r', action='store_true', default=False,
                        help='Start the random baseline experiment (evaluation).')
    parser.add_argument('--oracle', '-o', action='store_true', default=False,
                        help='Start the oracle baseline experiment (training + evaluation).')
    parser.add_argument('--oracle_saved_model', '-of', type=str, default=None,
                        help='Start the oracle baseline experiment (evaluation only) with the model saved in the given model.pt file path (if "", default model.pt file path is used).')
    parser.add_argument('--pretrained', '-p', action='store_true', default=False,
                        help='Start the pretrained baseline experiment (training + evaluation).')
    parser.add_argument('--pretrained_saved_model', '-pf', type=str, default=None,
                        help='Start the pretrained baseline experiment (evaluation only) with the model saved in the given model.pt file path (if "", default model.pt file path is used).')
    parser.add_argument('--maml', '-m', action='store_true', default=False,
                        help='Start the MAML experiment (training + evaluation).')
    parser.add_argument('--maml_saved_model', '-mf', type=str, default=None,
                        help='Start the MAML experiment (evaluation only) with the model saved in the given model.pt file path (if "", default model.pt file path is used).')
    parser.add_argument('--fomaml', '-fom', action='store_true', default=False,
                        help='Start the fist-order MAML experiment (training + evaluation).')
    parser.add_argument('--fomaml_saved_model', '-fomf', type=str, default=None,
                        help='Start the first-order MAML experiment (evaluation only) with the model saved in the given model.pt file path (if "", default model.pt file path is used).')
    parser.add_argument('--use_ppo', '-ppo', action='store_true', default=False,
                        help='Whether to switch from Vanilla Policy Gradients to Proximal Policy Optimization for the base-learner update in the meta-learning setup.')
    args = parser.parse_args()

    if args.search_learning_rate:
        search_learning_rates(args.search_learning_rate, args.use_ppo)
    else:
        if args.random:
            random()
        if args.oracle or args.oracle_saved_model is not None:
            oracle(args.oracle_saved_model)
        if args.pretrained or args.pretrained_saved_model is not None:
            pretrained(args.pretrained_saved_model)
        if args.maml or args.maml_saved_model is not None:
            maml(args.maml_saved_model, args.use_ppo)
        if args.fomaml or args.fomaml_saved_model is not None:
            fomaml(args.fomaml_saved_model, args.use_ppo)
        else:
            parser.print_help()
            exit()

    print("all tasks that were used for evaluation:")
    print(evaluation_tasks)
