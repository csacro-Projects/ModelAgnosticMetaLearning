# This file is based on spinup/algos/pytorch/vpg/vpg.py
# and implements MAML as presented in this paper https://arxiv.org/abs/1703.03400.
# The compute_loss_pi_ppo method is copied from spinup/algos/pytorch/ppo/ppo.py.
# The representation of the formula for single gradient step updates [g_i^(j), H_k^(l)]
# is taken from https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html.

import time

import gym
import numpy as np
import torch
from gym_twoDNavigation.envs import TwoDNavigationEnv
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import spinup.maml.pytorch.core as core
from spinup.maml.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads_in_module, mpi_avg_grads
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10,
        goal=None, saved_model_path=None, meta_batch_size=0, meta_learning=None, pi_lr_inner=None, vf_lr_inner=None,
		use_ppo=False, clip_ratio=0.2, target_kl=0.01):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    if saved_model_path is None:
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    else:
        ac = torch.load(saved_model_path)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf_ac = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up optimizers for learning in ac model
    pi_optimizer_ac = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer_ac = Adam(ac.v.parameters(), lr=vf_lr)
    # Set up schedulers for learning in ac model
    pi_scheduler_ac = CosineAnnealingLR(pi_optimizer_ac, epochs+1)
    vf_scheduler_ac = CosineAnnealingLR(vf_optimizer_ac, epochs+1)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Set up function for computing VPG policy loss
    def compute_loss_pi_vpg(model, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = model.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing PPO policy loss
    def compute_loss_pi_ppo(model, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = model.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(model, data):
        obs, ret = data['obs'], data['ret']
        return ((model.v(obs) - ret) ** 2).mean()

    def update(data, logging,
               model_inner=None, pi_opt_inner=None, vf_opt_inner=None, data_inner=None):
        if logging:
            d = data if data is not None else data_inner
            assert d is not None
            pi_l_old, pi_info_old = compute_loss_pi_vpg(ac, d)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(ac, d).item()
            kl, ent = pi_info_old['kl'], pi_info_old['ent']
            logger.store(LossPi=pi_l_old, LossV=v_l_old,
                         KL=kl, Entropy=ent)

        # Train policy with a single step of gradient descent
        if meta_learning is None:
            loss_pi, pi_info = compute_loss_pi_vpg(ac, data)
            grads = torch.autograd.grad(loss_pi, ac.pi.parameters())
            mpi_avg_grads(grads)  # average grads across MPI processes
            for (ac_param, grad) in zip(ac.pi.parameters(), grads):
                if ac_param.grad is None:
                    ac_param.grad = grad.clone()
                else:
                    ac_param.grad += grad.clone()
        elif data is None:
            # inner loop of meta-learning
            pi_opt_inner.zero_grad()
            compute_loss_pi = compute_loss_pi_ppo if use_ppo else compute_loss_pi_vpg
            loss_pi, pi_info = compute_loss_pi(model_inner, data_inner)
            kl = mpi_avg(pi_info['kl'])
            if not use_ppo or kl <= 1.5 * target_kl:
                loss_pi.backward()
                mpi_avg_grads_in_module(model_inner.pi)  # average grads across MPI processes
                pi_opt_inner.step()
            else:
                logger.log('No inner gradient update performed (early stopping) due to reaching max kl.')
        elif meta_learning == 'maml':
            # calculate H_0^(0)
            loss_pi, pi_info = compute_loss_pi_vpg(ac, data_inner)
            first_order_grads = torch.autograd.grad(loss_pi, ac.pi.parameters(), create_graph=True)
            mpi_avg_grads(first_order_grads)  # average grads across MPI processes
            second_order_grads = torch.autograd.grad(loss_pi, ac.pi.parameters())
            mpi_avg_grads(second_order_grads)  # average grads across MPI processes
            # calculate g_1^(1)
            loss_pi, pi_info = compute_loss_pi_vpg(model_inner, data)
            first_order_grads = torch.autograd.grad(loss_pi, model_inner.pi.parameters())
            mpi_avg_grads(first_order_grads)  # average grads across MPI processes
            # g_1_(1) - alpha g_1^(1) * H_0^(0)
            for (ac_param, second_order_grad, first_order_grad) in zip(ac.pi.parameters(), second_order_grads,
                                                                       first_order_grads):
                grad = first_order_grad - pi_lr_inner * first_order_grad * second_order_grad
                if ac_param.grad is None:
                    ac_param.grad = grad.clone()
                else:
                    ac_param.grad += grad.clone()
        elif meta_learning == 'fomaml':
            # calculate g_1^(1)
            loss_pi, pi_info = compute_loss_pi_vpg(model_inner, data)
            first_order_grads = torch.autograd.grad(loss_pi, model_inner.pi.parameters())
            mpi_avg_grads(first_order_grads)  # average grads across MPI processes
            # g_1^(1)
            for (ac_param, grad) in zip(ac.pi.parameters(), first_order_grads):
                if ac_param.grad is None:
                    ac_param.grad = grad.clone()
                else:
                    ac_param.grad += grad.clone()
        else:
            raise NotImplementedError

        # Value function learning
        for i in range(train_v_iters):
            if meta_learning is None:
                loss_v = compute_loss_v(ac, data)
                grads = torch.autograd.grad(loss_v, ac.v.parameters())
                mpi_avg_grads(grads)  # average grads across MPI processes
                for (ac_param, grad) in zip(ac.v.parameters(), grads):
                    if ac_param.grad is None:
                        ac_param.grad = grad.clone()
                    else:
                        ac_param.grad += grad.clone()
            elif data is None:
                # inner loop of meta-learning
                vf_opt_inner.zero_grad()
                loss_v = compute_loss_v(model_inner, data_inner)
                loss_v.backward()
                mpi_avg_grads_in_module(model_inner.v)  # average grads across MPI processes
                vf_opt_inner.step()
            elif meta_learning == 'maml':
                # calculate H_0^(0)
                loss_v = compute_loss_v(ac, data_inner)
                first_order_grads = torch.autograd.grad(loss_v, ac.v.parameters(), create_graph=True)
                mpi_avg_grads(first_order_grads)  # average grads across MPI processes
                second_order_grads = torch.autograd.grad(loss_v, ac.v.parameters())
                mpi_avg_grads(second_order_grads)  # average grads across MPI processes
                # calculate g_1^(1)
                loss_v = compute_loss_v(model_inner, data)
                first_order_grads = torch.autograd.grad(loss_v, model_inner.v.parameters())
                mpi_avg_grads(first_order_grads)  # average grads across MPI processes
                # g_1_(1) - alpha g_1^(1) * H_0^(0)
                for (ac_param, second_order_grad, first_order_grad) in zip(ac.v.parameters(), second_order_grads,
                                                                           first_order_grads):
                    grad = first_order_grad - vf_lr_inner * first_order_grad * second_order_grad
                    if ac_param.grad is None:
                        ac_param.grad = grad.clone()
                    else:
                        ac_param.grad += grad.clone()
            elif meta_learning == 'fomaml':
                # calculate g_1^(1)
                loss_v = compute_loss_v(model_inner, data)
                first_order_grads = torch.autograd.grad(loss_v, model_inner.v.parameters())
                mpi_avg_grads(first_order_grads)  # average grads across MPI processes
                # g_1^(1)
                for (ac_param, grad) in zip(ac.v.parameters(), first_order_grads):
                    if ac_param.grad is None:
                        ac_param.grad = grad.clone()
                    else:
                        ac_param.grad += grad.clone()
            else:
                raise NotImplementedError

    meanEpRets = []
    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(goal), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    all_train_tasks = []
    for epoch in range(epochs+1):
        if meta_batch_size == 0:
            # train on the same task
            train_tasks = [env.goal]
        else:
            # sample some new tasks
            train_tasks = [tuple(task) for task in TwoDNavigationEnv.sample_goals(meta_batch_size)]
        if epoch < epochs:
            all_train_tasks.append(train_tasks)

        pi_optimizer_ac.zero_grad()
        vf_optimizer_ac.zero_grad()
        for task in train_tasks:  # meta-batch

            def train(model_inner, pi_opt_inner, vf_opt_inner, buf_inner):

                def sample_trajectories(model, buffer, logging):
                    o, ep_ret, ep_len = env.reset(task), 0, 0
                    t = 0
                    while t < local_steps_per_epoch:
                        a, v, logp = model.step(torch.as_tensor(o, dtype=torch.float32))

                        next_o, r, d, _ = env.step(a)
                        ep_ret += r
                        ep_len += 1

                        # save and log
                        buffer.store(o, a, r, v, logp)
                        if logging:
                            logger.store(VVals=v)

                        # Update obs (critical!)
                        o = next_o

                        timeout = ep_len == max_ep_len
                        terminal = d or timeout
                        epoch_ended = t == local_steps_per_epoch - 1

                        if terminal or epoch_ended:
                            if epoch_ended and not (terminal):
                                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                            # if trajectory didn't reach terminal state, bootstrap value target
                            if timeout or epoch_ended:
                                _, v, _ = model.step(torch.as_tensor(o, dtype=torch.float32))
                            else:
                                v = 0
                            buffer.finish_path(v)
                            if terminal and logging:
                                # only save EpRet / EpLen if trajectory finished
                                logger.store(EpRet=ep_ret, EpLen=ep_len)
                            # set t and buffer as if we had needed max_ep_len,
                            # otherwise we have a larger batch_size than expected in case of done
                            diff = max_ep_len - ep_len
                            t += diff
                            buffer.ptr += diff
                            buffer.path_start_idx += diff
                            o, ep_ret, ep_len = env.reset(), 0, 0

                        t += 1

                sample_trajectories(model_inner, buf_inner, logging=True)

                if meta_learning is None:
                    # Save model
                    if (epoch % save_freq == 0) or (epoch == epochs - 1):
                        logger.save_state({'env': env}, None)
                    # Perform VPG update!
                    update(data=buf_inner.get(), logging=True)
                else:  # meta-learning
                    # Perform VPG update!
                    data_inner = buf_inner.get()
                    update(data=None, data_inner=data_inner,
                           model_inner=model_inner, pi_opt_inner=pi_opt_inner, vf_opt_inner=vf_opt_inner,
                           logging=True)
                    # sample trajectories from trained copied model with current task
                    sample_trajectories(model_inner, buf_ac, logging=False)

                    # Save model
                    if (epoch % save_freq == 0) or (epoch == epochs - 1):
                        logger.save_state({'env': env}, None)
                    # compute loss with sampled trajectories and perform meta gradient update with computed loss
                    if meta_learning == 'maml':
                        update(data=buf_ac.get(), data_inner=data_inner,
                               model_inner=model_inner, pi_opt_inner=pi_opt_inner, vf_opt_inner=vf_opt_inner,
                               logging=False)
                    elif meta_learning == 'fomaml':
                        update(data=buf_ac.get(),
                               model_inner=model_inner, pi_opt_inner=pi_opt_inner, vf_opt_inner=vf_opt_inner,
                               logging=False)
                    else:
                        raise NotImplementedError

            if meta_learning is not None:
                # Copy model
                ac_clone = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
                ac_clone.load_state_dict(ac.state_dict())
                sync_params(ac_clone)
                buf_ac_clone = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
                # Set up optimizers for policy and value function
                pi_optimizer_ac_clone = SGD(ac_clone.pi.parameters(),
                                            lr=pi_lr_inner if pi_lr_inner is not None else pi_lr)
                vf_optimizer_ac_clone = SGD(ac_clone.v.parameters(),
                                            lr=vf_lr_inner if vf_lr_inner is not None else vf_lr)
                # train
                train(ac_clone, pi_optimizer_ac_clone, vf_optimizer_ac_clone, buf_ac_clone)
            else:
                # do not perform any meta-learning
                train(ac, pi_optimizer_ac, vf_optimizer_ac, buf_ac)

        for (model_v_param, model_pi_param) in zip(ac.v.parameters(), ac.pi.parameters()):
            model_pi_param.grad = model_pi_param.grad / len(train_tasks)
            model_v_param.grad = model_v_param.grad / len(train_tasks)
        pi_optimizer_ac.step()
        vf_optimizer_ac.step()

        # Log info about epoch
        meanEpRets.append(logger.get_stats('EpRet')[0])
        logger.log_tabular('Iteration', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

        pi_scheduler_ac.step()
        vf_scheduler_ac.step()

    print("all task that were used for the parameter updates:")
    print(all_train_tasks)
    return meanEpRets


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vpg(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
