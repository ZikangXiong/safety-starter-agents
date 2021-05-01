import copy
from typing import Tuple

import numpy as np
from safe_rl.pg.agents import CPOAgent
from safe_rl.pg.network import mlp_actor_critic
from safe_rl.pg.run_agent import run_polopt_agent
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.load_utils import load_policy_plus


class CPO:
    def __init__(self,
                 env,
                 actor_critic_fn=mlp_actor_critic,
                 ac_kwargs=dict(),
                 seed=0,
                 render=False,
                 # Logging:
                 save_path=None,
                 exp_name=None,
                 save_freq=1,
                 _init_model=True):
        self.sess, self.pi, self.mu, self.x_ph = None, None, None, None

        if _init_model:
            self.env_fn = lambda: copy.deepcopy(env)
            cpo_kwargs = dict(
                reward_penalized=False,  # Irrelevant in CPO
                objective_penalized=False,  # Irrelevant in CPO
                learn_penalty=False,  # Irrelevant in CPO
                penalty_param_loss=False  # Irrelevant in CPO
            )
            self.agent = CPOAgent(**cpo_kwargs)
            self.actor_critic_fn = actor_critic_fn
            self.ac_kwargs = ac_kwargs
            self.seed = seed
            self.render = render

            self.logger = None
            self.logger_kwargs = setup_logger_kwargs(exp_name=exp_name, seed=seed, data_dir=save_path)
            self.logger_kwargs["output_fname"] = "log.csv"
            self.save_freq = save_freq

    def learn(self,
              # Experience collection:
              steps_per_epoch=4000,
              epochs=50,
              max_ep_len=1000,
              # Discount factors:
              gamma=0.99,
              lam=0.97,
              cost_gamma=0.99,
              cost_lam=0.97,
              # Policy learning:
              ent_reg=0.,
              # Cost constraints / penalties:
              cost_lim=25,
              penalty_init=1.,
              penalty_lr=5e-2,
              # KL divergence:
              target_kl=0.01,
              # Value learning:
              vf_lr=1e-3,
              vf_iters=80):
        self.sess, self.pi, self.mu, self.x_ph = run_polopt_agent(env_fn=self.env_fn,
                                                                  agent=self.agent,
                                                                  steps_per_epoch=steps_per_epoch,
                                                                  epochs=epochs,
                                                                  max_ep_len=max_ep_len,
                                                                  gamma=gamma,
                                                                  lam=lam,
                                                                  cost_gamma=cost_gamma,
                                                                  cost_lam=cost_lam,
                                                                  ent_reg=ent_reg,
                                                                  cost_lim=cost_lim,
                                                                  penalty_init=penalty_init,
                                                                  penalty_lr=penalty_lr,
                                                                  target_kl=target_kl,
                                                                  vf_lr=vf_lr,
                                                                  vf_iters=vf_iters,
                                                                  actor_critic=self.actor_critic_fn,
                                                                  ac_kwargs=self.ac_kwargs,
                                                                  seed=self.seed,
                                                                  render=self.render,
                                                                  logger=self.logger,
                                                                  logger_kwargs=self.logger_kwargs,
                                                                  save_freq=self.save_freq)

    def predict(self, obs: np.ndarray, deterministic=False) -> Tuple[np.ndarray, dict]:

        if deterministic:
            mu_val = self.sess.run([self.mu], feed_dict={self.x_ph: obs[np.newaxis]})
            return mu_val, {}
        else:
            pi_val = self.sess.run([self.pi], feed_dict={self.x_ph: obs[np.newaxis]})
            return pi_val, {}

    def save(self, save_path):
        # When running, the safe_rl will store model automatically
        pass

    @classmethod
    def load(cls, path, env):
        # Lazy implementation: only recover policy and env
        cpo = cls(env, _init_model=False)
        cpo.sess, cpo.pi, cpo.mu, cpo.x_ph = load_policy_plus(path)

        return cpo
