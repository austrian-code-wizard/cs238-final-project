import numpy as np
from gym.wrappers.monitor import Monitor, load_results
import mdp
import matplotlib.pyplot as plt

HORIZON = 5 * 12 * 8
UNITS_TO_SELL = 250

class LogisticPolicy:
    
    def __init__(self, θ, α, γ):
        # Initialize paramters θ, learning rate α and discount factor γ
        
        self.θ = θ
        self.α = α
        self.γ = γ
        
    def logistic(self, y):
        # definition of logistic function
        
        return 1/(1 + np.exp(-y))
    
    def probs(self, x):
        # returns probabilities of two actions
        
        y = x @ self.θ
        prob0 = self.logistic(y)
        
        return np.array([prob0, 1-prob0])        
    
    def act(self, x):
        # sample an action in proportion to probabilities
        
        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)
        
        return action, probs[action]
    
    def grad_log_p(self, x):
        # calculate grad-log-probs
        
        y = x @ self.θ        
        grad_log_p0 = x - x*self.logistic(y)
        grad_log_p1 = - x*self.logistic(y)
        
        return grad_log_p0, grad_log_p1
        
    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        # dot grads with future rewards for each action in episode
        
        return grad_log_p.T @ discounted_rewards
    
    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards
        
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.γ + rewards[i]
            discounted_rewards[i] = cumulative_rewards
            
        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob,action in zip(obs,actions)])
        
        # assert grad_log_p.shape == (len(obs), 4)
        
        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)
        
        # gradients times rewards
        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)
        
        # gradient ascent on parameters
        self.θ += self.α*dot

def parse_state(state):
    data = np.vstack([
        state["low"].to_numpy(),
        state["high"].to_numpy(),
        state["close"].to_numpy(),
        state["open"].to_numpy(),
        state["volume"].to_numpy(),
    ]).T[-1,:]
    return np.concatenate([data, np.array([state["units_sold"]]), np.array([state["cost_basis"]]),np.array([state["time"]]), np.array([1])])

def run_episode(env, policy, render=False):
    
    observation = env.reset(UNITS_TO_SELL, HORIZON, GLOBAL_SEED)
    
    totalreward = 0
    
    observations = []
    actions = []
    rewards = []
    probs = []
    
    done = False
    
    while not done:
        if render:
            env.render()
        
        observation = parse_state(observation)
        observations.append(observation)
        
        action, prob = policy.act(observation)
        observation, reward, done, _,_ = env.step(action)
        
        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)
    
    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)

def train(θ, α, γ, Policy, MAX_EPISODES=1000, seed=None, evaluate=False):

    env = mdp.TradeExecutionEnv()
    trade_sizes = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 250}
    env = mdp.DiscreteTradeSizeWrapper(env, trade_sizes)

    episode_rewards = []
    policy = Policy(θ, α, γ)
    
    # train until MAX_EPISODES
    for i in range(MAX_EPISODES):

        # run a single episode
        total_reward, rewards, observations, actions, probs = run_episode(env, policy)
                
        # keep track of episode rewards
        episode_rewards.append(total_reward)
        
        # update policy
        policy.update(rewards, observations, actions)
        print("EP: " + str(i) + " Score: " + str(total_reward) + " ",end="\r", flush=False) 

    # evaluation call after training is finished - evaluate last trained policy on 100 episodes
    # if evaluate:
    #     env = Monitor(env, 'pg_cartpole/', video_callable=False, force=True)
    #     for _ in range(100):
    #         run_episode(env, policy, render=False)
    #     env.env.close()
        
    return episode_rewards, policy

if __name__ == "__main__":
    # for reproducibility
    GLOBAL_SEED = 42
    np.random.seed(GLOBAL_SEED)

    episode_rewards, policy = train(θ=np.random.rand(9),
                                    α=0.1,
                                    γ=0.99,
                                    Policy=LogisticPolicy,
                                    MAX_EPISODES=25,
                                    seed=GLOBAL_SEED,
                                    evaluate=True)
    plt.plot(episode_rewards)
    print(policy)