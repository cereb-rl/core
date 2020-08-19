import numpy as np

class LinearSoftmaxPolicy:
    def __init__(self, num_features=4, num_actions=2):
        self.num_actions = num_actions
        self.num_features = num_features
        self.weights = np.random.rand(num_actions * num_features)  # Theta
    
    # def pi(self, obs):
    #     #print(obs)
    #     probs = np.exp(np.dot(self.weights, obs))
    #     #print(probs)
    #     if not np.all(probs > 0):
    #         print(self.weights, obs)
    #         print(np.dot(self.weights, obs))
    #         print(probs)
    #         print('less than zero')
    #     return probs/np.sum(probs)
    
    def pi(self, obs):
        probs = np.exp([np.dot(self.weights, self.phi(obs, action)) for action in range(self.num_actions)])
        return probs/np.sum(probs)

    def phi(self, obs, action):
        feat = np.zeros(self.num_actions*self.num_features)
        feat[action*self.num_features: (action+1)*self.num_features] = obs
        #print(feat)
        return feat

    def gradient(self, state, action):
        return self.phi(state, action) - np.sum([self.pi(state)[act]*self.phi(state, act) for act in range(self.num_actions)])

    def discount_rewards(self, rewards, gamma):
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards