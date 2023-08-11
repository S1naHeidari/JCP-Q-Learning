import numpy as np
import matplotlib.pyplot as plt
import random

class CarRentalEnvironment:
    def __init__(self, max_cars=20, max_move=5, move_cost=2, 
    rent_reward=10, discount_factor=0.9):
        self.max_cars = max_cars
        self.max_move = max_move
        self.move_cost = move_cost
        self.rent_reward = rent_reward
        self.discount_factor = discount_factor

        self.request_means = {
            1: [5, 3],  # Sunday
            2: [4, 3],  # Monday
            3: [3, 3],  # Tuesday
            4: [2, 1],  # Wednesday
            5: [1, 2],  # Thursday
            6: [4, 5],  # Friday
            7: [3, 5]   # Saturday
        }

        self.return_means = {
            1: [5, 4],  # Sunday
            2: [5, 3],  # Monday
            3: [4, 3],  # Tuesday
            4: [3, 3],  # Wednesday
            5: [2, 1],  # Thursday
            6: [1, 2],  # Friday
            7: [5, 4]   # Saturday
        }

        # There are 441 * 7 = 3087 states
        # We represent each state with a tuple of three elements: (cars_at_loc1, cars_at_loc2, day_of_the_week)
        self.state_space = [(i, j, k) for i in range(max_cars + 1) for j in range(max_cars + 1) for k in range(1,8)]
        # There are 11 actions: (-6, 6) exclusive
        self.action_space = range(-max_move, max_move+1)
        self.day_of_the_week = 1
        self.state = (0, 0, self.day_of_the_week)
        

    def step(self, action):
        assert action in self.action_space, "Invalid action!"

        # Get request and return rates for the current day of the week
        request_mean = self.request_means[self.day_of_the_week]
        return_mean = self.return_means[self.day_of_the_week]

        # Simulate car rental requests and returns based on the means for the day of the week
        rental_requests = [np.random.poisson(request_mean[i]) for i in range(2)]
        rental_returns = [np.random.poisson(return_mean[i]) for i in range(2)]

        if action < 0:
            cars_at_loc1 = min(max(self.state[0] + min(abs(action), self.state[1]), 0), self.max_cars)
            cars_at_loc2 = min(max(self.state[1] - min(abs(action), self.state[1]), 0), self.max_cars)
            move1 = abs(self.state[0] - cars_at_loc1)
            move2 = abs(self.state[1] - cars_at_loc2)
            move = max(move1,move2)
        if action >= 0:
            cars_at_loc1 = min(max(self.state[0] - min(abs(action), self.state[0]), 0), self.max_cars)
            cars_at_loc2 = min(max(self.state[1] + min(abs(action), self.state[0]), 0), self.max_cars)
            move1 = abs(self.state[0] - cars_at_loc1)
            move2 = abs(self.state[1] - cars_at_loc2)
            move = max(move1,move2)

        # Calculate reward for renting cars
        rent_reward = self.rent_reward * min(self.state[0], rental_requests[0])
        rent_reward += self.rent_reward * min(self.state[1], rental_requests[1])

        # Update the state based on rentals and returns
        cars_at_loc1 -= min(cars_at_loc1, rental_requests[0])
        cars_at_loc2 -= min(cars_at_loc2, rental_requests[1])
        cars_at_loc1 += rental_returns[0]
        cars_at_loc2 += rental_returns[1]

        # Ensure the number of cars at each location does not exceed the maximum limit
        cars_at_loc1 = min(cars_at_loc1, self.max_cars)
        cars_at_loc2 = min(cars_at_loc2, self.max_cars)
        
        # Calculate total reward as the sum of rent_reward and moving cost (if any)
        total_reward = rent_reward - move * self.move_cost

        self.day_of_the_week += 1
        if self.day_of_the_week > 7:
            self.day_of_the_week = 1
        self.state = (cars_at_loc1, cars_at_loc2, self.day_of_the_week)

        # Return the next state, reward, and done flag
        return self.state, total_reward

    def reset(self):
        # Reset the environment to the initial state
        self.state = (0, 0, 1)
        return self.state

def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a random action
        action = np.random.randint(Q.shape[1])
        action = action - 5
        if s == 21 and action == 5:
            pass
        return action
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy

    return the index corresponding to the maximum action-state value
    '''
    action = np.argmax(Q[s])
    action = action - 5
    return action


def run_episodes_continuous(env, Q, num_steps=100, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy in the continuous environment
    '''
    tot_rew = []

    for ep in range(num_episodes):
        state = env.reset()
        state = state_to_index(state, env.day_of_the_week)
        game_rew = 0

        for t in range(num_steps):
            # Select a greedy action
            action = greedy(Q, state)
            next_state, rew = env.step(action)

            state = state_to_index(next_state, env.day_of_the_week)
            game_rew += rew

        tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))

    return np.mean(tot_rew)

def state_to_index(state, day_of_the_week):
    loc1 = state[0]
    loc2 = state[1]
    index = loc1 * (147) + ((day_of_the_week - 1) * 21 + loc2)
    return index

def exploration_rate(min_rate, max_rate, decay_rate, episode):
    return min_rate + (max_rate - min_rate) * np.exp(-decay_rate * episode)

def Q_learning(env, lr=0.01, max_steps=1000, eps=0.3, gamma=0.95, eps_decay=0.000005, table = 1):
    nA = len(env.action_space)
    nS = len(env.state_space)

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represents a state and each column represents a different action
    # if table == 1:
    Q = np.zeros((nS, nA))
    # else:
    # Q = table
    games_reward = []
    test_rewards = []
    for step in range(max_steps):
        state = env.reset()
        state = state_to_index(state, env.day_of_the_week)
        tot_rew = 0
        next_state = None
        
        # decay the epsilon value until it reaches the threshold of 0.01
        eps = exploration_rate(0.1, 1.0, 0.00005, step)

        # loop until the maximum number of steps is reached
        for t in range(7):
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)
            next_state, rew = env.step(action)  # Take one step in the environment
            next_state = state_to_index(next_state, env.day_of_the_week)

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action+5] = Q[state][action+5] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action+5])
            
            state = next_state
            #state = state_to_index(state)
            tot_rew += rew
        games_reward.append(tot_rew)
        # Test the policy every 300 steps and print the results
        if (step % 50) == 0:
            test_rew = run_episodes_continuous(env, Q, 100)
            print("Step:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(step, eps, test_rew))
            test_rewards.append(test_rew)
            
    return Q, games_reward, test_rewards

if __name__ == '__main__':
    env = CarRentalEnvironment()
    # Q = np.load('q_table2.npy')
    Q_learning, games_reward, test_rewards = Q_learning(env, lr=0.01, max_steps=50000, eps=0.4, gamma=0.9)
    
    # Assuming you have already called the Q_learning function
    # and stored the `games_reward` and `test_rewards`
    # Testing Reward Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(0, 50000, 50), test_rewards)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Test Reward')
    plt.title('Testing Reward Plot')
    plt.grid(True)
    plt.savefig('testing_reward_plot.png')  # Save the plot as 'testing_reward_plot.png' in the current directory
    plt.show()
    # Save the Q-table in the current directory
    np.save('q_table.npy', Q_learning)


