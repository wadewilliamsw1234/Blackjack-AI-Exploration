# random strategy

import numpy as np
from BlackjackSM import BlackjackSM

# Number of games to simulate
num_episodes = 1000

# Initialize the Blackjack state machine
state_machine = BlackjackSM()

# Initialize counters for wins and games played
wins = 0
games_played = 0

# Run the simulation for the specified number of episodes
for _ in range(num_episodes):
    state_machine.new_hand()  # Start a new game
    games_played += 1
    
    while not state_machine.terminal:
        # Randomly choose to hit (0) or stand (1)
        action = np.random.choice([0, 1])
        
        # Perform the action
        state_machine.do(action)
        
        # Check if the game has ended and update wins
        if state_machine.terminal:
            reward = state_machine.reward()
            if reward > 0:
                wins += 1

# Calculate and print the win percentage
win_percentage = (wins / games_played) * 100
print(f"Win percentage: {win_percentage:.2f}%")
print(f"Number of games won: {wins}")
print(f"Number of simulations: {games_played}")
