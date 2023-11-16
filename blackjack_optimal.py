# optimal strategy

from BlackjackSM import BlackjackSM

def optimal_action(player_hand_value, dealer_up_card_value, is_soft_hand):
    if is_soft_hand:
        # Call soft_totals_strategy if the hand is soft
        return soft_totals_strategy(player_hand_value, dealer_up_card_value)
    else:
        # Call hard_totals_strategy if the hand is hard
        return hard_totals_strategy(player_hand_value, dealer_up_card_value)

def soft_totals_strategy(total, dealer_up_card_val):
    if total <= 17:
        return 0  # Hit
    elif total == 18:
        if dealer_up_card_val in [9, 10, 11]:  # Assuming 11 represents an Ace
            return 0  # Hit
        else:
            return 1  # Stand
    else:
        return 1  # Stand

def hard_totals_strategy(total, dealer_up_card_val):
    if total < 12:
        return 0  # Hit
    elif total == 12:
        if dealer_up_card_val in [4, 5, 6]:
            return 1  # Stand
        else:
            return 0  # Hit
    elif 13 <= total <= 16:
        if dealer_up_card_val in [2, 3, 4, 5, 6]:
            return 1  # Stand
        else:
            return 0  # Hit
    else:
        return 1  # Stand

# Simulation parameters
num_episodes = 1000
wins = 0
games_played = 0

# Initialize the state machine
state_machine = BlackjackSM()

# Run the simulation
for _ in range(num_episodes):
    state_machine.new_hand()
    games_played += 1

    while not state_machine.terminal:
        # Extract the necessary information from the state
        current_state = state_machine.state()
        player_hand_value = current_state[0]  # Player's hand value
        is_soft_hand = current_state[1] == 1  # Is the player's hand soft?
        dealer_up_card_value = current_state[4]  # Dealer's up card value
        
        # Determine the action using the optimal strategy
        action = optimal_action(player_hand_value, dealer_up_card_value, is_soft_hand)
        
        # Execute the action
        state_machine.do(action)
        
        # Check if the game has ended and update wins
        if state_machine.terminal:
            reward = state_machine.reward()
            if reward > 0:
                wins += 1

# Calculate and print results
win_percentage = (wins / games_played) * 100
print(f"Win percentage: {win_percentage:.2f}%")
print(f"Number of games won: {wins}")
print(f"Number of simulations: {games_played}")
