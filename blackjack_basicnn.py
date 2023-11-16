# Basic NN 

import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd
from BlackjackSM import BlackjackSM

# Define the optimal action based on player's hand and dealer's up card
def optimal_action(player_hand_value, dealer_up_card_value, is_soft_hand):
    # The logic of optimal_action goes here
    # For example:
    if is_soft_hand:
        if player_hand_value <= 17:
            return 'hit'
        elif player_hand_value == 18 and dealer_up_card_value in [9, 10, 11]:
            return 'hit'
        else:
            return 'stand'
    else:
        if player_hand_value < 17:
            return 'hit'
        else:
            return 'stand'

# Simulation parameters
num_episodes = 100
data = []

# Initialize the state machine
state_machine = BlackjackSM()

# Run the simulation
for _ in range(num_episodes):
    state_machine.new_hand()

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
        
        # Store state, action, and reward
        next_state = state_machine.state()
        reward = state_machine.reward() if state_machine.terminal else 0
        data.append({
            'player_hand_value': player_hand_value,
            'dealer_up_card_value': dealer_up_card_value,
            'is_soft_hand': is_soft_hand,
            'action': action,
            'reward': reward,
            'terminal': state_machine.terminal
        })

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('blackjack_optimal_strategy_data.csv', index=False)

# Print the first few rows of the dataframe to verify
print(df.head())


# Load your dataset
df = pd.read_csv('blackjack_dataset.csv')

# Preprocess the dataset
feature_list = ['player_total', 'has_ace', 'dealer_card_val']
X = df[feature_list].values
y = df['correct_action'].apply(lambda x: 1 if x == 'hit' else 0).values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network
model = Sequential([
    Dense(16, input_dim=len(feature_list), activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and scaler
model.save('blackjack_nn_model.h5')
np.save('scaler.npy', scaler.mean_)