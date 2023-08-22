import numpy as np
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt


# Generate synthetic data ------TODO Change with real data
np.random.seed(0)
num_samples = 1000              # Number of samples to feed the Neural Network
num_features = 1                # Number of columns in Ground Motion (GM) curve (Just acceleration)
gm_sequence_length = 100        # Maximum length of GM curve
st_sequence_length = 10         # The number of structural parameters

# Inputs data
structural_data = np.random.uniform(0, 1, (num_samples, st_sequence_length, num_features))              # random number between 0 to 1 structural_data (1000, 10, 1)
ground_motion_data = np.random.uniform(0, 1, (num_samples, gm_sequence_length, num_features))           # random number between 0 to 1 ground_motion_data (1000, 100, 1)
# Combine structural_data and ground_motion_data (Generate an array of 110 length)
combined_data = np.concatenate((structural_data, ground_motion_data), axis=1)                           # ground_motion_data + structural_data (1000, 110, 1)

# Output data just X2 the value of input GM just for testing
displacement_data = ground_motion_data * 2                                                              # data X2 ground_motion_data just for testing

# Data preprocessing and sequencing
input_sequences = []
output_sequences = []

for i in range(num_samples):
    input_sequences.append(combined_data[i])
    output_sequences.append(displacement_data[i])

X = np.array(input_sequences)
Y = np.array(output_sequences)

# Splitting into train, validation, and test sets
split_ratio = [0.6, 0.2, 0.2]
split_index_1 = int(split_ratio[0] * len(X))
split_index_2 = int((split_ratio[0] + split_ratio[1]) * len(X))
X_train, X_val, X_test = X[:split_index_1], X[split_index_1:split_index_2], X[split_index_2:]
Y_train, Y_val, Y_test = Y[:split_index_1], Y[split_index_1:split_index_2], Y[split_index_2:]

# Define and compile the model (the encoder-decoder architecture)
model = keras.Sequential([
    # SimpleRNN(32, input_shape=(gm_sequence_length + st_sequence_length, num_features), use_bias=False, activation='relu', return_sequences=True),
    # Encoder
    LSTM(64, activation='relu', input_shape=(gm_sequence_length + st_sequence_length, num_features), return_sequences=True),  #
    # Repeat the output of the encoder for each time step in the output sequence
    # RepeatVector(gm_sequence_length),
    # Decoder
    # LSTM(units=64, return_sequences=True),
    # LSTM(64, activation='relu', return_sequences=True),
    # TimeDistributed layer to apply the same dense layer to each time step
    # TimeDistributed(Dense(units=num_features))
    Dense(1, activation='relu', use_bias=False)
    # Dense(units=gm_sequence_length, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
epochs = 100
batch_size = 32
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)

# Evaluate the model
mse = model.evaluate(X_test, Y_test)
print("Test Mean Squared Error:", mse)

# Make predictions (Generate Random Data to predict the equivalent displacement) # new_acceleration_sequence = X_test[0]
new_acceleration_sequence = np.random.uniform(0, 1, (gm_sequence_length+st_sequence_length, num_features))

# Predict the displacement sequence
predicted_displacement_sequence = model.predict(np.array([new_acceleration_sequence]))

print("New Acceleration Sequence:")
print(new_acceleration_sequence)
print("Predicted Displacement Sequence:")
print(predicted_displacement_sequence)

# Plot the predicted displacement data.
plt.plot(new_acceleration_sequence[0, :, 0]*2, label='True displacement')
plt.plot(predicted_displacement_sequence[0, :, 0], label='Predicted displacement')
plt.legend()
plt.show()
