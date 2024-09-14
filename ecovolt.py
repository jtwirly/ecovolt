import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## Step 2: Data Collection and Preprocessing

# Load the dataset
data = pd.read_csv('energy_data.csv', parse_dates=['timestamp'])
print(data.head())
print(data.info())

# Display the first few rows
print(data.head())

# Forward-fill missing values
data.fillna(method='ffill', inplace=True)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data (excluding the timestamp)
scaled_features = scaler.fit_transform(data.drop('timestamp', axis=1))

# Convert back to DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[1:])
scaled_data['timestamp'] = data['timestamp']

# d. Create Time-Series Sequences 
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length].drop('timestamp', axis=1).values
        target = data.iloc[i+seq_length]['consumption']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Set sequence length (e.g., past 24 hours)
seq_length = 24
sequences, targets = create_sequences(scaled_data, seq_length)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    sequences, targets, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False)

# Convert to tensors
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Step 3: Defining the Liquid Neural Network Architecture

#a. Define the Liquid Time-Constant Layer
class LiquidTimeConstantLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeConstantLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.constant_(self.tau, 1.0)

    def forward(self, x, h_prev):
        # Update rule mimicking LNN dynamics
        h_new = h_prev + (torch.tanh(self.W_in(x) + self.W_rec(h_prev)) - h_prev) / self.tau
        return h_new
    
#b. Define the Liquid Neural Network Model
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_layer = LiquidTimeConstantLayer(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        for t in range(x.size(1)):
            h = self.liquid_layer(x[:, t, :], h)
        out = self.fc(h)
        return out

#c. Instantiate the Model
# Define input, hidden, and output sizes
input_size = X_train.shape[2]  # Number of features
hidden_size = 64
output_size = 1  # Predicting consumption

# Initialize the model
model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

##Step 4: Training the Network
#a. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#b. Training Loop
num_epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {np.mean(train_losses):.4f}, '
          f'Val Loss: {np.mean(val_losses):.4f}')

##Step 5: Model Evaluation
#a. Evaluate on Test Set
model.eval()
test_losses = []
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_losses.append(loss.item())

        predictions.extend(outputs.squeeze().cpu().numpy())
        actuals.extend(y_batch.squeeze().cpu().numpy())

print(f'Test Loss: {np.mean(test_losses):.4f}')

#b. Denormalize Predictions
# Get index of 'consumption' in the original data
consumption_index = data.columns.get_loc('consumption') - 1  # Adjust for timestamp

# Retrieve scale and min parameters
scale = scaler.scale_[consumption_index]
min_ = scaler.min_[consumption_index]

# Denormalize
predictions_denorm = np.array(predictions) * scale + min_
actuals_denorm = np.array(actuals) * scale + min_

#c. Calculate Performance Metrics
mae = mean_absolute_error(actuals_denorm, predictions_denorm)
mse = mean_squared_error(actuals_denorm, predictions_denorm)
rmse = np.sqrt(mse)
r2 = r2_score(actuals_denorm, predictions_denorm)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R^2 Score: {r2:.4f}')

#d. Plot Predictions vs. Actuals
plt.figure(figsize=(12,6))
plt.plot(predictions_denorm[:200], label='Predicted')
plt.plot(actuals_denorm[:200], label='Actual')
plt.legend()
plt.title('Predicted vs Actual Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.show()

#Step 6: Deployment and Integration
#a. Save the Model
torch.save(model.state_dict(), 'lnn_energy_model.pth')

#b. Load the Model for Inference
# Re-instantiate the model and load weights
model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('lnn_energy_model.pth'))
model.eval()

#c. Real-Time Prediction Function
def predict_next_consumption(sequence):
    sequence = torch.Tensor(sequence).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
    # Denormalize
    prediction = output.item() * scale + min_
    return prediction

#Test the function with a sample sequence from your data.
# Select a sequence from the test set
test_sequence = X_test[0]  # First sequence in test set

# Make a prediction
predicted_consumption = predict_next_consumption(test_sequence)

# Denormalize actual consumption
actual_consumption = y_test[0] * scale + min_

print(f'Predicted Consumption: {predicted_consumption:.2f}')
print(f'Actual Consumption: {actual_consumption:.2f}')

#d. Integrate with Grid Systems
#Integration would involve:
#API Development: Create an API endpoint that takes input data and returns predictions.
#Data Pipeline: Set up real-time data ingestion using tools like Apache Kafka.
#Control Systems: Connect predictions to grid control mechanisms for automated decision-making.

#Step 7: Additional Considerations
#a. Continuous Learning
#Set up mechanisms to retrain the model with new data periodically.

#b. Model Optimization
#Hyperparameter Tuning: Use techniques like grid search or Bayesian optimization.
#Regularization: Apply dropout or L1/L2 regularization to prevent overfitting.
#Advanced Architectures: Explore using Neural ODEs or actual LNN implementations as they become available.
#c. Security and Compliance
#Data Security: Ensure secure handling of sensitive data.
#Regulatory Compliance: Adhere to energy industry regulations.
#d. Scalability
#Consider deploying the model using scalable infrastructure like cloud services