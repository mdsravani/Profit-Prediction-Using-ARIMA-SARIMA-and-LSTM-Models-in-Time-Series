import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import asyncio

# Fix asyncio issue

# Fix for asyncio event loop issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


st.title("LSTM Model for Profit Prediction")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load the data
data = pd.read_csv("refined_profit.csv")

# Fix SettingWithCopyWarning
data = data.copy()
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)

# Extract target variable
target = data['Total Profit'].values.astype(float)

# Normalize the target variable
scaler = MinMaxScaler(feature_range=(-1, 1))
target_normalized = scaler.fit_transform(target.reshape(-1, 1))

# Define sequence length
sequence_length = 30

# Create sequences of data
sequences = []
for i in range(len(target_normalized) - sequence_length):
    sequences.append(target_normalized[i:i+sequence_length+1])

# Convert sequences to numpy array
sequences = np.array(sequences)

# Split into input and output
X = sequences[:, :-1]
y = sequences[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=42)

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Reshape input data
input_size = X_train.shape[-1]  # Number of features
X_train = X_train.view(-1, sequence_length, input_size)
X_test = X_test.view(-1, sequence_length, input_size)

# Extract dates for plotting
dates = data.index[-len(y_test):]

# Hyperparameters
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 2000
learning_rate = 0.001

# Training function
def train_model():
    progress_bar = st.progress(0)

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        progress_bar.progress((epoch + 1) / num_epochs)

    st.success("Training complete!")
    return model

if st.button("TRAIN THE MODEL"):
    trained_model = train_model()
    torch.save(trained_model.state_dict(), 'lstm_model.pth')
    st.success("Model saved successfully!")

# Load the trained model
loaded_model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))
loaded_model.eval()

if "show_prediction_ui" not in st.session_state:
    st.session_state.show_prediction_ui = False  # To track if prediction UI should be visible

if st.button("Predict Future Values"):
    st.session_state.show_prediction_ui = True  # Keep UI elements visible

# Ensure the prediction UI remains visible
if st.session_state.show_prediction_ui:

    future_days = st.number_input(
        'Enter number of days to predict:',
        min_value=1, max_value=30, value=st.session_state.get("future_days", 1), key="days_input"
    )

    if st.button('Submit'):
        st.session_state.future_days = future_days  # Store value in session_state
        st.session_state.predicted = True  # Flag to show predictions

    # Ensure predictions persist after reruns
    if st.session_state.get("predicted", False):
        extended_sequence = []
        last_sequence = X_test[-1].clone()

        for _ in range(st.session_state.future_days):
            with torch.no_grad():
                output = loaded_model(last_sequence.unsqueeze(0))
                extended_sequence.append(output.cpu().numpy())  # Convert tensor to NumPy
                last_sequence = torch.cat((last_sequence[1:], output), axis=0).detach()

        # Inverse transform the actual values
        actual_profit = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

        # Inverse transform the future predictions
        future_predicted_profit = scaler.inverse_transform(np.array(extended_sequence).reshape(-1, 1))

        # Generate future dates
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=st.session_state.future_days, freq='D')

        # Plot actual and future predictions
        plt.figure(figsize=(10, 5))
        plt.plot(dates, actual_profit, label='Actual Profit', color='blue')
        plt.plot(future_dates, future_predicted_profit, label='Forecast', color='orange', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('Profit')
        plt.title('Actual and Forecasted Profit')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
