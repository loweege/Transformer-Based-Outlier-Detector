import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

'--------------------------------------pre-processing----------------------------------'
def signals_extractor(df):
    columns = df.columns[:-1]
    #mac_columns = [col for col in df.columns if "MAC" in col]
    vectors = df[columns].values
    sample_times = df['SampleTimes'].values

    signals_tensor = torch.tensor(vectors, dtype=torch.float32)
    ts_tensor = torch.tensor(sample_times, dtype=torch.float32).squeeze()
    return signals_tensor, ts_tensor

def embeddings_extractor(signals_tensor):
    """
    Extract embeddings from the MAC address tensor using PCA.
    """
    pca = PCA(n_components=370)
    signals_tensor = pca.fit_transform(signals_tensor)
    return torch.tensor(signals_tensor, dtype=torch.float32)

'-----------------------------------------dataset--------------------------------------'
class SequenceDataset(Dataset):
    """
    Dataset that generates sliding windows of fixed length (e.g., 30),
    splits each window into k parts, and for each part, uses the first (n-1)
    embeddings as input and the last as the target.
    """
    def __init__(self, embeddings: torch.Tensor, window_size: int = 30, splits: int = 3):
        assert window_size % splits == 0, "Window size must be divisible by number of splits"
        self.embeds = embeddings
        self.window_size = window_size
        self.split_size = window_size // splits
        self.splits = splits
        self.num_windows = embeddings.size(0) - window_size + 1

    def __len__(self):
        return self.num_windows * self.splits

    def __getitem__(self, idx):
        window_idx = idx // self.splits
        split_idx = idx % self.splits
        window = self.embeds[window_idx : window_idx + self.window_size]
        
        part = window[split_idx * self.split_size : (split_idx + 1) * self.split_size]
        x = part[:-1]
        y = part[-1]
        return x, y

'-------------------------------------model-definition---------------------------------'
class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class AutoregressiveTransformer(nn.Module):
    """
    Simple autoregressive Transformer model to predict the next embedding.
    """
    def __init__(self, 
                 embed_dim: int = 256, 
                 nhead: int = 10, 
                 num_layers: int = 3,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, src: torch.Tensor):
        src = src.transpose(0, 1)
        src = self.positional_encoding(src)
        encoded = self.transformer_encoder(src)
        last = encoded[-1] 
        return self.fc_out(last)

'--------------------------------------model-trainer----------------------------------'
def model_trainer(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int = 200,
    test_loader: torch.utils.data.DataLoader = None,
    device: torch.device = None,
    training: bool = True,
    checkpoint_dir: str = None
):
    """
    Trains `model` on `train_loader` and evaluates on `test_loader` after every epoch if provided.

    Returns:
        train_losses: List of average training loss per epoch.
        test_losses:  List of average test loss per epoch (empty if no test_loader).
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []

    if training:
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            n_samples = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                total_train_loss += loss.item() * batch_size
                n_samples += batch_size
                #train_loss = loss.item() * batch_size
                #train_losses.append(train_loss)

            avg_train_loss = total_train_loss / n_samples
            train_losses.append(avg_train_loss)

            # Evaluate test loss during training
            if test_loader is not None:
                model.eval()
                total_test_loss = 0.0
                n_test_samples = 0
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)

                        preds = model(x_batch)
                        loss = criterion(preds, y_batch)

                        batch_size = x_batch.size(0)
                        total_test_loss += loss.item() * batch_size
                        n_test_samples += batch_size
                        #test_loss = loss.item() * batch_size
                        #test_losses.append(test_loss)

                avg_test_loss = total_test_loss / n_test_samples
                test_losses.append(avg_test_loss)
                print(f"Epoch {epoch:02d}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            else:
                print(f"Epoch {epoch:02d}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    else:
        print("Skipping training...")
        model.load_state_dict(torch.load("checkpoints/model_epoch_006.pth"))
        # Optional test loss evaluation after loading
        if test_loader is not None:
            model.eval()
            total_test_loss = 0.0
            n_test_samples = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)

                    batch_size = x_batch.size(0)
                    total_test_loss += loss.item() * batch_size
                    n_test_samples += batch_size

            avg_test_loss = total_test_loss / n_test_samples
            test_losses.append(avg_test_loss)
            print(f"Test Loss: {avg_test_loss:.4f}")

    return train_losses, test_losses

'--------------------------------------plotting---------------------------------------'
def plot_losses(train_losses, test_losses):
    # Plot losses after training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(test_losses, label='Test Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("losses.png")
    plt.show()

'--------------------------------------prediction-------------------------------------'
def predict_next_signal(model: torch.nn.Module, input_sequence: torch.Tensor, device: torch.device):
    """
    Predicts the next signal given an input sequence using the trained model.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        input_sequence = input_sequence.to(device)
        prediction = model(input_sequence.unsqueeze(0))  # Add batch dimension
    return prediction.squeeze(0).cpu()  # Remove batch dimension and move to CPU

def calculate_nmse(actual: torch.Tensor, predicted: torch.Tensor):
    """
    Calculates the Normalized Mean Squared Error (NMSE).
    """
    mse = F.mse_loss(predicted, actual, reduction='sum')
    actual_power = torch.sum(actual**2)
    if actual_power == 0:
        return float('inf') # Avoid division by zero
    nmse = mse / actual_power
    return nmse.item()

def main():

    window_size = 30
    seq_len = window_size - 1
    batch_size = 32
    lr = 1e-4
    epochs = 6
    checkpoint_dir = "checkpoints"


    csv_path = 'datasets/SODIndoorLoc-main/SYL/Training_SYL_All_30.csv'
    df = pd.read_csv(csv_path)
    mac_tensor, ts_tensor = signals_extractor(df)
    embeds = embeddings_extractor(mac_tensor)


    # Set the standard deviation of the noise (e.g., 0.1)
    std_dev = 0.1
    # Generate Gaussian noise and add it
    noise = torch.randn_like(embeds) * std_dev
    noisy_embedding = embeds + noise
    combined_embeddings = torch.cat([embeds, noisy_embedding], dim=0)
    # Optionally duplicate the timestamps if needed
    combined_timestamps = torch.cat([ts_tensor, ts_tensor], dim=0)
    embeds = combined_embeddings
    ts_tensor = combined_timestamps



    dataset = SequenceDataset(embeds, window_size=window_size, splits=5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embed_dim = embeds.size(1)
    model = AutoregressiveTransformer(embed_dim=embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    df_test = pd.read_csv('datasets/SODIndoorLoc-main/SYL/Testing_SYL_All.csv')
    mac_tensor_test, ts_tensor_test = signals_extractor(df_test)
    embeds_test = embeddings_extractor(mac_tensor_test)

    dataset_test = SequenceDataset(embeds_test, window_size=window_size, splits=3)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_losses, test_losses = model_trainer(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        test_loader=loader_test,
        training=True,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    print("Training and evaluation complete.")

    plot_losses(train_losses, test_losses)

    # Predict the next signal
    # Get one sample from the test dataset for prediction
    for i in range(100):
        sample_input_sequence, actual_next_signal = dataset_test[i] # Take the first sample
        
        # Ensure the sample input sequence has the correct shape for the model (seq_len, embed_dim)
        # The dataset returns x as (split_size - 1, embed_dim)
        
        predicted_next_signal = predict_next_signal(model, sample_input_sequence, device)

        print("\n--- Prediction ---")
        print(f"Sample Input Sequence Shape: {sample_input_sequence.shape}")
        print(f"Actual Next Signal Shape: {actual_next_signal.shape}")
        print(f"Predicted Next Signal Shape: {predicted_next_signal.shape}")
        # You might want to print a comparison or calculate a metric for the prediction
        print(f"Actual Next Signal (first 5 elements): {actual_next_signal[:5]}")
        print(f"Predicted Next Signal (first 5 elements): {predicted_next_signal[:5]}")
        
        # Calculate Mean Squared Error for the single prediction
        prediction_mse = F.mse_loss(predicted_next_signal, actual_next_signal)
        print(f"Prediction MSE: {prediction_mse.item():.4f}")

        # Calculate Normalized Mean Squared Error (NMSE)
        prediction_nmse = calculate_nmse(actual_next_signal, predicted_next_signal)
        print(f"Prediction NMSE: {prediction_nmse:.4f}")

    print('done')

if __name__ == "__main__":
    main()