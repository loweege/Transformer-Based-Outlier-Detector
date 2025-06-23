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
def signals_extractor(df, dataset):
    if dataset == 'SODIndoorLoc':
        columns = df.columns[:-1]
        vectors = df[columns].values
        sample_times = df['SampleTimes'].values
        signals_tensor = torch.tensor(vectors, dtype=torch.float32)
        ts_tensor = torch.tensor(sample_times, dtype=torch.float32).squeeze()
        return signals_tensor, ts_tensor
    
    if dataset == 'Ipin2016Dataset_raw':
        timestamps = df.iloc[:, 0]
        signals = df.iloc[:, 1:]
        ts_tensor = torch.tensor(timestamps.values, dtype=torch.float32)
        signals_tensor = torch.tensor(signals.values, dtype=torch.float32)
        return signals_tensor, ts_tensor

class CNNExtractor(nn.Module):
    """
    CNN-based embedding extractor.
    Takes signals as input and outputs fixed-size embeddings.
    """
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1) # Global average pooling to get fixed-size output
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Permute to (batch_size, features, sequence_length) for Conv1d
        #x = x.permute(0, 2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1) # Output shape (batch_size, 128)
        x = self.fc(x)
        return x

def embeddings_extractor_cnn(signals_tensor, cnn_model):
    """
    Extract embeddings from the signals tensor using a CNN model.
    """
    signals_tensor_reshaped = signals_tensor.unsqueeze(1) # Adds a channel dimension: (num_samples, 1, num_features)
    
    with torch.no_grad():
        embeddings = cnn_model(signals_tensor_reshaped)
    return embeddings


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
    Autoregressive Transformer model to predict the next embedding.
    """
    def __init__(self, 
                 embed_dim: int = 256, 
                 nhead: int = 8, 
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

                avg_test_loss = total_test_loss / n_test_samples
                test_losses.append(avg_test_loss)
                print(f"Epoch {epoch:02d}/{epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
            else:
                print(f"Epoch {epoch:02d}/{epochs}, Train Loss: {avg_train_loss:.6f}")
            
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
def plot_losses(train_losses, test_losses, train_path, test_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(test_losses, label='Test Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    
    train_name = train_path.split('/')[-1]
    test_name = test_path.split('/')[-1]
    filename = f"loss_plot_{train_name}_vs_{test_name}.png"
    
    plt.savefig(filename)
    plt.show()
'--------------------------------------prediction-------------------------------------'
def predict_next_signal(model: torch.nn.Module, input_sequence: torch.Tensor, device: torch.device):
    """
    Predicts the next signal given an input sequence using the trained model.
    """
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.to(device)
        prediction = model(input_sequence.unsqueeze(0))
    return prediction.squeeze(0).cpu()

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

def prediction_evaluation(model, test_dataset, device):
    print("\n--- Individual Predictions and Metrics ---")
    all_prediction_mse = []
    all_prediction_nmse = []
    num_predictions_to_evaluate = len(test_dataset)

    for i in range(num_predictions_to_evaluate):
        sample_input_sequence, actual_next_signal = test_dataset[i]
        predicted_next_signal = predict_next_signal(model, sample_input_sequence, device)

        # Calculate MSE and NMSE for the single prediction
        prediction_mse = F.mse_loss(predicted_next_signal, actual_next_signal).item()
        all_prediction_mse.append(prediction_mse)
        prediction_nmse = calculate_nmse(actual_next_signal, predicted_next_signal)
        all_prediction_nmse.append(prediction_nmse)

        if i < 10:
            print(f"\n--- Prediction Sample {i+1} ---")
            print(f"Sample Input Sequence Shape: {sample_input_sequence.shape}")
            print(f"Actual Next Signal Shape: {actual_next_signal.shape}")
            print(f"Predicted Next Signal Shape: {predicted_next_signal.shape}")
            print(f"Actual Next Signal (first 5 elements): {actual_next_signal[:5]}")
            print(f"Predicted Next Signal (first 5 elements): {predicted_next_signal[:5]}")
            print(f"Prediction MSE: {prediction_mse:.4f}")
            print(f"Prediction NMSE: {prediction_nmse:.4f}")

    average_mse = np.mean(all_prediction_mse)
    average_nmse = np.mean([nmse for nmse in all_prediction_nmse if nmse != float('inf')])
    
    print("\n--- Average Metrics ---")
    print(f"Average Prediction MSE over {num_predictions_to_evaluate} samples: {average_mse:.4f}")
    print(f"Average Prediction NMSE over {num_predictions_to_evaluate} samples: {average_nmse:.4f}")


def main():
    window_size = 30
    seq_len = window_size - 1
    batch_size = 16
    lr = 1e-4
    epochs = 30
    checkpoint_dir = "checkpoints"
    dataset_name = 'SODIndoorLoc' # or 'SODIndoorLoc'

    Ipin2016Dataset_raw = [
        'datasets/Ipin2016Dataset/measure1_smartphone_wifi.csv',
        'datasets/Ipin2016Dataset/measure2_smartphone_wifi.csv'
    ]

    SODIndoorLoc_CETC331 = [
        'datasets/SODIndoorLoc-main/CETC331/Testing_CETC331.csv',
        'datasets/SODIndoorLoc-main/CETC331/Training_CETC331.csv'
    ]

    SODIndoorLoc_HCXY = [
        'datasets/SODIndoorLoc-main/HCXY/Testing_HCXY_All.csv',
        'datasets/SODIndoorLoc-main/HCXY/Training_HCXY_All_30.csv'
    ]

    # this the baseline
    SODIndoorLoc_SYL = [
        'datasets/SODIndoorLoc-main/SYL/Testing_SYL_All.csv',
        'datasets/SODIndoorLoc-main/SYL/Training_SYL_All_30.csv'
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_name == 'SODIndoorLoc':
        datasets_path = {
            'train_path': SODIndoorLoc_HCXY[1],
            'test_path': SODIndoorLoc_HCXY[0]
        }
        train_df = pd.read_csv(datasets_path['train_path'])
        signals_tensor_train_raw, train_ts_tensor = signals_extractor(train_df, dataset_name)

        df_test = pd.read_csv(datasets_path['test_path'])
        signals_tensor_test_raw, test_ts_tensor = signals_extractor(df_test, dataset_name)

        # Determine input_channels for CNN
        input_channels_cnn = signals_tensor_train_raw.shape[1] # Number of features
        # Define the output dimension of the CNN embeddings
        cnn_embed_dim = 128 # You can tune this
        cnn_extractor = CNNExtractor(input_channels=1, output_dim=cnn_embed_dim).to(device) # input_channels=1 because we unsqueeze
        
        # Extract embeddings using CNN
        train_embeds = embeddings_extractor_cnn(signals_tensor_train_raw, cnn_extractor).to(device)
        test_embeds = embeddings_extractor_cnn(signals_tensor_test_raw, cnn_extractor).to(device)

        # Adding noise for SODIndoorLoc, if desired (now applied to the CNN embeddings)
        '''std_dev = 0.1
        noise = torch.randn_like(train_embeds) * std_dev
        noisy_embedding = train_embeds + noise
        train_embeds = torch.cat([train_embeds, noisy_embedding], dim=0)'''

    elif dataset_name == 'Ipin2016Dataset_raw':
        datasets_path = {
            'train_path': Ipin2016Dataset_raw[0],
            'test_path': Ipin2016Dataset_raw[0]
        }
        df = pd.read_csv(Ipin2016Dataset_raw[0])
        signals_tensor_raw, ts_tensor = signals_extractor(df, dataset_name)
        
        train_signal_tensor_raw = signals_tensor_raw[:-121]
        test_signal_tensor_raw = signals_tensor_raw[-120:]

        # Determine input_channels for CNN
        input_channels_cnn = train_signal_tensor_raw.shape[1] # Number of features
        # Define the output dimension of the CNN embeddings
        cnn_embed_dim = 120 # Matching the original PCA output dimension for consistency in this example
        cnn_extractor = CNNExtractor(input_channels=1, output_dim=cnn_embed_dim).to(device) # input_channels=1 because we unsqueeze
        
        # Extract embeddings using CNN
        train_embeds = embeddings_extractor_cnn(train_signal_tensor_raw, cnn_extractor).to(device)
        test_embeds = embeddings_extractor_cnn(test_signal_tensor_raw, cnn_extractor).to(device)

    train_dataset = SequenceDataset(train_embeds, window_size=window_size, splits=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SequenceDataset(test_embeds, window_size=window_size, splits=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '-------------------------------outlier-predictor-------------------------------'
    # The embed_dim for the Transformer should now be the output dimension of the CNN
    embed_dim = cnn_embed_dim 
    model = AutoregressiveTransformer(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    training = True
    train_losses, test_losses = model_trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        test_loader=test_loader,
        training=training,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    print("Training and evaluation complete.")

    if training:
        plot_losses(train_losses, 
                    test_losses, 
                    train_path=datasets_path['train_path'], 
                    test_path=datasets_path['test_path'])

    prediction_evaluation(model=model, 
                          test_dataset=test_dataset, 
                          device=device)


if __name__ == "__main__":
    main()

    '''
    TO DO:
    - Do fine tuning to train the model with the other building data. If it is not possible retrain the model with the same architecture to check if it is general 
    - build a pipeline that run over all of the datasets
    - take the best epoch not the 6th
    '''

    #training on another building