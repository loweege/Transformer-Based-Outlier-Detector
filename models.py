import torch.nn as nn
import torch.nn.functional as F
import torch
import os 

'-------------------------------------embeddings-extractor---------------------------------'
class Convolutional_block(nn.Module):
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

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
            
            # Save only the best model based on test loss
            if checkpoint_dir and test_loader is not None:
                if epoch == 1 or avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"Best model updated and saved to {best_checkpoint_path}")

    else:
        print("Skipping training...")
        model.load_state_dict(torch.load("checkpoints/best_model.pth"))
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
