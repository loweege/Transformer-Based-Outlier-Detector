import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models import Convolutional_block, AutoregressiveTransformer, model_trainer
from pre_processing import signals_extractor, embeddings_extractor_cnn
from utils import plot_losses, prediction_evaluation
from dataset import SequenceDataset

def main():
    window_size = 30
    seq_len = window_size - 1
    batch_size = 16
    lr = 1e-4
    epochs = 30
    checkpoint_dir = "checkpoints"
    dataset_name = 'Ipin2016Dataset_raw' # 'SODIndoorLoc' - 'Ipin2016Dataset_raw'

    Ipin2016Dataset_raw = [
        'raw_datasets/Ipin2016Dataset/measure1_smartphone_wifi.csv',
        'raw_datasets/Ipin2016Dataset/measure2_smartphone_wifi.csv'
    ]

    SODIndoorLoc_CETC331 = [
        'raw_datasets/SODIndoorLoc-main/CETC331/Testing_CETC331.csv',
        'raw_datasets/SODIndoorLoc-main/CETC331/Training_CETC331.csv'
    ]

    SODIndoorLoc_HCXY = [
        'raw_datasets/SODIndoorLoc-main/HCXY/Testing_HCXY_All.csv',
        'raw_datasets/SODIndoorLoc-main/HCXY/Training_HCXY_All_30.csv'
    ]

    # baseline
    SODIndoorLoc_SYL = [
        'raw_datasets/SODIndoorLoc-main/SYL/Testing_SYL_All.csv',
        'raw_datasets/SODIndoorLoc-main/SYL/Training_SYL_All_30.csv'
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_name == 'SODIndoorLoc':
        datasets_path = {
            'train_path': SODIndoorLoc_CETC331[1],
            'test_path': SODIndoorLoc_HCXY[0]
        }
        train_df = pd.read_csv(datasets_path['train_path'])
        signals_tensor_train_raw, train_ts_tensor = signals_extractor(train_df, dataset_name)

        df_test = pd.read_csv(datasets_path['test_path'])
        signals_tensor_test_raw, test_ts_tensor = signals_extractor(df_test, dataset_name)

        cnn_embed_dim = 128 
        cnn_extractor = Convolutional_block(input_channels=1, output_dim=cnn_embed_dim).to(device)   
        train_embeds = embeddings_extractor_cnn(signals_tensor_train_raw, cnn_extractor).to(device)
        test_embeds = embeddings_extractor_cnn(signals_tensor_test_raw, cnn_extractor).to(device)

    elif dataset_name == 'Ipin2016Dataset_raw':
        datasets_path = {
            'train_path': Ipin2016Dataset_raw[0],
            'test_path': Ipin2016Dataset_raw[0]
        }
        df = pd.read_csv(datasets_path['train_path'])
        signals_tensor_raw, ts_tensor = signals_extractor(df, dataset_name)
        
        train_signal_tensor_raw = signals_tensor_raw[:-121]
        test_signal_tensor_raw = signals_tensor_raw[-120:]

        cnn_embed_dim = 128
        cnn_extractor = Convolutional_block(input_channels=1, output_dim=cnn_embed_dim).to(device) 
        train_embeds = embeddings_extractor_cnn(train_signal_tensor_raw, cnn_extractor).to(device)
        test_embeds = embeddings_extractor_cnn(test_signal_tensor_raw, cnn_extractor).to(device)

    train_dataset = SequenceDataset(train_embeds, window_size=window_size, splits=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SequenceDataset(test_embeds, window_size=window_size, splits=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '-------------------------------signal-predictor-------------------------------'
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