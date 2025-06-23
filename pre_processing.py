import torch

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


def embeddings_extractor_cnn(signals_tensor, cnn_model):
    """
    Extract embeddings from the signals tensor using a CNN model.
    """
    signals_tensor_reshaped = signals_tensor.unsqueeze(1) 
    with torch.no_grad():
        embeddings = cnn_model(signals_tensor_reshaped)
    return embeddings