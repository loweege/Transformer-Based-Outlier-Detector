import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

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
    filename = f"pics/loss_plot_{train_name}_vs_{test_name}.png"
    
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