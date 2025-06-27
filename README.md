# Transformer-Based-Outlier-Detector

A Transformer-based signal prediction module for an outlier detection system of WiFi RSSI fingerprinting in indoor localization. The goal is to enable On-demand calibration by identifying signal outliers that indicate anomalous events, thus preventing costly manual recalibration.

For this work, it is used an encoder-decoder architecture using a Convolutional Block and an Autoregressive Transformer to predict RSSI values by capturing long-range temporal dependencies via self-attention. Evaluations on the SODIndoorLoc and IPIN datasets show strong performance with MSE 0.0001 and NMSE 0.0036 for the first dataset, and MSE 0.0025 and NMSE 0.0564 for the other one, when training and testing are done within the same environment, confirming the model’s ability to generalize in consistent settings. However, performance significantly degrades in cross-building scenarios due to environmental variations.

# Model Architecture 

<img width="1041" alt="Screenshot 2025-06-27 at 23 28 40" src="https://github.com/user-attachments/assets/1d361e96-0499-4bfc-a6a3-3fe6c94f4420" />
