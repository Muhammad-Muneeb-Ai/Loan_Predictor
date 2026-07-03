"""
Extract model weights + scaler params for pure numpy inference.
Model architecture: Dense(32,relu) -> Dense(16,relu) -> Dense(1,sigmoid)
"""
import numpy as np
import pickle
import json
import tensorflow as tf

# 1. Load the Keras model
print("Loading Keras model...")
model = tf.keras.models.load_model('loan_model.keras')
model.summary()

# 2. Extract weights from each layer
print("\nExtracting weights...")
weights_data = {}
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if len(w) == 2:  # Dense layer has [weights, biases]
        weights_data[f'layer_{i}_weights'] = w[0].tolist()
        weights_data[f'layer_{i}_biases'] = w[1].tolist()
        print(f"  Layer {i} ({layer.name}): weights {w[0].shape}, biases {w[1].shape}, activation: {layer.get_config().get('activation', 'none')}")

with open('model_weights.json', 'w') as f:
    json.dump(weights_data, f)
print("Model weights saved to model_weights.json")

# 3. Extract scaler parameters
print("\nExtracting scaler parameters...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'n_features': int(scaler.n_features_in_),
    'feature_names': getattr(scaler, 'feature_names_in_', None)
}
if scaler_params['feature_names'] is not None:
    scaler_params['feature_names'] = scaler_params['feature_names'].tolist()

with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print(f"Scaler params saved. Features: {scaler_params['n_features']}")
print(f"Feature names: {scaler_params['feature_names']}")

# 4. Verify: test with a sample input
print("\nVerifying extraction with a test prediction...")
test_input = np.zeros((1, 11), dtype=np.float32)
keras_pred = model.predict(test_input, verbose=0)[0][0]

# Manual numpy prediction
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = test_input
# Layer 0: Dense(32, relu)
w0 = np.array(weights_data['layer_0_weights'])
b0 = np.array(weights_data['layer_0_biases'])
x = relu(x @ w0 + b0)

# Layer 1: Dense(16, relu)
w1 = np.array(weights_data['layer_1_weights'])
b1 = np.array(weights_data['layer_1_biases'])
x = relu(x @ w1 + b1)

# Layer 2: Dense(1, sigmoid)
w2 = np.array(weights_data['layer_2_weights'])
b2 = np.array(weights_data['layer_2_biases'])
x = sigmoid(x @ w2 + b2)

numpy_pred = x[0][0]

print(f"  Keras prediction:  {keras_pred:.6f}")
print(f"  NumPy prediction:  {numpy_pred:.6f}")
print(f"  Match: {abs(keras_pred - numpy_pred) < 1e-5}")
print("\nDone! Files created: model_weights.json, scaler_params.json")
#
