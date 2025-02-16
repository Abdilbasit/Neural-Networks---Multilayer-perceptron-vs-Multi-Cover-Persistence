Original file is located at
 https://colab.research.google.com/drive/1WCz5euEauFRSftPUDt5FelwzFg2c1ce-?usp=sharing#scrollTo=2STaaueCS2Kk

MCP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palmerpenguins as penguins
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns


class MulticlassPerceptron:
    def __init__(self, input_features, output_classes, learning_rate):
        self.weights = np.random.randn(input_features, output_classes) * 0.01
        self.biases = np.zeros((1, output_classes))
        self.learning_rate = learning_rate

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            logits = np.dot(X, self.weights) + self.biases
            probs = self.softmax(logits)

            m = X.shape[0]
            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(m), y] = 1

            gradient = (probs - y_one_hot) / m
            self.weights -= self.learning_rate * np.dot(X.T, gradient)
            self.biases -= self.learning_rate * np.sum(gradient, axis=0, keepdims=True)

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.biases
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

data = penguins.load_penguins().dropna()
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values
y = data['species'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Part A:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

learning_rate = 0.001
initial_epochs = 100
model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=learning_rate)
train_error_rates = []
test_error_rates = []


for epoch in range(initial_epochs):
    model.train(X_train, y_train, 1)


    train_predictions = model.predict(X_train)
    train_error_rate = 1 - np.mean(train_predictions == y_train)
    train_error_rates.append(train_error_rate)


    test_predictions = model.predict(X_test)
    test_error_rate = 1 - np.mean(test_predictions == y_test)
    test_error_rates.append(test_error_rate)


results_table_a = pd.DataFrame({
    'Epoch': range(1, initial_epochs + 1),
    'Training Error Rate': train_error_rates,
    'Testing Error Rate': test_error_rates
})
print("Results for Part A:")
print(results_table_a.head(10))


plt.figure(figsize=(10, 6))
plt.plot(range(initial_epochs), train_error_rates, label="Training Error Rate", color="blue")
plt.plot(range(initial_epochs), test_error_rates, label="Testing Error Rate", color="orange")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Error Rate", fontsize=12)
plt.title("Error Rates vs Epochs (Part A)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()



max_epochs = 2000
increment = 100
all_error_rates = {}

test_accuracies = []
for epochs in range(100, max_epochs + 1, increment):
    model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=learning_rate)
    model.train(X_train, y_train, epochs)

    test_predictions = model.predict(X_test)
    accuracy = np.mean(test_predictions == y_test)
    test_accuracies.append(accuracy)
    all_error_rates[epochs] = 1 - accuracy

optimal_epochs = max_epochs


# Part B:

learning_rates = np.linspace(0.001, 5, 50)
train_error_rates_lr = []
test_error_rates_lr = []

for lr in learning_rates:
    model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=lr)
    model.train(X_train, y_train, optimal_epochs)

    train_predictions = model.predict(X_train)
    train_error_rate = 1 - np.mean(train_predictions == y_train)
    train_error_rates_lr.append(train_error_rate)

    test_predictions = model.predict(X_test)
    test_error_rate = 1 - np.mean(test_predictions == y_test)
    test_error_rates_lr.append(test_error_rate)


min_test_error_lr = min(test_error_rates_lr)
optimal_lr = learning_rates[test_error_rates_lr.index(min_test_error_lr)]


plt.figure(figsize=(10, 6))
plt.plot(learning_rates, train_error_rates_lr, label="Training Error Rate", color="blue")
plt.plot(learning_rates, test_error_rates_lr, label="Testing Error Rate", color="orange")
plt.axvline(x=optimal_lr, color="red", linestyle="--", label=f"Optimal LR = {optimal_lr:.3f}")
plt.xlabel("Learning Rate", fontsize=12)
plt.ylabel("Error Rate", fontsize=12)
plt.title("Error Rates vs Learning Rate (Part B)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# Part C:
splitting_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
accuracies = []

for ratio in splitting_ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=ratio, random_state=42)

    model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=optimal_lr)
    model.train(X_train, y_train, optimal_epochs)

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    accuracies.append(accuracy)


results_table = pd.DataFrame({
    'Training Sample Size (%)': [int(ratio * 100) for ratio in splitting_ratios],
    'Accuracy': accuracies
})
print(results_table)

plt.figure(figsize=(10, 6))
plt.bar([f"{int(ratio * 100)}%" for ratio in splitting_ratios], accuracies, color="skyblue")
plt.xlabel("Training Sample Size (%)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs Training Sample Size (Part C)", fontsize=14)
plt.ylim(0.0, 1.0)
plt.grid(axis='y', alpha=0.5)
plt.show()



# Part D:
# Split data into Training, Validation, and Test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


optimal_learning_rate = optimal_lr

train_error_rates = []
val_error_rates = []

model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=optimal_learning_rate)


for epoch in range(optimal_epochs):
    model.train(X_train, y_train, 1)  # Train for one epoch at a time

    # Calculate training error rate
    train_predictions = model.predict(X_train)
    train_error_rate = 1 - np.mean(train_predictions == y_train)
    train_error_rates.append(train_error_rate)

    # Calculate validation error rate
    val_predictions = model.predict(X_val)
    val_error_rate = 1 - np.mean(val_predictions == y_val)
    val_error_rates.append(val_error_rate)

# Part D Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(optimal_epochs), train_error_rates, label="Training Error Rate", color="blue")
plt.plot(range(optimal_epochs), val_error_rates, label="Validation Error Rate", color="green")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Error Rate", fontsize=12)
plt.title("Error Rates vs Epochs (Part D)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# Evaluate final performance on the test set
final_test_predictions = model.predict(X_test)
final_test_accuracy = np.mean(final_test_predictions == y_test)
print(f"Final Test Set Accuracy: {final_test_accuracy * 100:.2f}%")



# Part E: Use of Cross-Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

cv_train_error_rates = []
cv_val_error_rates = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = MulticlassPerceptron(input_features=X_train.shape[1], output_classes=len(np.unique(y)), learning_rate=optimal_learning_rate)

    fold_train_error_rates = []
    fold_val_error_rates = []

    for epoch in range(optimal_epochs):
        model.train(X_train, y_train, 1)

        # Calculate training error rate
        train_predictions = model.predict(X_train)
        train_error_rate = 1 - np.mean(train_predictions == y_train)
        fold_train_error_rates.append(train_error_rate)

        # Calculate validation error rate
        val_predictions = model.predict(X_val)
        val_error_rate = 1 - np.mean(val_predictions == y_val)
        fold_val_error_rates.append(val_error_rate)

    cv_train_error_rates.append(fold_train_error_rates)
    cv_val_error_rates.append(fold_val_error_rates)

# Calculate average training and validation error rates across folds
avg_train_error_rates = np.mean(cv_train_error_rates, axis=0)
avg_val_error_rates = np.mean(cv_val_error_rates, axis=0)

# Part E Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(optimal_epochs), avg_train_error_rates, label="Average Training Error Rate", color="blue")
plt.plot(range(optimal_epochs), avg_val_error_rates, label="Average Validation Error Rate", color="green")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Error Rate", fontsize=12)
plt.title("Error Rates vs Epochs (Part E: Cross-Validation)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()


final_test_predictions = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, final_test_predictions)

# Visualize confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix (Part F: Test Set Analysis)", fontsize=14)
plt.show()


misclassified_indices = np.where(final_test_predictions != y_test)[0]
misclassified_cases = pd.DataFrame({
    'True Label': y_test[misclassified_indices],
    'Predicted Label': final_test_predictions[misclassified_indices],
    'Features': [X_test[i] for i in misclassified_indices]
})
print("Misclassified Cases (Part F):")
print(misclassified_cases.head(10))  # Display the first 10 misclassified cases

# Aggregate insights on misclassifications
print("Insights from Confusion Matrix:")
for i, true_label in enumerate(label_encoder.classes_):
    true_positive = conf_matrix[i, i]
    false_negative = sum(conf_matrix[i, :]) - true_positive
    false_positive = sum(conf_matrix[:, i]) - true_positive
    print(f"Class {true_label}:")
    print(f"  True Positives: {true_positive}")
    print(f"  False Negatives: {false_negative}")
    print(f"  False Positives: {false_positive}")



MLP



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palmerpenguins as penguins
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Define activation functions and their derivatives
def relu_activation(inputs):
    return np.maximum(0, inputs)

def relu_derivative(inputs):
    return (inputs > 0).astype(float)

def softmax_activation(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def compute_cross_entropy_loss(true_labels, predicted_probs):
    epsilon = 1e-15
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    num_samples = true_labels.shape[0]
    log_likelihood = -np.log(predicted_probs[range(num_samples), true_labels])
    return np.sum(log_likelihood) / num_samples

# Define the Custom Multi-Layer Perceptron class
class NeuralNetwork:
    def __init__(self, layer_sizes, learn_rate):
        self.layer_sizes = layer_sizes
        self.learn_rate = learn_rate
        self.weights_list = []
        self.biases_list = []
        self._initialize_params()

    def _initialize_params(self):
        for i in range(len(self.layer_sizes) - 1):
            self.weights_list.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2. / self.layer_sizes[i]))
            self.biases_list.append(np.zeros((1, self.layer_sizes[i+1])))

    def forward_propagation(self, inputs):
        activations = [inputs]
        z_values = []
        for i in range(len(self.weights_list)):
            z = np.dot(activations[-1], self.weights_list[i]) + self.biases_list[i]
            z_values.append(z)
            activation = softmax_activation(z) if i == len(self.weights_list) - 1 else relu_activation(z)
            activations.append(activation)
        return z_values, activations

    def backward_propagation(self, z_values, activations, true_labels):
        gradients_weights = [np.zeros_like(w) for w in self.weights_list]
        gradients_biases = [np.zeros_like(b) for b in self.biases_list]

        delta = activations[-1]
        delta[range(true_labels.shape[0]), true_labels] -= 1
        gradients_weights[-1] = np.dot(activations[-2].T, delta)
        gradients_biases[-1] = np.sum(delta, axis=0)

        for l in range(2, len(self.layer_sizes)):
            z = z_values[-l]
            delta = np.dot(delta, self.weights_list[-l+1].T) * relu_derivative(z)
            gradients_weights[-l] = np.dot(activations[-l-1].T, delta)
            gradients_biases[-l] = np.sum(delta, axis=0)

        return gradients_weights, gradients_biases

    def update_params(self, gradients_weights, gradients_biases):
        for i in range(len(self.weights_list)):
            self.weights_list[i] -= self.learn_rate * gradients_weights[i]
            self.biases_list[i] -= self.learn_rate * gradients_biases[i]

    def train_model(self, inputs, true_labels, num_epochs):
        for epoch in range(num_epochs):
            z_values, activations = self.forward_propagation(inputs)
            gradients_weights, gradients_biases = self.backward_propagation(z_values, activations, true_labels)
            self.update_params(gradients_weights, gradients_biases)

    def predict_labels(self, inputs):
        _, activations = self.forward_propagation(inputs)
        return np.argmax(activations[-1], axis=1)

# Load and preprocess dataset
data = penguins.load_penguins().dropna()
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])
features = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values
labels = data['species'].values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Part A: Initial Setup
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
learning_rate = 0.001
epochs_to_train = 100

# hidden layer configuration
hidden_layers = [8]
network_structure = [X_train.shape[1]] + hidden_layers + [len(np.unique(labels))]

mlp_network = NeuralNetwork(network_structure, learning_rate)
training_error_rates, testing_error_rates = [], []

for epoch in range(epochs_to_train):
    mlp_network.train_model(X_train, y_train, 1)
    training_preds = mlp_network.predict_labels(X_train)
    testing_preds = mlp_network.predict_labels(X_test)

    train_error = 1 - np.mean(training_preds == y_train)
    test_error = 1 - np.mean(testing_preds == y_test)

    training_error_rates.append(train_error)
    testing_error_rates.append(test_error)

plt.plot(range(epochs_to_train), training_error_rates, label="Training Error")
plt.plot(range(epochs_to_train), testing_error_rates, label="Testing Error")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.legend()
plt.title("Error Rates vs Epochs")
plt.show()

# Display the fixed model structure
print(f"Model Structure: {network_structure}")
print("Training complete.")

# Part B: Optimal Learning Rate
epochs_for_learning_rate = 200
learning_rate_values = np.linspace(0.001, 1, 20)
learning_rate_error_rates = []

for lr in learning_rate_values:
    temp_network = NeuralNetwork(network_structure, lr)
    temp_network.train_model(X_train, y_train, epochs_for_learning_rate)
    predictions = temp_network.predict_labels(X_test)
    error_rate = 1 - np.mean(predictions == y_test)
    learning_rate_error_rates.append(error_rate)

optimal_learning_rate = learning_rate_values[np.argmin(learning_rate_error_rates)]
plt.plot(learning_rate_values, learning_rate_error_rates, label="Test Error")
plt.axvline(optimal_learning_rate, color='red', linestyle='--', label=f"Optimal LR: {optimal_learning_rate}")
plt.xlabel("Learning Rate")
plt.ylabel("Error Rate")
plt.legend()
plt.title("Error Rate vs Learning Rate")
plt.show()

# Part C: Training Sample Size
epochs_for_final_training = 300
training_data_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
accuracy_by_sample_size = []

for ratio in training_data_ratios:
    X_partial_train, _, y_partial_train, _ = train_test_split(X_train, y_train, train_size=ratio, random_state=42)
    sample_network = NeuralNetwork(network_structure, optimal_learning_rate)
    sample_network.train_model(X_partial_train, y_partial_train, epochs_for_final_training)
    predictions = sample_network.predict_labels(X_test)
    accuracy = np.mean(predictions == y_test)
    accuracy_by_sample_size.append(accuracy)

plt.bar([f"{int(ratio*100)}%" for ratio in training_data_ratios], accuracy_by_sample_size, color="skyblue")
plt.xlabel("Training Sample Size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Training Sample Size")
plt.show()

# Part D: Use of a Separate Validation Set
X_train_full, X_temp, y_train_full, y_temp = train_test_split(scaled_features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

validation_network = NeuralNetwork(network_structure, optimal_learning_rate)
training_errors, validation_errors = [], []

for epoch in range(epochs_for_final_training):
    validation_network.train_model(X_train_full, y_train_full, 1)

    training_predictions = validation_network.predict_labels(X_train_full)
    validation_predictions = validation_network.predict_labels(X_val)

    training_error = 1 - np.mean(training_predictions == y_train_full)
    validation_error = 1 - np.mean(validation_predictions == y_val)

    training_errors.append(training_error)
    validation_errors.append(validation_error)

plt.plot(range(epochs_for_final_training), training_errors, label="Training Error")
plt.plot(range(epochs_for_final_training), validation_errors, label="Validation Error")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.legend()
plt.title("Training vs Validation Error Rates")
plt.show()

final_test_predictions = validation_network.predict_labels(X_test)
final_test_accuracy = np.mean(final_test_predictions == y_test)
print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")

# Part E: Cross-Validation
folds = 5
cross_validator = KFold(n_splits=folds, shuffle=True, random_state=42)
cross_train_errors, cross_validation_errors = [], []

for train_indices, val_indices in cross_validator.split(scaled_features):
    X_fold_train, X_fold_val = scaled_features[train_indices], scaled_features[val_indices]
    y_fold_train, y_fold_val = labels[train_indices], labels[val_indices]

    fold_network = NeuralNetwork(network_structure, optimal_learning_rate)
    fold_train_errors, fold_val_errors = [], []

    for epoch in range(epochs_for_final_training):
        fold_network.train_model(X_fold_train, y_fold_train, 1)

        training_predictions = fold_network.predict_labels(X_fold_train)
        validation_predictions = fold_network.predict_labels(X_fold_val)

        training_error = 1 - np.mean(training_predictions == y_fold_train)
        validation_error = 1 - np.mean(validation_predictions == y_fold_val)

        fold_train_errors.append(training_error)
        fold_val_errors.append(validation_error)

    cross_train_errors.append(fold_train_errors)
    cross_validation_errors.append(fold_val_errors)

avg_training_errors = np.mean(cross_train_errors, axis=0)
avg_validation_errors = np.mean(cross_validation_errors, axis=0)

plt.plot(range(epochs_for_final_training), avg_training_errors, label="Avg Training Error")
plt.plot(range(epochs_for_final_training), avg_validation_errors, label="Avg Validation Error")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.legend()
plt.title("Cross-Validation Training vs Validation Error Rates")
plt.show()

# Part F: Deeper Analysis of Observed Performance
# Confusion Matrix Analysis
conf_matrix = confusion_matrix(y_test, final_test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report for Detailed Analysis
classification_results = classification_report(y_test, final_test_predictions, target_names=label_encoder.classes_)
print("Classification Report:")
print(classification_results )
