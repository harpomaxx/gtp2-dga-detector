"""
This script performs classification and evaluation using a pre-trained GPT2-based model.
It loads a dataset, makes predictions, calculates evaluation metrics, and saves the metrics in a file.
"""

import optparse
import datasets
from transformers import pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import time

# Set the default value for the model_checkpoint
DEFAULT_MODEL_CHECKPOINT = "/home/harpo/CEPH/gpt2-dga-detector/"
DEFAULT_EXPERIMENT_NAME = "default-experiment"
DEFAULT_WORKING_DIR = "./"
# Create an OptionParser object
parser = optparse.OptionParser()
parser.add_option("-n", "--experiment_name", dest="experiment_name", default = DEFAULT_EXPERIMENT_NAME, help="Name of the experiment")
parser.add_option("-m", "--model_checkpoint", dest="model_checkpoint", default= DEFAULT_MODEL_CHECKPOINT, help="Path or name of the pre-trained model checkpoint")
parser.add_option("-o", "--n_obs", dest="n_obs", type=int, help="Number of observations to consider")
parser.add_option("-b", "--batch_size", dest="batch_size", default = 1024, type=int, help="Batch Size")
parser.add_option("-w", "--working_dir", dest="working_dir", default=DEFAULT_WORKING_DIR, help="Working directory to save the metrics and predictions files")
(options, args) = parser.parse_args()



# Load the dataset
dataset = datasets.load_dataset("harpomaxx/dga-detection")

# Calculate the total size of the dataset
total_size = len(dataset['validation'])

# Set the number of observations to consider
n_obs = options.n_obs if options.n_obs is not None else total_size

# Load the text classification pipeline
classifier = pipeline("text-classification", model=options.model_checkpoint, device=0, batch_size=options.batch_size)

# Shuffle and select a subset of the validation dataset
dataset = dataset['validation'].shuffle(seed=42)

# Print header
print("GPT2-DGA-DETECTOR:\n")
# Print selected parameters
print("Selected Parameters:")
print(f"Experiment Name: {options.experiment_name}")
print(f"Model Checkpoint: {options.model_checkpoint}")
print(f"Number of Observations: {n_obs}")
print(f"Batch Size: {options.batch_size}\n")
print(f"Working Directory: {options.working_dir}\n")

print("[] Calculating predictions...")
start_time = time.time()  # Start timer
# Make predictions on the selected subset
predictions = classifier(dataset['domain'][1:n_obs])
end_time = time.time()  # End timer
elapsed_time = end_time - start_time
print("[] Done.")

# Extract the predicted labels and convert them to numpy array
predicted_labels = np.array([pred['label'] for pred in predictions])
predicted_labels = np.where(predicted_labels == 'LABEL_0', 0, 1)

# Extract the true labels from the dataset and convert them to numpy array
true_labels = np.array(dataset['class'][1:n_obs])

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Calculate the evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)

print("[] Results")
print(cm)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall Score: {recall}")
print(f"Precision Score: {precision}")

print(f"\nElapsed Time: {elapsed_time} seconds")

# Save the predictions to a CSV file
#filename_predictions = f"{options.experiment_name}_predictions.csv"
filename_predictions = os.path.join(options.working_dir, f"{options.experiment_name}_predictions.csv")

with open(filename_predictions, "w") as file:
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(filename_predictions, index=False)

# Save the metrics in a separate file
#filename_metrics = f"{options.experiment_name}_metrics.txt"
filename_metrics = os.path.join(options.working_dir, f"{options.experiment_name}_metrics.txt")

with open(filename_metrics, "w") as file:
    file.write("Confusion Matrix:\n")
    file.write(np.array2string(cm, separator=', '))
    file.write("\n\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"Recall Score: {recall}\n")
    file.write(f"Precision Score: {precision}\n")