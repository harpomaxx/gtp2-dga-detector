"""
This script fine tunes a pre-trained GPT2-based model.
"""
import optparse
import datasets
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
import pandas as pd

DEFAULT_MODEL_OUTDIR = "gpt2-dga-detector
DEFAULT_MODEL_SAVEDIR = "./CEPH/gpt2-dga-detector/"
DEFAULT_EXPERIMENT_NAME = "default-experiment"

# Create an OptionParser object
parser = optparse.OptionParser()
parser.add_option("-e", "--experiment_name", dest="experiment_name", default=DEFAULT_EXPERIMENT_NAME, help="Name of the experiment")
parser.add_option("-o", "--output_dir", dest="output_dir", default=DEFAULT_MODEL_OUTDIR, help="Output directory for training and evaluation")
parser.add_option("-s", "--save_dir", dest="save_dir", default=DEFAULT_MODEL_SAVEDIR, help="Directory to save the trained model and tokenizer")
(options, args) = parser.parse_args()

# Function to tokenize the examples
def tokenize_function(examples):
    return tokenizer(examples["domain"])

# Function to compute evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load the dataset
dataset = datasets.load_dataset("harpomaxx/dga-detection")

# Define the model checkpoint and tokenizer
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the datasets
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=["domain", "label"]
)

# Create the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Rename the column "class" to "labels"
tokenized_datasets = tokenized_datasets.rename_column("class", "labels")

# Initialize the model
model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.config.pad_token_id = model.config.eos_token_id  # Fix model padding token ID

# Load the evaluation metric
metric = evaluate.load("f1")

# Define the training arguments
training_args = TrainingArguments(
    output_dir = options.output_dir,
    learning_rate = 2e-5,
    optimizer = "adamw",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 3,
    weight_decay = 0.01,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    push_to_hub = False,
    logging_steps = 10,
    save_total_limit = 3,
    overwrite_output_dir = True
)

# Create the trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['test'],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Save the metrics in a Pandas DataFrame
metrics_df = pd.DataFrame(metrics, index=[options.experiment_name])
metrics_df.to_csv(f"{options.experiment_name}_metrics.csv")

# Save the model and tokenizer
model.save_pretrained(options.save_dir)
tokenizer.save_pretrained(options.save_dir)
