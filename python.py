# Import libraries
import torch
from arabert.preprocess import ArabertPreprocessor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import streamlit as st
from pyngrok import ngrok
import subprocess
import os
dataset = load_dataset('arcd')

print(dataset)
# --- Define Model & Tokenizer ---

# TODO: Define the model_name (use "aubmindlab/bert-base-arabertv2")
model_name = "aubmindlab/bert-base-arabertv2"

# TODO: Load the tokenizer from the model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Set Hyperparameters ---

# TODO: Set a max length (e.g., 384)
max_length = 384
# TODO: Set a doc stride (e.g., 128)
doc_stride = 128

def preprocess_function(examples):
    # Strip leading/trailing whitespace from questions
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
    questions,                # الأسئلة بالعربية
    examples["context"],                 # النصوص أو الفقرات
    max_length=384,           # أقصى طول للمدخلات
    truncation="only_second", # نقطع من الـ context فقط إذا كان طويل
    stride=128,               # يسمح بتداخل أثناء التقطيع لتغطية الإجابات الطويلة
    return_overflowing_tokens=True,   # نرجع الأجزاء الزائدة (مهمة للـ QA)
    return_offsets_mapping=True,      # نحتاجها لتحديد موقع الإجابة داخل النص
    padding="max_length"              # نضيف padding إلى أقصى طول ثابت
)

    # Get the offset mapping to find the start/end tokens
    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")

    # Get the answers
    answers = examples["answers"]

    # Initialize lists for our new labels
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        # Get the ID of the original example
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]

        # Get the answer's start and end character positions
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Get the sequence IDs (0=question, 1=context, None=special tokens)
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # --- TODO: Find the start and end token positions ---
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)  # آخر توكن يمر عليه قبل بداية الإجابة

            # 2️⃣ تحديد موقع نهاية الإجابة داخل التوكنات
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    # TODO: Add the start_positions and end_positions to the `inputs` dictionary
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

# TODO: Apply the preprocessing function to the dataset using .map()
# Remember to set batched=True and remove the old columns
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print(tokenized_datasets)
#@title Ignore warnings

import warnings
from transformers.utils import logging

# Suppress the specific FutureWarning about the 'tokenizer' argument
warnings.filterwarnings("ignore", message="`tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`.*")

# Suppress the UserWarning about 'pin_memory' when no GPU is found
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator is found.*")

# Set transformers logging level to 'error' to hide the info messages
# This will hide the "Some weights..." and "tokenizer has new PAD..." messages
logging.set_verbosity_error()
# --- Load the Model ---

# TODO: Load the pre-trained model for Question Answering
# Use AutoModelForQuestionAnswering.from_pretrained() with your model_name
model = AutoModelForQuestionAnswering.from_pretrained("aubmindlab/bert-base-arabertv2")

# Define where to save the model
output_dir = "./arabert_qa_results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# --- Define Training Arguments ---

# TODO: Create an instance of TrainingArguments
# We have filled in the recommended settings for you.
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,

    # Batch size settings (to prevent Colab from crashing)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
    fp16=True,                      # Use mixed precision (faster, less memory)

    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",               # Disables wandb login
)

# --- Create the Trainer ---

# TODO: Create an instance of the Trainer
# It needs the model, args, train_dataset, eval_dataset, and tokenizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # TODO: Use the 'train' split
    eval_dataset=tokenized_datasets["validation"], # TODO: Use the 'validation' split
    tokenizer=tokenizer,
)

print("\nStarting model training ...")
# TODO: Start the training
trainer.train()


print("Training complete. Saving model...")
# TODO: Save the final model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

...
print(f"Model saved to {output_dir}")