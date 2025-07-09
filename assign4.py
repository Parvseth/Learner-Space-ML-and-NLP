# Week 4 - Wrap Up Project: Next Word Predictor using Transformers (GPT-2)
# Clean, practical, ready-to-run fine-tuning script for WikiText-2 and OpenWebText using Hugging Face

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import math
import torch

dataset_choice = "wikitext-2"  # change to "openwebtext-10k" if needed

# 1️⃣ Load dataset
if dataset_choice == "wikitext-2":
    dataset = load_dataset("mikasenghaas/wikitext-2")
else:
    dataset = load_dataset("stas/openwebtext-10k")

# 2️⃣ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure padding token alignment for DataCollator (GPT-2 has no pad token by default)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# 3️⃣ Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4️⃣ Prepare DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5️⃣ Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./next_word_predictor_gpt2",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    logging_steps=50,
)

# 6️⃣ Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"].select(range(1000)),  # use a subset for quick evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7️⃣ Train the model
trainer.train()

# 8️⃣ Evaluate: Perplexity and top-k accuracy
import numpy as np

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")

# (Optional) Top-k Accuracy calculation
def compute_top_k_accuracy(eval_dataset, k=5):
    correct = 0
    total = 0
    model.eval()
    for batch in torch.utils.data.DataLoader(eval_dataset, batch_size=4):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(model.device)
            labels = input_ids.clone()
            outputs = model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            preds = torch.topk(shift_logits, k, dim=-1).indices
            correct += (preds == shift_labels.unsqueeze(-1)).any(-1).sum().item()
            total += shift_labels.numel()
    return correct / total

top_k_acc = compute_top_k_accuracy(tokenized_datasets["train"].select(range(1000)), k=5)
print(f"Top-5 Accuracy: {top_k_acc:.4f}")

# Done: You now have a fine-tuned GPT-2 next-word predictor for your wrap-up project.