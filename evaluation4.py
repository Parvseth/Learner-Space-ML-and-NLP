# next_word_predictor_eval.py

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
import torch
import math

# ✅ 1️⃣ Load fine-tuned model and tokenizer
model_path = "./next_word_predictor_gpt2/checkpoint-13167"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 2️⃣ Load dataset and tokenize
dataset = load_dataset("mikasenghaas/wikitext-2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ✅ 3️⃣ Setup DataLoader with data_collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
eval_dataset = tokenized_datasets["train"].select(range(1000))

dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=4,
    collate_fn=data_collator
)

# ✅ 4️⃣ Compute Perplexity
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_tokens += batch_size

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print(f"✅ Perplexity: {perplexity:.4f}")

# ✅ 5️⃣ Compute Top-5 Accuracy
def compute_top_k_accuracy(dataloader, k=5):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            preds = torch.topk(shift_logits, k, dim=-1).indices
            correct += (preds == shift_labels.unsqueeze(-1)).any(-1).sum().item()
            total += shift_labels.numel()
    return correct / total

top_k_acc = compute_top_k_accuracy(dataloader, k=5)
print(f"✅ Top-5 Accuracy: {top_k_acc:.4f}")
