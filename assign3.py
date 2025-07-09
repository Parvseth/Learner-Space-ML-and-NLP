# sentiment_pipeline.py
"""
Assignment 3: Machine Learning Pipeline with Hugging Face
Sentiment Analysis using BERT on IMDb Dataset
"""

import os
import warnings

# Suppress TensorFlow warnings - must be set before importing any ML libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism warnings

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import pipeline

# Suppress general warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    """
    Main function to run the sentiment analysis pipeline
    """
    
    # ============================================================================
    # STEP 1: SETUP AND DEVICE CONFIGURATION
    # ============================================================================
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS PIPELINE - ASSIGNMENT 3")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    # ============================================================================
    # STEP 2: LOAD IMDB DATASET
    # ============================================================================
    
    print("\nStep 1: Loading IMDb dataset...")
    try:
        dataset = load_dataset("imdb")
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        print(f"Sample review: {dataset['train'][0]['text'][:100]}...")
        print(f"Sample label: {dataset['train'][0]['label']} (0=negative, 1=positive)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # ============================================================================
    # STEP 3: LOAD TOKENIZER AND MODEL
    # ============================================================================
    
    print("\nStep 2: Loading BERT tokenizer and model...")
    model_checkpoint = "bert-base-uncased"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        print(f"Tokenizer loaded: {model_checkpoint}")
        
        # Load model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, 
            num_labels=2  # Binary classification: positive/negative
        ).to(device)
        print(f"Model loaded and moved to {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return
    
    # ============================================================================
    # STEP 4: PREPROCESS DATASET (TOKENIZATION)
    # ============================================================================
    
    print("\nStep 3: Preprocessing dataset (tokenization)...")
    
    def tokenize_function(examples):
        """
        Tokenize the input texts using BERT tokenizer
        """
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,  # BERT's maximum sequence length
        )
        # Ensure labels are preserved
        tokenized["labels"] = examples["label"]
        return tokenized
    
    try:
        # Tokenize the datasets - IMPORTANT: Keep the labels column!
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True,
            batch_size=1000,
            remove_columns=["text"]  # Only remove text column, keep labels
        )
        
        print("Tokenization completed successfully!")
        print(f"Tokenized training samples: {len(tokenized_datasets['train'])}")
        print(f"Tokenized test samples: {len(tokenized_datasets['test'])}")
        print(f"Available columns: {tokenized_datasets['train'].column_names}")
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return
    
    # ============================================================================
    # STEP 5: PREPARE TRAINING SETUP
    # ============================================================================
    
    print("\nStep 4: Setting up training configuration...")
    
    # Use smaller subsets for faster training (as per your original code)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    print(f"Training subset size: {len(train_dataset)}")
    print(f"Evaluation subset size: {len(eval_dataset)}")
    
    # Debug: Check the first sample to ensure labels are present
    print("\nDebug - First training sample:")
    first_sample = train_dataset[0]
    print(f"Keys in sample: {list(first_sample.keys())}")
    if 'labels' in first_sample:
        print(f"Label value: {first_sample['labels']}")
    else:
        print("WARNING: No 'labels' key found in the sample!")
        return
    
    # Define metrics computation
    def compute_metrics(eval_pred):
        """
        Compute accuracy and F1-score metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./sentiment_model",
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        save_total_limit=2,
        report_to=None,  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # ============================================================================
    # STEP 6: FINE-TUNE THE MODEL
    # ============================================================================
    
    print("\nStep 5: Fine-tuning BERT model...")
    print("This may take several minutes...")
    
    try:
        # Start training
        trainer.train()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # ============================================================================
    # STEP 7: EVALUATE THE MODEL
    # ============================================================================
    
    print("\nStep 6: Evaluating model performance...")
    
    try:
        # Evaluate on test set
        eval_results = trainer.evaluate()
        
        print("=" * 50)
        print("EVALUATION RESULTS:")
        print("=" * 50)
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # ============================================================================
    # STEP 8: SAVE THE FINE-TUNED MODEL
    # ============================================================================
    
    print("\nStep 7: Saving fine-tuned model...")
    
    try:
        # Save model and tokenizer
        model_save_path = "./sentiment_finetuned_bert"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Model saved successfully to: {model_save_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # ============================================================================
    # STEP 9: LOAD MODEL FOR INFERENCE
    # ============================================================================
    
    print("\nStep 8: Loading model for inference...")
    
    try:
        # Load the saved model
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            model_save_path
        ).to(device)
        
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        
        # Create inference pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )
        
        print("Model loaded successfully for inference!")
        
    except Exception as e:
        print(f"Error loading model for inference: {e}")
        return
    
    # ============================================================================
    # STEP 10: DEMONSTRATE INFERENCE ON SAMPLE TEXTS
    # ============================================================================
    
    print("\nStep 9: Demonstrating inference on sample texts...")
    print("=" * 60)
    
    # Sample texts for testing
    sample_texts = [
        "The movie was surprisingly touching and beautifully acted!",
        "This film was absolutely terrible and boring.",
        "I loved every minute of this fantastic movie!",
        "The plot was confusing but the acting was decent.",
        "Best movie I've seen all year! Highly recommend!",
        "Waste of time. Poor acting and terrible storyline."
    ]
    
    try:
        for i, text in enumerate(sample_texts, 1):
            result = sentiment_pipeline(text)
            
            # Get the prediction with highest score
            prediction = max(result[0], key=lambda x: x['score'])
            
            # Map labels to sentiment
            sentiment = "Positive" if prediction['label'] == 'LABEL_1' else "Negative"
            confidence = prediction['score']
            
            print(f"\nSample {i}:")
            print(f"Text: {text}")
            print(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # ============================================================================
    # CLEANUP
    # ============================================================================
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nPipeline execution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()