import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import advanced_config

class FakeNewsTransformer:
    def __init__(self, load_saved=False):
        """
        Initializes the Transformer model for Fake News detection.
        If load_saved is True and the model exists, it loads it from disk.
        """
        # Load fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(advanced_config.MODEL_NAME)
        
        # Check if we should load the saved fine-tuned model
        if load_saved and os.path.exists(advanced_config.MODEL_DIR):
            print(f"Loading saved model from {advanced_config.MODEL_DIR}")
            self.model = AutoModelForSequenceClassification.from_pretrained(advanced_config.MODEL_DIR)
            self.tokenizer = AutoTokenizer.from_pretrained(advanced_config.MODEL_DIR)
        else:
            print(f"Loading base pre-trained model: {advanced_config.MODEL_NAME}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                advanced_config.MODEL_NAME, 
                num_labels=2 # 0: Fake, 1: Real
            )
            
    def compute_metrics(self, pred):
        """Compute evaluation metrics during training."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def train(self, train_dataset, eval_dataset):
        """Fine-tunes the model on the provided dataset using Trainer API."""
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=advanced_config.LEARNING_RATE,
            per_device_train_batch_size=advanced_config.BATCH_SIZE,
            per_device_eval_batch_size=advanced_config.BATCH_SIZE,
            num_train_epochs=advanced_config.EPOCHS,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir='./logs',
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the fine-tuned model and tokenizer
        os.makedirs(advanced_config.MODEL_DIR, exist_ok=True)
        self.model.save_pretrained(advanced_config.MODEL_DIR)
        self.tokenizer.save_pretrained(advanced_config.MODEL_DIR)
        print(f"Model saved to {advanced_config.MODEL_DIR}")
        
    def predict(self, text):
        """Predicts the class (Fake/Real) and probability for a single text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=advanced_config.MAX_LENGTH
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
            
        probs_list = probs.tolist()
        predicted_class = np.argmax(probs_list)
        
        return predicted_class, probs_list
        
    def predict_proba(self, texts):
        """
        Returns numpy array of probabilities [P(Fake), P(Real)] 
        for a list of texts (required by LIME explainer).
        """
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=advanced_config.MAX_LENGTH
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        return probs

if __name__ == "__main__":
    print("FakeNewsTransformer loaded. You can import this class to train or test.")
