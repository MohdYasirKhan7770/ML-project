import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import advanced_config

class AdvancedDataLoader:
    def __init__(self, model_name=advanced_config.MODEL_NAME, max_length=advanced_config.MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def load_and_prepare_data(self, csv_path, text_col='text', label_col='label'):
        """
        Loads CSV and prepares a HuggingFace Dataset.
        Assumes binary classification where label is 0 or 1.
        """
        df = pd.read_csv(csv_path)
        
        # Clean dataframe
        df = df.dropna(subset=[text_col, label_col])
        df[label_col] = df[label_col].astype(int)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df[[text_col, label_col]])
        
        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_col], 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            )
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # Rename label column to 'labels' for Trainer API compatibility
        if label_col != 'labels' and label_col in tokenized_datasets.column_names:
            tokenized_datasets = tokenized_datasets.rename_column(label_col, 'labels')
            
        tokenized_datasets.set_format('torch')
        return tokenized_datasets
