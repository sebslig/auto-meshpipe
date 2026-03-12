from datasets import Dataset
import torch

def load_and_preprocess_data(tokenizer, texts: list, labels: list, max_length: int = 128):
    """Loads and preprocesses text data into a suitable format for training."""

    # Create a simple dataset from lists
    data_dict = {
        'text': texts,
        'label': labels
    }
    raw_dataset = Dataset.from_dict(data_dict)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels") # HuggingFace expects 'labels'
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_dataset


