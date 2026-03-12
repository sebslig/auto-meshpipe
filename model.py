from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def tokenize(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

