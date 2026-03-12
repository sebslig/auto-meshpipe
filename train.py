import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from model import TransformerClassifier
from data import load_and_preprocess_data

# --- Configuration --- #
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_SEQUENCE_LENGTH = 128

# Dummy Data
dummy_texts = [
    "This is a positive sentence about the product.",
    "I hate this brand, it was terrible.",
    "Neutral feedback, neither good nor bad.",
    "Absolutely fantastic service, highly recommend!",
    "The worst experience ever, purely negative."
]
dummy_labels = [1, 0, 1, 1, 0] # 1 for positive/neutral, 0 for negative

# --- Main Training Script --- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = TransformerClassifier(model_name=MODEL_NAME, num_labels=NUM_LABELS).to(device)
    print(f"Model initialized with {MODEL_NAME}")

    # 2. Load and Preprocess Data
    dataset = load_and_preprocess_data(model.tokenizer, dummy_texts, dummy_labels, MAX_SEQUENCE_LENGTH)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} samples")

    # 3. Setup Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                avg_batch_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_samples
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {avg_batch_loss:.4f}, Accuracy: {accuracy:.4f}")

        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}\n")

    print("Training complete.")
    # TODO: Add model saving functionality
