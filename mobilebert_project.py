import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    MobileBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import matplotlib.pyplot as plt


train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

from transformers import MobileBertTokenizer
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")


def tokenize_function(example):
    return tokenizer(example["Review"], truncation=True, padding="max_length", max_length=128)


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["Review"]).rename_column("label", "labels")
val_dataset = val_dataset.remove_columns(["Review"]).rename_column("label", "labels")
train_dataset.set_format("torch")
val_dataset.set_format("torch")


model = MobileBertForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


model.save_pretrained("./mobilebert_spotify")
tokenizer.save_pretrained("./mobilebert_spotify")


log_history = trainer.state.log_history

train_loss = [log["loss"] for log in log_history if "loss" in log]
eval_acc = [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]

plt.figure()
plt.plot(train_loss, label="Training Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("training_loss.png")

plt.figure()
plt.plot(eval_acc, label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("validation_accuracy.png")



model.save_pretrained("saved_model/")
tokenizer.save_pretrained("saved_model/")
print("모델과 토크나이저 저장 완료 (saved_model/)")
