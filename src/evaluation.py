import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, data_loader, device):
    model = model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds)
            labels.extend(label)

    predictions = torch.stack(predictions).cpu()
    labels = torch.stack(labels).cpu()

    print(classification_report(labels, predictions, target_names=["negative", "positive"]))
    print(confusion_matrix(labels, predictions))
