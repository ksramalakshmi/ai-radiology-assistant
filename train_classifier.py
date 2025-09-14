import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from data_utils import get_loaders
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml

with open("classifier_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = f"cuda:{cfg['model']['device']}" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

train_loader, val_loader = get_loaders(cfg)

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, cfg['model']['num_classes'])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['step_size'], gamma=float(cfg['training']['gamma']))

best_val_acc = 0
patience_counter = 0
epochs = cfg['training']['epochs']
early_stopping_patience = cfg['training']['early_stopping_patience']

for epoch in range(epochs):
    model.train()
    train_losses = []
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    scheduler.step()
    avg_train_loss = sum(train_losses) / len(train_losses)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
        print("Saved best model!")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(os.listdir(cfg['data']['train_dir'])),
            yticklabels=sorted(os.listdir(cfg['data']['train_dir'])))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
