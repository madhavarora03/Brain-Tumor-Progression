import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BraTSPRODataset
from models.simple_3d_cnn import CNN3D

# Load dataset
dataset = BraTSPRODataset(
    root_dir="data",
    patients_json="data/patients.json",
    target_shape=(128, 128, 128),
    use_registered=True
)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count())

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN3D(in_channels=12, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # Adjust as needed
    print(f"Epoch {epoch + 1}")
    model.train()
    total_loss = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(dim=1) == y).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
