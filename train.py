import copy
import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models
from data import load_data


data_dir = "/content/drive/MyDrive/Processing_Image/Data"
train_loader, val_loader, class_index = load_data(data_dir)

# (CPU)
device = torch.device("cpu")

# Mô hình
model = models.densenet161(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)  # 2 classes
model.to(device)

# Loss function và optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Scheduler (điều chỉnh learning rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ======================================
#   Định nghĩa các hàm bổ sung
# ======================================

def training_step(model, loader, loss_function):
    model.train()
    epoch_loss = 0
    epoch_correct = 0

    for images, labels in iter(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(output, dim=1)

        epoch_loss += loss.item() * images.size(0)
        epoch_correct += torch.sum(predictions == labels)

    epoch_loss = epoch_loss / len(loader.dataset)
    accuracy = epoch_correct.double() / len(loader.dataset)
    return epoch_loss, accuracy


def evaluate_model(model, loader, loss_function):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0

    for images, labels in iter(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            output = model(images)
            loss = loss_function(output, labels)
            _, predictions = torch.max(output, dim=1)

        epoch_loss += loss.item() * images.size(0)
        epoch_correct += torch.sum(predictions == labels)

    epoch_loss = epoch_loss / len(loader.dataset)
    accuracy = epoch_correct.double() / len(loader.dataset)
    return epoch_loss, accuracy


# ======================================
#   Vòng lặp huấn luyện
# ======================================

# Số epoch
epochs = 15
best_val_loss = float('inf')
best_model = copy.deepcopy(model.state_dict())

# Lưu kết quả
train_loss_savings = []
train_acc_savings = []
val_loss_savings = []
val_acc_savings = []

# Huấn luyện
for epoch in range(epochs):
    train_loss, train_acc = training_step(model, train_loader, loss_function)
    train_loss_savings.append(train_loss)
    train_acc_savings.append(train_acc.item())

    val_loss, val_acc = evaluate_model(model, val_loader, loss_function)
    val_loss_savings.append(val_loss)
    val_acc_savings.append(val_acc.item())

    print(f"Epoch {epoch+1:02}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        print(f"Epoch {epoch+1:02} - val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model.")
        best_val_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())
    else:
        print(f"Epoch {epoch+1:02} - val_loss did not improve.")

    scheduler.step()

# ======================================
#   Model Saved
# ======================================
torch.save(best_model, 'best-model-weighted.pt')
print("Mô hình tốt nhất đã được lưu.")
