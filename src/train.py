import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import torch_directml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ======================
# RUTAS ROBUSTAS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# ======================
# DEVICE (AMD GPU)
# ======================
device = torch_directml.device()
print("Usando dispositivo:", device)

# ======================
# CONFIG
# ======================
BATCH_SIZE = 8
IMG_SIZE = 224
EPOCHS = 10

# ======================
# TRANSFORMACIONES
# ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ======================
# DATASETS
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# MODELO (EfficientNet)
# ======================
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)

model = model.to(device)

# ======================
# LOSS Y OPTIMIZADOR
# ======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ======================
# ENTRENAMIENTO
# ======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ======================
# EVALUACIÓN
# ======================
model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ======================
# MÉTRICAS
# ======================
print("\n📊 Reporte de clasificación:")
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("\n📉 Matriz de confusión:")
print(cm)

sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)

print(f"\n🧠 Sensibilidad (TB): {sensibilidad:.4f}")
print(f"🧠 Especificidad: {especificidad:.4f}")

auc = roc_auc_score(all_labels, all_probs)
print(f"📈 AUC: {auc:.4f}")

# ======================
# CURVA ROC
# ======================
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.show()

# ======================
# GUARDAR MODELO
# ======================
MODEL_PATH = os.path.join(MODEL_DIR, "model_pytorch.pth")

torch.save(model.state_dict(), MODEL_PATH)

print(f"\n✅ Modelo guardado en: {MODEL_PATH}")