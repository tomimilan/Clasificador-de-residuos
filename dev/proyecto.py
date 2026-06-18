# -*- coding: utf-8 -*-
import os
import zipfile
import shutil
import glob
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# 1. RESOLUCIÓN DINÁMICA DE LA RAÍZ DEL PROYECTO
# ==============================================================================
cwd = os.getcwd()
if os.path.basename(cwd) == "dev":
    os.chdir("..")
    print("📍 Ejecución local desde /dev detectada. Ajustando raíz al repositorio principal.")
else:
    print(f"📍 Ejecución desde la raíz del proyecto: {cwd}")

os.makedirs("data/raw", exist_ok=True)
os.makedirs("dev", exist_ok=True)
os.makedirs("prod", exist_ok=True)

print(f"📁 Directorio raíz de trabajo unificado en: {os.path.abspath(os.getcwd())}\n")

# ==============================================================================
# 2. DESCARGA AUTOMATIZADA Y CONDICIONAL DEL DATASET
# ==============================================================================
os.system("python -m pip install gdown -q")
import gdown

drive_file_id = '1pNlHIgPxUm0GoqhOUlLUy6Ow1-y1jfWx'
local_zip_path = "Residuos_contenedores.zip"

try:
    if not os.listdir("data/raw"):
        print(f"📥 'data/raw' está vacío. Descargando dataset desde Drive ID: {drive_file_id}...")
        gdown.download(id=drive_file_id, output=local_zip_path, quiet=False)

        print("📦 Descomprimiendo en data/raw/...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw")
        print("✓ Dataset listo para el pipeline.")
        os.remove(local_zip_path)
    else:
        print("✓ El dataset ya existe en 'data/raw/', omitiendo descarga automática.")
except Exception as e:
    print(f"⚠️ Error en la sincronización de datos: {e}")

# ==============================================================================
# 3. CONFIGURACIÓN DEL ARCHIVO .GITIGNORE
# ==============================================================================
gitignore_content = "data/raw/\n*.zip\n*.jpg\n*.jpeg\n*.png\n__pycache__/\n.ipynb_checkpoints/"
with open(".gitignore", "w") as f:
    f.write(gitignore_content.strip())
print("✓ Archivo .gitignore verificado en la raíz.")

# ==============================================================================
# 4. MAPEO DE CLASES Y SPLITS (CSV)
# ==============================================================================
CONTENEDORES_NAMES = {
    0: "VERDE (Secos reciclables)",
    1: "NEGRO (Humedos rechazo)",
    2: "AMARILLO (Electronicos rechazo parcial)",
    3: "MARRON (Textil especial)"
}

REVERSE_CONTENEDORES_NAMES = {v: k for k, v in CONTENEDORES_NAMES.items()}

extensiones = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
archivos_imagenes = []
for ext in extensiones:
    archivos_imagenes.extend(glob.glob(f"data/raw/**/*/{ext}", recursive=True))

data_rows = []
for filepath in archivos_imagenes:
    container_folder_name = os.path.basename(os.path.dirname(filepath))
    if container_folder_name in REVERSE_CONTENEDORES_NAMES:
        label = REVERSE_CONTENEDORES_NAMES[container_folder_name]
        ruta_relativa = os.path.relpath(filepath, start=".")
        data_rows.append({"filepath": ruta_relativa, "label": label})
    else:
        print(f"Advertencia: Nombre de carpeta '{container_folder_name}' no reconocido. Saltando {filepath}")

df_total = pd.DataFrame(data_rows)

if df_total.empty:
    print("Error: No se encontraron imágenes válidas. Revisa la estructura del dataset.")
else:
    print("Distribución total por Contenedor destino:")
    print(df_total['label'].value_counts().rename(index=CONTENEDORES_NAMES))

    SEED = 31
    train_df, val_test_df = train_test_split(df_total, test_size=0.30, random_state=SEED, stratify=df_total['label'])
    val_df, test_df = train_test_split(val_test_df, test_size=0.50, random_state=SEED, stratify=val_test_df['label'])

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"\nSplits generados con éxito dentro de data/")

# ==============================================================================
# 5. DATASET Y DATALOADERS PYTORCH
# ==============================================================================
class GarbageGIRSUDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        label = int(self.data_frame.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

IMG_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

train_dataset = GarbageGIRSUDataset("data/train.csv", transform=train_transforms)
val_dataset   = GarbageGIRSUDataset("data/val.csv", transform=val_test_transforms)
test_dataset  = GarbageGIRSUDataset("data/test.csv", transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Hardware y Pérdida Ponderada
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en el dispositivo: {device}")

conteo_clases = train_df['label'].value_counts().sort_index().values
total_muestras = len(train_df)
pesos_clases = total_muestras / (len(conteo_clases) * conteo_clases)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(pesos_clases).to(device))

# ==============================================================================
# 6. MODELADO Y ENTRAINAMIENTO
# ==============================================================================
def inicializar_modelo_efficientnet(num_clases=4, entrenar_completo=True):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = entrenar_completo
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_clases)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += images.size(0)
    return running_loss / total, (correct.double() / total).item()

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += images.size(0)
    return running_loss / total, (correct.double() / total).item()

# ==============================================================================
# 7. EJECUCIÓN DE EXPERIMENTOS
# ==============================================================================
experimentos = [
    {"nombre": "Feature Extractor", "entrenar_completo": False, "lr": 1e-4},
    {"nombre": "Fine Tuning", "entrenar_completo": True, "lr": 1e-4},
    {"nombre": "Fine Tuning LR Bajo", "entrenar_completo": True, "lr": 1e-5}
]

resultados = []
global_best_val_acc = 0.0

for exp in experimentos:
    print("\n" + "="*70)
    print(f"EXPERIMENTO: {exp['nombre']}")
    print("="*70)

    model = inicializar_modelo_efficientnet(num_clases=4, entrenar_completo=exp["entrenar_completo"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-2)
    best_exp_val_acc = 0.0
    start_time = time.time()

    for epoch in range(5): # 5 épocas para experimentos rápidos en local
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"[{epoch+1}/5] Train Acc={t_acc*100:.2f}% | Val Acc={v_acc*100:.2f}%")
        
        if v_acc > best_exp_val_acc:
            best_exp_val_acc = v_acc

    resultados.append({
        "Experimento": exp["nombre"], "Learning Rate": exp["lr"],
        "Fine Tuning": exp["entrenar_completo"], "Best Val Acc (%)": round(best_exp_val_acc * 100, 2)
    })

print("\n=================== TABLA COMPARATIVA ===================")
df_resultados = pd.DataFrame(resultados)
print(df_resultados) # CORREGIDO: display cambiado por print para consola nativa
df_resultados.to_csv("comparacion_experimentos.csv", index=False)

# ==============================================================================
# 8. ENTRENAMIENTO FINAL DEL GANADOR
# ==============================================================================
print("\n" + "═"*60 + "\n🚀 INICIANDO ENTRENAMIENTO FINAL DEL GANADOR\n" + "═"*60)
BEST_LR = 1e-4
BEST_FINE_TUNING = True
EPOCHS_FINAL = 10

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
model = inicializar_modelo_efficientnet(num_clases=4, entrenar_completo=BEST_FINE_TUNING).to(device)
optimizer = optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=1e-2)
best_val_acc = 0.0

for epoch in range(EPOCHS_FINAL):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"[{epoch+1}/{EPOCHS_FINAL}] Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}%")

    # CORREGIDO: Sangría estándar de 4 espacios y guardado físico en dev/
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "dev/modelo_final_ganador.pth")
        print(f"💾 Guardado óptimo en 'dev/modelo_final_ganador.pth'")

print(f"\n🏆 Proceso finalizado. Mejor Validation Accuracy: {best_val_acc*100:.2f}%")

# ==============================================================================
# 9. EVALUACIÓN DE TEST Y AUDITORÍA
# ==============================================================================
model.load_state_dict(torch.load("dev/modelo_final_ganador.pth", map_location=device))
model.eval()
print("\n✓ Modelo cargado desde /dev para auditoría final.")

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(f"\nAccuracy Final en Test Set: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(CONTENEDORES_NAMES.values())))