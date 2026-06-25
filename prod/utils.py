import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import streamlit as st

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Etiquetas limpias y unificadas con el script de desarrollo
CONTENEDORES_NAMES = {
    0: "VERDE (Secos reciclables)",
    1: "NEGRO (Húmedos rechazo, comida, basura)",
    2: "AMARILLO (Electrónicos rechazo parcial)",
    3: "MARRÓN (Textil especial)"
}

@st.cache_resource
def cargar_modelo(path_pesos):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=4)
    
    try:
        state_dict = torch.load(path_pesos, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        state_dict = torch.load(path_pesos, map_location=torch.device('cpu'))
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocesar_imagen(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform_pipeline(image.convert("RGB")).unsqueeze(0)

def predecir(model, tensor_imagen):
    with torch.no_grad():
        outputs = model(tensor_imagen)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    prob_dict = {CONTENEDORES_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
    best_class_idx = torch.argmax(probabilities).item()
    clase_predicha = CONTENEDORES_NAMES[best_class_idx]
    
    return clase_predicha, prob_dict