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

# Diccionario de Metadatos Educativos extraído de la vista para mantener el código limpio
DETALLES_CONTENEDORES = {
    "VERDE": {
        "descripcion": "Materiales secos que pueden reingresar al circuito productivo gracias a los recuperadores urbanos.",
        "ley": "Ley 9659 (GIRSU): Su separación es el pilar de la Economía Circular en la provincia, evitando que recursos útiles se entierren.",
        "bolsa": "Bolsa transparente o verde. Siempre limpios y secos.",
        "ejemplos": "Cartón, botellas plásticas, frascos de vidrio, latas de aluminio, papel de diario.",
        "puntos": ["Punto Limpio Capital (Plaza Independencia)", "Punto Limpio Godoy Cruz (Parque Benegas)"],
        "coords": [[-32.8894, -68.8451], [-32.9342, -68.8491]]
    },
    "NEGRO": {
        "descripcion": "Residuos orgánicos o húmedos que no pueden ser recuperados y tienen como destino el relleno sanitario.",
        "ley": "Ley 5961: Su correcta disposición evita la proliferación de vectores y reduce las emisiones de gas metano a cielo abierto.",
        "bolsa": "Bolsa negra o gris común de residuos.",
        "ejemplos": "Restos de comida, cáscaras de fruta, yerba, pañales, servilletas usadas, cerámicos rotos.",
        "puntos": ["Recolección domiciliaria nocturna según cronograma municipal"],
        "coords": [[-32.8900, -68.8300]]  
    },
    "AMARILLO": {
        "descripcion": "Residuos de Aparatos Eléctricos y Electrónicos (RAEE) y componentes con metales pesados.",
        "ley": "Normativa de Residuos Peligrosos: Contienen plomo y mercurio que contaminan de forma irreversible las napas de agua si van a un basural común.",
        "bolsa": "Disposición en caja o bolsa cerrada especial, llevar directamente al centro de recepción. ¡No sacar a la vereda!",
        "ejemplos": "Pilas, baterías de celular, cargadores rotos, mouses, teclados, tubos fluorescentes.",
        "puntos": ["Punto Limpio RAEE Guaymallén", "Punto Especial Maipú Municipio"],
        "coords": [[-32.8952, -68.8101], [-32.9721, -68.7905]]
    },
    "MARRÓN": {
        "descripcion": "Fracción textil en desuso que requiere un procesamiento de acopio o donación.",
        "ley": "Enfoque Sustentable: Mitiga el impacto de la industria textil, una de las actividades con mayor huella hídrica a nivel global.",
        "bolsa": "Bolsa cerrada para proteger los materiales de la humedad.",
        "ejemplos": "Ropa vieja, calzado gastado, sábanas, retazos de tela, alfombras.",
        "puntos": ["Centros de acopio textil municipales", "Contenedores especiales de cooperativas aliadas"],
        "coords": [[-32.9100, -68.8400]]
    }
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