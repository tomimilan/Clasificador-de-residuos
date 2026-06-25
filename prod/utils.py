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

CONTENEDORES_NAMES = {
    0: "VERDE (Secos reciclables)",
    1: "NEGRO (Húmedos rechazo, comida, basura)",
    2: "AMARILLO (Electrónicos rechazo parcial)",
    3: "MARRÓN (Textil especial)"
}

# Base de datos educativa integrada (Abstracción de Datos)
DETALLES_CONTENEDORES = {
    "VERDE": {
        "descripcion": "Materiales secos y limpios aptos para reciclaje mecánico o valorización material.",
        "ley": "Ley 9.659 (GIRSU Mendoza): Reintroducción de recursos al sistema económico mediante cooperativas de recuperadores urbanos.",
        "bolsa": "Bolsa transparente o verde. El material debe ingresarse completamente limpio y seco.",
        "ejemplos": "Cajas de cartón, botellas plásticas de gaseosa, envases de vidrio, latas de conserva, papel blanco."
    },
    "NEGRO": {
        "descripcion": "Residuos húmedos, restos biológicos o materiales compuestos no recuperables.",
        "ley": "Ley Provincial 5.961: Disposición final controlada para mitigar la emanación de lixiviados y gases de efecto invernadero.",
        "bolsa": "Bolsa negra o gris estándar de residuos domiciliarios.",
        "ejemplos": "Restos de comida, cáscaras de frutas, yerba mate, pañales desechables, servilletas usadas, tazas rotas."
    },
    "AMARILLO": {
        "descripcion": "Residuos de Aparatos Eléctricos y Electrónicos (RAEE) y componentes con presencia de metales pesados.",
        "ley": "Reglamento de Residuos Peligrosos: Prohibición de descarte en la vía pública por riesgo de contaminación de acuíferos.",
        "bolsa": "Caja de cartón o bolsa hermética aislada. Traslado obligatorio por parte del ciudadano al centro de acopio.",
        "ejemplos": "Pilas AA/AAA, baterías de litio, cables, cargadores dañados, periféricos informáticos, lámparas en desuso."
    },
    "MARRÓN": {
        "descripcion": "Fracción de descartes textiles, calzado e indumentaria en desuso.",
        "ley": "Estrategia de Economía Circular: Mitigación del impacto del modelo 'Fast Fashion' y fomento de la donación o supra-reciclaje.",
        "bolsa": "Bolsa plástica cerrada de manera hermética para evitar el ingreso de humedad ambiente.",
        "ejemplos": "Ropa en desuso, zapatos gastados, sábanas viejas, retazos textiles de costura, mantas."
    }
}

# MOTOR DE GEORREFERENCIACIÓN INTELIGENTE (Cruza de variables Municipio x Categoría)
PUNTOS_GEOLOCALIZADOS = {
    "Capital": {
        "VERDE": [
            {"name": "Punto Verde Plaza Independencia", "lat": -32.8895, "lon": -68.8448, "tipo": "Papel, Cartón, Plásticos y Vidrio"},
            {"name": "Contenedor Reciclaje Parque Central", "lat": -32.8732, "lon": -68.8420, "tipo": "Secos Limpios Solamente"}
        ],
        "AMARILLO": [
            {"name": "Centro RAEE Municipalidad de Capital", "lat": -32.8970, "lon": -68.8425, "tipo": "Pilas, Baterías y Pequeños Electrodomésticos"}
        ],
        "MARRÓN": [
            {"name": "Punto Textil Paseo Pellegrini", "lat": -32.8930, "lon": -68.8510, "tipo": "Ropa y Calzado para Donación/Acopio"}
        ]
    },
    "Godoy Cruz": {
        "VERDE": [
            {"name": "Punto Limpio Central Parque Benegas", "lat": -32.9345, "lon": -68.8488, "tipo": "Cartón, PET, Aluminio y Envases"},
            {"name": "Estación Verde Plaza Godoy Cruz", "lat": -32.9252, "lon": -68.8415, "tipo": "Clasificación Completa de Secos"}
        ],
        "AMARILLO": [
            {"name": "Contenedor RAEE Hiper Libertad", "lat": -32.9230, "lon": -68.8610, "tipo": "Pilas Usadas y Componentes Electrónicos"}
        ],
        "MARRÓN": [
            {"name": "Planta Industrial Cruz Textil (Acopio)", "lat": -32.9150, "lon": -68.8450, "tipo": "Recuperación e Hilado Textil"}
        ]
    },
    "Guaymallén": {
        "VERDE": [
            {"name": "Punto Limpio Cultural Espacio Le Parc", "lat": -32.8960, "lon": -68.8105, "tipo": "Materiales Reciclables Limpios"},
            {"name": "Estación de Reciclaje Plaza Belgrano", "lat": -32.9110, "lon": -68.7950, "tipo": "Plásticos, Vidrios y Metales"}
        ],
        "AMARILLO": [
            {"name": "Depósito RAEE Municipalidad de Guaymallén", "lat": -32.9180, "lon": -68.8120, "tipo": "Residuos Tecnológicos y Pilas"}
        ],
        "MARRÓN": [
            {"name": "Punto Verde Textil Dorrego", "lat": -32.9100, "lon": -68.8250, "tipo": "Recopilación de Telas e Indumentaria"}
        ]
    },
    "Luján de Cuyo": {
        "VERDE": [
            {"name": "Punto Verde Plaza Departamental Luján", "lat": -33.0355, "lon": -68.8795, "tipo": "Cartón, Vidrio, Plásticos y Latas"},
            {"name": "Estación de Reciclaje Chacras de Coria", "lat": -33.0040, "lon": -68.8720, "tipo": "Materiales Secos Seleccionados"}
        ],
        "AMARILLO": [
            {"name": "Centro RAEE Delegación Carrodilla", "lat": -32.9890, "lon": -68.8640, "tipo": "Pilas de Reloj, Baterías y Celulares Viejos"}
        ],
        "MARRÓN": [
            {"name": "Centro Textil de Acopio Ugarteche", "lat": -33.1600, "lon": -68.8900, "tipo": "Ropa Usada y Retazos de Costura"}
        ]
    },
    "Maipú": {
        "VERDE": [
            {"name": "Estación Verde Parque Metropolitano", "lat": -32.9780, "lon": -68.7890, "tipo": "Plásticos, Cartones e Insumos Secos"},
            {"name": "Punto Limpio Plaza Departamental Maipú", "lat": -32.9723, "lon": -68.7906, "tipo": "Vidrio y Envases Metálicos"}
        ],
        "AMARILLO": [
            {"name": "Eco-Punto Polideportivo Juan Ribosqui", "lat": -32.9740, "lon": -68.7950, "tipo": "Pilas y Componentes de Computación"}
        ],
        "MARRÓN": [
            {"name": "Centro de Acopio Coquimbito", "lat": -32.9650, "lon": -68.7550, "tipo": "Donaciones e Indumentaria en Desuso"}
        ]
    },
    "Las Heras": {
        "VERDE": [
            {"name": "Punto Limpio Plaza Marcos Burgos", "lat": -32.8482, "lon": -68.8332, "tipo": "Papel, Cartón y Botellas PET"},
            {"name": "Estación de Separación Parque de la Familia", "lat": -32.8590, "lon": -68.8450, "tipo": "Residuos Secos Autorizados"}
        ],
        "AMARILLO": [
            {"name": "Centro RAEE Municipalidad de Las Heras", "lat": -32.8490, "lon": -68.8350, "tipo": "Dispositivos Electrónicos y Baterías de Plomo"}
        ],
        "MARRÓN": [
            {"name": "Punto Textil Histórico El Plumerillo", "lat": -32.8350, "lon": -68.8120, "tipo": "Campañas de Donación e Indumentaria Textil"}
        ]
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