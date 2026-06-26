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
        "descripcion": "Materiales aptos para reciclaje o valorización material.",
        "ley": "Ley 9.659 (GIRSU Mendoza): Reintroducción de recursos al sistema económico mediante cooperativas de recuperadores urbanos.",
        "bolsa": "Bolsa transparente o verde. El material debe ingresarse completamente limpio y seco.",
        "ejemplos": "Cajas de cartón, botellas plásticas de gaseosa, envases de vidrio, latas de conserva, papel blanco."
    },
    "NEGRO": {
        "descripcion": "Residuos húmedos, restos biológicos/orgánicos o materiales compuestos no recuperables.",
        "ley": "Ley Provincial 5.961: Disposición final controlada para mitigar la emanación de lixiviados y gases de efecto invernadero.",
        "bolsa": "Bolsa negra o gris estándar de residuos domiciliarios.",
        "ejemplos": "Restos de comida, cáscaras de frutas, yerba mate, basura en general."
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

# BASE DE DATOS GEOGRÁFICA UNIFICADA (Estructura plana por Departamento, Coordenadas exactas provistas)
PUNTOS_GEOLOCALIZADOS = {
    "Capital": [
        {"name": "Plaza Independencia", "lat": -32.88940391700496, "lon": -68.8449077057134},
        {"name": "Parque Central", "lat": -32.87444786375254, "lon": -68.84242724543873},
        {"name": "Nave Cultural", "lat": -32.8781641182645, "lon": -68.83939499052592}
    ],
    "Godoy Cruz": [
        {"name": "Parque Benegas", "lat": -32.95234372585135, "lon": -68.85310509236986},
        {"name": "Plaza Godoy Cruz", "lat": -32.925101961991736, "lon": -68.8443104445004},
        {"name": "Hiper Libertad", "lat": -32.92880615894723, "lon": -68.85649621935909}
    ],
    "Guaymallén": [
        {"name": "Espacio Cultural Julio LeParc", "lat": -32.88750642688741, "lon": -68.81468678214},
        {"name": "Municipalidad de Guaymallén", "lat": -32.89969978668092, "lon": -68.78799624771794}
    ],
    "Luján de Cuyo": [
        {"name": "Centro Verde", "lat": -33.04947299728296, "lon": -68.87275918969102},
        {"name": "Plaza Departamental de Luján de Cuyo", "lat": -33.03890620263929, "lon": -68.87979029236624}
    ],
    "Maipú": [
        {"name": "Parque Metropolitano", "lat": -32.97421625974557, "lon": -68.80195966168651},
        {"name": "Polideportivo Juan Ribosqui", "lat": -32.97653296673632, "lon": -68.79442797702758},
        {"name": "ChangoMás", "lat": -32.978349370886235, "lon": -68.79748874449801}
    ],
    "Las Heras": [
        {"name": "Polimeni", "lat": -32.8515251882089, "lon": -68.83946151821625},
        {"name": "Parque Industrial", "lat": -32.81784874896546, "lon": -68.80444537334037}
    ]
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