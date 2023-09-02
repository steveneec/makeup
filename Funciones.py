import os
import cv2
import sqlite3
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf
from deepface import DeepFace
from sklearn.cluster import KMeans

### Función para extraer el color RGB dominante de una imagen

def color_extractor(image, k=4):
    
    # Redimensionar la imagen a un tamaño estándar
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Cambiar la forma del array para que sea una lista de píxeles
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # Aplicar el algoritmo de clustering K-means
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    label_counts = np.bincount(labels)
    
    # Identificar el cluster más popular y obtener su color central
    dominant_color = clt.cluster_centers_[np.argmax(label_counts)]
    
    return dominant_color.astype(int)  # Devolver el color dominante


# Funcion que recibe como argumento el path de la imagen que queremos analizar
def agepredictor(image_name):
    agep = DeepFace.analyze(img_path = image_name, 
        actions = 'age')
    jsonage = agep[0]
    age = jsonage['age']
    x = jsonage['region']['x']
    y = jsonage['region']['y']
    w = jsonage['region']['w']
    h = jsonage['region']['h']
    return age,(x,y,w,h)   


# Función que devuelve la edad y el color RGB dominante del rostro
def extract_dominant_color(image_name):
    # Obtiene la edad y coordenadas del rostro usando la función 'agepredictor'
    age, coords = agepredictor(image_name)
    # Lee la imagen y la convierte en un array
    image_array = cv2.imread(image_name)
    # Convierte la imagen de BGR (formato predeterminado de OpenCV) a RGB
    rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # Extrae las coordenadas del rostro de la imagen
    x, y, w, h = coords
    # Obtiene el área de la imagen que corresponde al rostro detectado
    face_region = rgb_array[y:y+h, x:x+w]
    # Extrae el color RGB dominante de esa región del rostro usando la función 'color_extractor'
    RGB = color_extractor(face_region)
    # Devuelve la edad y el color RGB dominante
   
    return age , RGB


# Función que clasifica el tono de piel basado en un vector RGB
def classify_skin_tone(rgb_vector):
    # Diccionario con tonos de piel y sus valores RGB representativos
    COLOR_RANGES = {
        "Muy claro / pálido": [229, 214, 197],
        "Clara": [230, 198, 183],
        "Trigeña": [218, 171, 145],
        "Medio / moreno": [190, 142 , 119],
        "Oscuro": [177, 120, 84]
    }
    
    # Inicializamos la menor distancia con un valor infinitamente grande
    min_distance = float('inf')
    closest_skin_tone = None
    
    # Iteramos sobre los tonos de piel predefinidos
    for skin_tone, central_rgb in COLOR_RANGES.items():
        # Calculamos la distancia euclidiana entre el vector RGB y el valor RGB central del tono de piel actual
        distance = np.linalg.norm(np.array(rgb_vector) - np.array(central_rgb))
        # Si la distancia es menor que la más pequeña registrada hasta el momento, la actualizamos
        if distance < min_distance:
            min_distance = distance
            closest_skin_tone = skin_tone
    
    # Devolvemos el tono de piel más cercano al vector RGB ingresado
    return closest_skin_tone

### Funcion que classifica la edad en un rango determinado.

def classify_age_range1(age):
    if age < 18:
        return "menores de 18"
    elif 18 <= age <= 25:
        return "18-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 45:
        return "36-45"
    else:
        return "46 en adelante"
    
### Función que devuelve el rango de edad y el tono de piel de la persona en la imagen
def age_skin_tone(image_name):
    # Extrae la edad y el color RGB dominante del rostro en la imagen
    age, rgb_input = extract_dominant_color(image_name)
    
    # Clasifica el vector RGB para determinar el tono de piel
    skin_tone = classify_skin_tone(rgb_input)
    
    # Clasifica la edad en un rango determinado
    age_range = classify_age_range(age)
    
    # Devuelve el rango de edad y el tono de piel
    
    return age_range, skin_tone
   

### Funcion que predice la forma del rostro y la probabilidad
def predict_face_shape(file_path, model, img_size1=100, img_size2=120):
    
    # Procesa la imagen para prepararla para la predicción
    img = Image.open(file_path)
    img = img.resize((img_size1, img_size2))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normaliza la imagen al rango [0,1]
    
    # Realiza predicciones sobre la forma del rostro
    predictions = model.predict(img_array)
    
    # Encuentra el índice de la categoría con la probabilidad más alta
    i_max = tf.argmax(predictions[0])
    prob = predictions[0][i_max]
    category_index = tf.argmax(predictions[0]).numpy()
    
    # Define las categorías de formas de rostro
    category_names = ['CORAZON', 'ALARGADA', 'OVALADA', 'REDONDA', 'CUADRADA']

    # Devuelve el nombre de la forma del rostro y su probabilidad asociada
    return category_names[category_index], prob


def obtener_recomendaciones_db(rango_edad, tono_piel, forma_cara, prob, filename="recommendationsT.db"):
    # Conectarse a la db
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Buscar y obtener las recomendaciones de la base de datos
    cursor.execute('SELECT recomendacion, link FROM edad_recomendaciones WHERE rango=?', (rango_edad,))
    recomendacion_edad_data = cursor.fetchone()
    recomendacion_edad = f"Basado en tu edad ({rango_edad}):\n{recomendacion_edad_data[0]}\n\n"
    link = recomendacion_edad_data[1]

    cursor.execute('SELECT recomendacion, linkTono FROM tono_recomendaciones WHERE tono=?', (tono_piel,))
    recomendacion_tono_data = cursor.fetchone()
    recomendacion_tono = f"Basado en tu tono de piel ({tono_piel}):\n{recomendacion_tono_data[0]}\n\n"
    #linkTono = recomendacion_tono_data[1]
    
    cursor.execute('SELECT recomendacion FROM forma_cara_recomendaciones WHERE forma=?', (forma_cara,))
    recomendacion_cara_data = cursor.fetchone()
    recomendacion_cara = f"Basado en tu forma de cara ({forma_cara}):\n{recomendacion_cara_data[0]}"
    #link_forma = recomendacion_cara_data[1]
    
    #Consultas a la tabla matriz
    
    consulta = 'SELECT linkedad, linktono, linkforma  FROM matriz WHERE rango=? AND tono=? AND forma=?'
    cursor.execute(consulta, (rango_edad, tono_piel, forma_cara))
    recomendacion_informacion_data = cursor.fetchone()
    linkEdad = recomendacion_informacion_data[0]
    linkTono = recomendacion_informacion_data[1]
    linkForma = recomendacion_informacion_data[2]
    
    #consulta = 'SELECT linkedad FROM matriz WHERE rango=? AND tono=? AND forma=?'
    #cursor.execute(consulta, (rango_edad, tono_piel, forma_cara))
    #recomendacion_eddad_data = cursor.fetchone()
    #linkEdad = recomendacion_eddad_data[0]
    
    #consulta1 = 'SELECT linktono FROM matriz WHERE rango=? AND tono=? AND forma=?'
    #cursor.execute(consulta1, (rango_edad, tono_piel, forma_cara))
    #recomendacion_tono_data = cursor.fetchone()
    #linkTono = recomendacion_tono_data[0]
    
    #consulta2 = 'SELECT linkforma FROM matriz WHERE rango=? AND tono=? AND forma=?'
    #cursor.execute(consulta2, (rango_edad, tono_piel, forma_cara))
    #recomendacion_forma_data = cursor.fetchone()
    #linkForma = recomendacion_forma_data[0]
    
    # Cerrando la conexion
    conn.close()

    return recomendacion_edad, recomendacion_tono, recomendacion_cara, link, linkEdad, linkTono , linkForma
   
def classify_age_range(age):
    if age < 18:
        return "menores de 18"
    elif 18 <= age <= 25:
        return "menores de 18"
    elif 26 <= age <= 35:
        return "18-25"
    elif 36 <= age <= 45:

        return "46 en adelante"
    else:
        return "46 en adelante"