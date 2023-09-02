import os
import sqlite3
import tensorflow as tf
import json

from PIL import Image
from Funciones import *
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from flask import redirect, url_for, flash, send_from_directory
import requests
import json

# Crear una instancia de la aplicación Flask
app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')

# Clave secreta para la aplicación (cambia "clave_secreta" por tu clave real)
app.secret_key = "clave_secreta"

# Carpeta donde se cargarán los archivos
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cargar el modelo de red neuronal desde un archivo previamente entrenado
model = tf.keras.models.load_model("modelCaso5.h5")

# Configurar la carpeta de carga de archivos en la aplicación
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Función para verificar si la extensión de un archivo es permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Definir la ruta para la URL raíz y especificar los métodos que puede manejar
@app.route('/', methods=['POST'])
def upload_file():
    # Verificar si no se proporcionó ningún archivo en la solicitud
    if 'file' not in request.files:
        flash('No se proporcionó archivo')
        return redirect(request.url)

    # Obtener el archivo enviado en la solicitud
    file = request.files['file']

    # Verificar si no se seleccionó ningún archivo
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)

    # Verificar si el archivo tiene una extensión permitida
    if file and allowed_file(file.filename):
        # Generar un nombre de archivo seguro
        filename = secure_filename(file.filename)

        # Construir la ruta completa donde se guardará el archivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Guardar el archivo en la ruta especificada
        file.save(filepath)
        print(filepath)

        try:
            # Obtener edad y tono de piel utilizando una función (age_skin_tone) definida en otro lugar
            _ed, tono = age_skin_tone(filepath)

            edad = classify_age_range1(int(request.args.get("edad")))

            # Estimar la forma de la cara utilizando una función (predict_face_shape) y el modelo cargado
            forma, prob = predict_face_shape(filepath, model)

            # Obtener recomendaciones basadas en la edad, tono de piel y forma de la cara
            recomendacion_edad, recomendacion_tono, recomendacion_cara, link, linkEdad, linkTono, linkForma = obtener_recomendaciones_db(edad, tono, forma, prob)
            
            #obtener imagenes de stable difusion
            images =  generate_images("college makeup, 19 years old, simple", f"http://3.217.16.126:5000/{filepath}",4)

        except ValueError as e:
            # Manejar casos de error al procesar la imagen
            if "Face could not be detected" in str(e):
                return json.dumps(), 500
            else:
                return json.dumps(), 500
            

    #devolver resultados
    return json.dumps({"edad":edad, 
                       "tono":tono, 
                       "forma":forma, 
                       "recomendacion_edad":recomendacion_edad, 
                       "recomendacion_tono": recomendacion_tono, 
                       "recomendacion_cara": recomendacion_cara, 
                       "link": link, 
                       "linkEdad": linkEdad, 
                       "linkTono": linkTono, 
                       "linkForma": linkForma,
                       "imagenes": json.loads(images.text)})

def generate_images(prompt, init_image, samples):
    url = "https://stablediffusionapi.com/api/v3/img2img"

    payload = json.dumps({
    "key": "0izA2Ocf5Z7vOOZPYnSFWWX17xpvE8GC8d3EI9xT2dVjNoS3hosU8rOF59cX",
    "prompt": prompt,
    "negative_prompt": None,
    "init_image": init_image,
    "width": "512",
    "height": "512",
    "samples": samples,
    "num_inference_steps": "30",
    "safety_checker": "no",
    "enhance_prompt": "yes",
    "guidance_scale": 7.5,
    "strength": 0.7,
    "seed": None,
    "webhook": None,
    "track_id": None
    })

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response

# Ejecutar la aplicación si el script es ejecutado directamente
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")