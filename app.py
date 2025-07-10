from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import os

app = Flask(__name__)

# Cargar Imagenes
nombre_a_imagen = {
    "Café Bourbon": "Cafe_Bourbon.png",
    "Café Geisha": "Cafe_Geisha.png",
    "Café de Proceso Natural": "Cafe_Natural.png",
    "Café Caturra": "Cafe_Caturra.png",
    "Café Catuai": "Cafe_Catuai.png",
    "Café Heirloom de Etiopia": "Cafe_Heirloom.png",
    "Crema de Café": "Crema_Cafe.jpg",
    "Jabón Exfoliante de Café": "Jabon_Cafe.jpg",
}

# Cargar productos
productos = [
    {"producto": "Café Bourbon", "categoria": "especialidades", "perfil_sabor": ["dulce", "frutal", "caramelo"]},
    {"producto": "Café Geisha", "categoria": "especialidades", "perfil_sabor": ["floral", "frutal", "limón"]},
    {"producto": "Café de Proceso Natural", "categoria": "especialidades", "perfil_sabor": ["frutal", "vino", "ácido"]},
    {"producto": "Café Caturra", "categoria": "especialidades", "perfil_sabor": ["chocolate", "nuez", "suave"]},
    {"producto": "Café Catuai", "categoria": "especialidades", "perfil_sabor": ["miel", "cítrico", "suave"]},
    {"producto": "Café Heirloom de Etiopia", "categoria": "especialidades", "perfil_sabor": ["floral", "limón", "arándano"]},
    {"producto": "Crema de Café", "categoria": "derivado", "perfil_sabor": ["dulce", "cremoso", "café"]},
    {"producto": "Jabón Exfoliante de Café", "categoria": "derivado", "perfil_sabor": ["intenso", "café", "terroso"]},
]

df_productos = pd.DataFrame(productos)
df_productos["perfil_sabor_str"] = df_productos["perfil_sabor"].apply(lambda x: " ".join(x))

# Machine Learning
vectorizer = TfidfVectorizer()
vectores_perfil = vectorizer.fit_transform(df_productos["perfil_sabor_str"])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_productos["cluster"] = kmeans.fit_predict(vectores_perfil)

# Cargar Usuarios
df_usuarios = pd.read_csv('db_usuarios_Finca_expandidos.csv', encoding='utf-8')
usuarios_csv = df_usuarios['usuario'].str.lower().unique().tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recomendaciones = []
    historial = []
    perfil_usuario = []
    nombre = ""
    perfil = ""

    if request.method == 'POST':
        nombre = request.form['nombre'].strip().lower()
        # perfil = request.form['perfil'].strip().lower()
        perfil = request.form.getlist('perfil')

        # Verificamos si es usuario nuevo
        es_nuevo = nombre not in df_usuarios['usuario'].str.lower().unique()

        if es_nuevo:

            if not perfil or len(perfil) > 3:
                mensaje_error = "Selecciona entre 1 y 3 sabores si eres nuevo."
            else:
                perfil_usuario = perfil
                recomendaciones = recomendar_nuevo_por_cluster(perfil_usuario)
                
        else:
            historial = df_usuarios[df_usuarios['usuario'].str.lower() == nombre]["producto"].tolist()
            recomendaciones = recomendar_por_historial(nombre)

    return render_template('index.html',
                            recomendaciones=recomendaciones,
                            historial=historial,
                            perfil_usuario=perfil_usuario,
                            nombre=nombre,
                            perfil=perfil,
                            mensaje_error=mensaje_error if 'mensaje_error' in locals() else None,
                            usuarios_csv=usuarios_csv,
                            nombre_a_imagen=nombre_a_imagen)


# Recomendar por clustering
def recomendar_nuevo_por_cluster(perfil_usuario):
    perfil_str = " ".join(perfil_usuario)
    vector_usuario = vectorizer.transform([perfil_str])
    cluster_usuario = kmeans.predict(vector_usuario)[0]

    def contar_coincidencias(p_sabor):
        return len(set(p_sabor) & set(perfil_usuario))

    # 🔄 Comparar con TODOS los productos (no solo los del clúster)
    candidatos = df_productos.copy()
    candidatos["coincidencias"] = candidatos["perfil_sabor"].apply(contar_coincidencias)
    recomendados = candidatos[candidatos["coincidencias"] > 0]
    recomendados = recomendados.sort_values(by="coincidencias", ascending=False)

    return recomendados["producto"].tolist()[:5]



# Usuario existente con historial + fallback
def recomendar_por_historial(usuario):
    historial = df_usuarios[df_usuarios['usuario'].str.lower() == usuario.lower()]
    if historial.empty:
        return []

    productos_usuario = historial['producto'].tolist()
    perfiles_historial = df_productos[df_productos['producto'].isin(productos_usuario)]["perfil_sabor_str"]

    if perfiles_historial.empty:
        return []

    vectores_usuario = vectorizer.transform(perfiles_historial)
    vector_promedio = np.asarray(vectores_usuario.mean(axis=0)).reshape(1, -1)

    cluster_usuario = kmeans.predict(vector_promedio)[0]
    productos_cluster = df_productos[df_productos["cluster"] == cluster_usuario]

    recomendaciones = []
    for producto in productos_cluster["producto"]:
        if producto not in productos_usuario and producto not in recomendaciones:
            recomendaciones.append(producto)
        if len(recomendaciones) >= 5:
            break

    # Fallback: si hay menos de 3 recomendaciones, buscar por perfil de sabor
    if len(recomendaciones) < 3:
        perfil_usuario = df_productos[df_productos["producto"].isin(productos_usuario)]["perfil_sabor"]
        flat_perfil = [p for sublist in perfil_usuario for p in sublist]
        top_tags = pd.Series(flat_perfil).value_counts().head(3).index.tolist()

        def contar_coincidencias(p_sabor):
            return len(set(p_sabor) & set(top_tags))

        candidatos = df_productos[~df_productos["producto"].isin(productos_usuario)].copy()
        candidatos["coincidencias"] = candidatos["perfil_sabor"].apply(contar_coincidencias)
        candidatos = candidatos[candidatos["coincidencias"] > 0].sort_values(by="coincidencias", ascending=False)

        for producto in candidatos["producto"]:
            if producto not in recomendaciones:
                recomendaciones.append(producto)
            if len(recomendaciones) >= 5:
                break

    return recomendaciones

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render usa la variable de entorno PORT
    app.run(host='0.0.0.0', port=port)
