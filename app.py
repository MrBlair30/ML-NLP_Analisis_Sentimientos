import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Análisis de Sentimiento - Educación", page_icon="🎓")
st.title("Sistema de Clasificación de Sentimiento")
st.markdown("### Análisis de opiniones sobre la Senescyt y el sistema de admisión")

# 2. Caché para cargar los modelos pesados una sola vez y que la app sea rápida
@st.cache_resource
def cargar_modelos():
    # Cargar BETO (Ya no descargamos NLTK ni stopwords)
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model_beto = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

    # Cargar tu SVM entrenado
    svm = joblib.load('clasificador_svm_beto.pkl')

    return tokenizer, model_beto, svm

tokenizer_beto, model_beto, svm_clf = cargar_modelos()

# 3. Funciones de procesamiento (Optimizadas para BETO: preservando el contexto, puntuación y números)

def limpiar_texto(texto):
    # 1. Convertir a minúsculas
    texto = texto.lower()
    
    # 2. Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    # 3. Eliminar menciones y hashtags
    texto = re.sub(r'\@\w+|\#', '', texto)
    
    # 4. Reducir letras repetidas
    texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
    
    # ¡LA SOLUCIÓN MAGISTRAL!: 5. Reemplazar cualquier número por la palabra "puntaje"
    texto = re.sub(r'\d+', ' puntaje ', texto)
    
    # 6. Eliminar símbolos extraños, PERO conservar letras y puntuación básica
    texto = re.sub(r'[^\w\s\.,;:¡!¿\?_]', '', texto)
    
    # 7. Reducir múltiples signos de puntuación seguidos
    texto = re.sub(r'([.,;:\?!¿¡])\1+', r'\1', texto)
    
    # 8. Eliminar espacios dobles
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def aplicar_embeddings(texto_limpio):
    inputs = tokenizer_beto([texto_limpio], padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model_beto(**inputs)
    # Retornamos el token [CLS]
    return outputs.last_hidden_state[:, 0, :].numpy()

# 4. Interfaz de Usuario
texto_usuario = st.text_area("Ingresa un comentario u opinión aquí:", height=150)

if st.button("Analizar Sentimiento", type="primary"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, ingresa un texto para analizar.")
    else:
        with st.spinner('Procesando texto con BETO...'):
            # Flujo de predicción
            texto_procesado = limpiar_texto(texto_usuario)
            embedding = aplicar_embeddings(texto_procesado)
            prediccion = svm_clf.predict(embedding)[0]

            # Mostrar resultados
            st.markdown("---")
            st.markdown("### Resultado de la Clasificación:")

            if prediccion == 0:
                st.error("Sentimiento: **Negativo**")
            elif prediccion == 1:
                st.info("Sentimiento: **Neutral**")
            elif prediccion == 2:
                st.success("Sentimiento: **Positivo**")

            with st.expander("Ver texto preprocesado (Limpieza)"):
                st.code(texto_procesado)