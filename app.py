import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
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
    # Descargar stopwords
    nltk.download('stopwords', quiet=True)
    stop_words_es = set(stopwords.words('spanish'))

    # Cargar BETO
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model_beto = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

    # Cargar tu SVM entrenado
    svm = joblib.load('clasificador_svm_beto.pkl')

    return tokenizer, model_beto, svm, stop_words_es

tokenizer_beto, model_beto, svm_clf, stop_words_es = cargar_modelos()

# 3. Funciones de procesamiento (idénticas a las de tu entrenamiento)
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\@\w+|\#', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = texto.split()
    tokens_limpios = [word for word in tokens if word not in stop_words_es]
    return " ".join(tokens_limpios)

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

            with st.expander("Ver texto preprocesado (Tokenización)"):
                st.code(texto_procesado)
