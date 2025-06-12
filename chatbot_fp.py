# -------------------------------------------------------------------
#pip install streamlit pandas faiss-cpu sentence-transformers requests
#pip install openai langchain PyPDF2 langdetect langchain-community
#pip install spacy scikit-learn openpyxl pymupdf gtts
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Importar las librerías necesarias
# -------------------------------------------------------------------
import fitz  # PyMuPDF
import pandas as pd
import os
import requests

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from PIL import Image

# Para la conversión de texto a voz
# pip install gtts
from gtts import gTTS
import io  # Necesario para manejar el buffer de audio
import base64 # Necesario para incrustar audio directamente en HTML
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Definición de rutas y ficheros de datos ---
url = "https://raw.githubusercontent.com/antoniojasweb/chatbot/main/pdf/"
FilePDF = "25_26_OFERTA_por_Familias.pdf"
FileExcel = "oferta_formativa_completa.xlsx"
FileLogo = "logo.jpg"

# --- Modelo de embeddings ---
# Otro modelo de Sentence Transformers, para Ingés: 'all-MiniLM-L6-v2'
ModeloEmbeddings = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Configuración de la API de Gemini (Desde Colab, puedes dejar apiKey vacío para que Canvas lo gestione) ---
# Si quieres usar modelos diferentes a gemini-2.0-flash o imagen-3.0-generate-002, proporciona una clave API aquí. De lo contrario, déjalo como está.
API_KEY = "AIzaSyCf_fP-atrKzMJGwUgMCdHReTQtPoXKW8o"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Definición de funciones ---
# -------------------------------------------------------------------
def color_to_rgb(color_int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r, g, b)

# Descargar el fichero del logo del chatbot
def descargar_logo(fichero_logo):
    FileURL = url + fichero_logo
    if not os.path.exists(fichero_logo):
        response = requests.get(FileURL)
        # Verificamos que la solicitud fue exitosa
        if response.status_code == 200:
            # Abrimos el archivo en modo de lectura binaria
            with open(FileLogo, 'wb') as file:
                file.write(response.content)
            #print("Logo descargado y guardado localmente.")
        #else:
            #print(f"Error al descargar el logo: {response.status_code}")

# Descargar fichero PDF desde URL
def descargar_pdf(fichero_pdf):
    FileURL = url + fichero_pdf
    if not os.path.exists(fichero_pdf):
        response = requests.get(FileURL)
        # Verificamos que la solicitud fue exitosa
        if response.status_code == 200:
            # Abrimos el archivo PDF en modo de lectura binaria
            with open(FilePDF, 'wb') as file:
                file.write(response.content)
            #print("Archivo descargado y guardado localmente.")
        #else:
            #print(f"Error al descargar el archivo: {response.status_code}")

# Extraer información del PDF y guardarla en un DataFrame
def extraer_informacion_pdf(fichero_pdf):
    # Comprobar si el archivo Excel ya existe, y lo eliminamos, para generar uno nuevo
    if os.path.exists(FileExcel):
        os.remove(FileExcel)

    # Abrir el PDF
    print("Fichero PDF procesado: " + fichero_pdf)
    doc = fitz.open(fichero_pdf)

    # Variables de contexto
    data = []
    familia_actual = ""
    grado_actual = ""
    codigo_ciclo = ""
    nombre_ciclo = ""
    provincia_actual = ""

    # Recorrer cada página y extraer información
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for span in l["spans"]:
                    text = span["text"].strip()
                    rgb = color_to_rgb(span["color"])
                    font = span["font"]
                    is_bold = "Bold" in font or "bold" in font.lower()

                    # Familia profesional (verde y mayúsculas)
                    if text.isupper() and rgb[1] > 120 and rgb[0] < 100 and rgb[2] < 100:
                        familia_actual = text

                    # Grado formativo (naranja y mayúsculas)
                    # elif text.isupper() and rgb[0] > 150 and rgb[1] > 90 and rgb[2] < 50:
                    #     grado_actual = text
                    elif rgb[0] > 150 and 90 < rgb[1] < 190 and rgb[2] > 80:
                        grado_actual = text  # Guardamos el texto literal como grado actual

                    # Ciclo formativo (azul, con código entre paréntesis)
                    elif text.startswith("(") and ")" in text and rgb[2] > 100:
                        if ")" in text:
                            codigo_ciclo = text.split(")")[0].strip("()")
                            nombre_ciclo = text.split(")", 1)[1].strip()

                    # Provincia (negrita + negro)
                    elif text in ["BADAJOZ", "CÁCERES"] and is_bold:
                        provincia_actual = text

                    # Centro educativo (normal, negro, contiene ' - ' al menos 2 veces)
                    elif text.count(" - ") >= 2 and not is_bold and rgb == (0, 0, 0):
                        try:
                            municipio, instituto, curso_raw = text.split(" - ", 2)

                            # Extras
                            curso = curso_raw
                            turno = "Diurno"
                            bilingue = "No"
                            nuevo = "No"

                            if "Vespertino" in curso:
                                turno = "Vespertino"
                            if "Bilingüe" in curso or "Bilingue" in curso:
                                bilingue = "Sí"
                            if "Nuevo" in curso:
                                nuevo = "Sí"

                            # Limpiar texto del campo curso
                            curso = (curso
                                    .replace("Vespertino", "")
                                    .replace("Diurno", "")
                                    .replace("Nuevo", "")
                                    .replace("Bilingüe: IN", "")
                                    .strip())

                            # Añadir fila
                            data.append({
                                "Familia Profesional": familia_actual,
                                "Grado": grado_actual,
                                "Código Ciclo": codigo_ciclo,
                                "Nombre Ciclo": nombre_ciclo,
                                "Provincia": provincia_actual,
                                "Municipio": municipio.strip(),
                                "Instituto": instituto.strip(),
                                "Curso": curso.strip(),
                                "Turno": turno,
                                "Bilingüe": bilingue,
                                "Nuevo": nuevo
                            })

                        except ValueError:
                            continue  # línea malformada

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Exportar a Excel
    df.to_excel(FileExcel, index=False)
    print("Fichero Excel creado : " + FileExcel)

    # Mostrar primeras filas
    #print(df.head())
    #df.head()

    return df
#--------------------------------------------------------------------

# --- Cargar el modelo de embeddings y crear el índice FAISS ---
@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo de embeddings pre-entrenado.
    Se usa un modelo multilingüe para mejor rendimiento con español.
    """
    #st.write("Cargando modelo de embeddings (esto puede tardar unos segundos)...")
    model = SentenceTransformer(ModeloEmbeddings)
    #st.write("Modelo de embeddings cargado.")
    return model

def create_faiss_index(df: pd.DataFrame, model: SentenceTransformer):
    """
    Crea un índice FAISS a partir de los datos del DataFrame.
    """
    #st.write("Creando embeddings e índice FAISS...")
    # Concatenar las columnas relevantes en una sola cadena de texto para el embedding
    df['combined_text'] = df.apply(
        lambda row: f"Ciclo: {row.get('Nombre Ciclo', '')}. Nivel: {row.get('Grado', '')}. Familia: {row.get('Familia Profesional', '')}. Centro: {row.get('Instituto', '')}. Ciudad: {row.get('Municipio', '')}. Provincia: {row.get('Provincia', '')}. Turno: {row.get('Turno', '')}",
        axis=1
    )

    # Rellenar cualquier NaN con cadena vacía para evitar errores de embedding
    df['combined_text'] = df['combined_text'].fillna('')

    corpus = df['combined_text'].tolist()
    embeddings = model.encode(corpus, show_progress_bar=True)

    # Crear un índice FAISS Flat (Simple)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32')) # FAISS requiere float32

    #st.write("Índice FAISS creado.")
    return index, corpus # Devolvemos también el corpus para poder mapear los resultados

def get_gemini_response(prompt: str):
    """
    Hace una llamada a la API de Gemini para obtener una respuesta.
    """
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lanza una excepción para errores HTTP
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Lo siento, no pude obtener una respuesta de la IA. La estructura de la respuesta fue inesperada."
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de Gemini: {e}. Por favor, verifica tu conexión y clave API.")
        return "Lo siento, hubo un problema al conectar con el servicio de IA."
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        return "Lo siento, ocurrió un error inesperado al procesar tu solicitud."

def ask_rag_model(query: str, index, corpus: list, model: SentenceTransformer, df: pd.DataFrame, top_k: int = 10):
    """
    Realiza la consulta RAG:
    1. Embed de la consulta.
    2. Busca documentos relevantes en el índice FAISS.
    3. Construye un prompt contextualizado para el LLM.
    4. Llama al LLM para generar la respuesta.
    """
    query_embedding = model.encode([query]).astype('float32')

    # Realiza la búsqueda en FAISS
    D, I = index.search(query_embedding, top_k) # D es la distancia, I son los índices

    # Recupera los documentos relevantes
    retrieved_docs_text = [corpus[idx] for idx in I[0]]
    # También recuperamos las filas completas del DataFrame para más detalles si son necesarios
    retrieved_docs_df = df.iloc[I[0]]

    context = "\n\n".join(retrieved_docs_text)

    # Crear un prompt más detallado para guiar a Gemini
    prompt_template = f"""
    Eres un asistente experto en ciclos formativos en Extremadura.
    Aquí tienes información relevante sobre ciclos formativos en Extremadura:

    ---
    {context}
    ---

    Basándote ÚNICAMENTE en la información proporcionada anteriormente y en tu conocimiento general, responde a la siguiente pregunta de forma concisa y útil. Si la información proporcionada no es suficiente para responder a la pregunta, indícalo.

    Pregunta: {query}

    Respuesta:
    """

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": f"Buscando información relacionada con '{query}'..."})

    # Muestra los documentos recuperados para depuración o información al usuario
    # with st.expander("Ver información recuperada: " + str(top_k) + " opciones más relevantes"):
    #     st.write(retrieved_docs_df[['Nombre Ciclo', 'Municipio', 'Provincia', 'Grado', 'Familia Profesional']])

    #st.write("Los datos encontrados a tu pregunta son:")
    #st.write(context)

    return get_gemini_response(prompt_template)

# --- Función para convertir texto a audio y obtenerlo en base64 ---
def text_to_audio_base64(text, lang='es'):
    """
    Convierte texto a audio usando gTTS y devuelve el audio codificado en base64.
    """
    try:
        text = text.replace('*', ' ')
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Codificar el buffer de audio en base64
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        return audio_base64
    except Exception as e:
        st.error(f"Error al generar audio: {e}")
        return None

# -------------------------------------------------------------------
# --- Comprobación de existencia de archivos y carga de datos ---
# Comprobar si el archivo PDF existe, si no, descargarlo
if not os.path.exists(FilePDF):
    descargar_pdf(FilePDF)

# Extraer información del PDF y crear el DataFrame
df = extraer_informacion_pdf(FilePDF)

# Descargar el logo del chatbot si no existe
if not os.path.exists(FileLogo):
    descargar_logo(FileLogo)
image = Image.open(FileLogo)

# Mostrar las primeras filas del DataFrame para verificar que se ha cargado correctamente
#st.write(df.head())
#st.dataframe(df.head())  # Alternativa para mostrar el DataFrame de forma interactiva
#st.write("Datos cargados desde el archivo Excel existente.")
#df.head()
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Configuración de la aplicación Streamlit ---
st.set_page_config(page_title="Chatbot de Ciclos Formativos", layout="centered")

# Mostrar el logo del chatbot
st.image(image, caption='Chatbot-FP', width=200)

st.title("📚 Chatbot de Ciclos Formativos")

# Inicializar el estado de la sesión si no existe
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "model" not in st.session_state:
    st.session_state.model = None

# Cargar el modelo de embeddings solo una vez
if st.session_state.model is None:
    # Mostrar mensaje de preparación del entorno
    st.write("Preparando el entorno (esto puede tardar unos segundos)...")
    st.session_state.model = load_embedding_model()

if st.session_state.excel_data is None:
    try:
        # Asegurarse de que las columnas esperadas existan o manejar su ausencia
        required_cols = ['Familia Profesional', 'Grado', 'Código Ciclo', 'Nombre Ciclo', 'Provincia', 'Municipio', 'Instituto', 'Curso', 'Turno', 'Bilingüe', 'Nuevo']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Columnas faltantes en el Excel: {', '.join(missing_cols)}. El chatbot podría no funcionar correctamente.")
            # Intentar crear las columnas faltantes con valores vacíos para que el script no falle
            for col in missing_cols:
                df[col] = ''

        st.session_state.excel_data = df
        st.session_state.faiss_index, st.session_state.corpus = create_faiss_index(st.session_state.excel_data, st.session_state.model)
        st.success("¡Chatbot iniciado correctamente! Ahora puedes hacer preguntas.")

        # Limpiar historial de chat al cargar un nuevo archivo
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Error al leer el archivo de datos o crear el índice: {e}")
        st.session_state.excel_data = None
        st.session_state.faiss_index = None
        st.session_state.corpus = None

# Mostrar historial de chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de usuario
if st.session_state.excel_data is not None and st.session_state.faiss_index is not None:
    user_query = st.chat_input("Haz tu pregunta sobre los ciclos formativos...")
    if user_query:
        with st.spinner("Pensando...", show_time=True):
            response = ask_rag_model(
                user_query,
                st.session_state.faiss_index,
                st.session_state.corpus,
                st.session_state.model,
                st.session_state.excel_data
            )
        with st.chat_message("assistant"):
            st.write(response)
            # Convertir respuesta del bot a audio base64
            audio_b64 = text_to_audio_base64(response, lang='es')
            if audio_b64:
                st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                # Alternativa para incrustar el audio directamente en HTML (opcional)
                # Nota: Streamlit no soporta incrustar audio directamente en HTML de forma nativa.
                # Puedes usar el siguiente código para incrustar el audio en HTML, pero ten en cuenta que
                # puede no funcionar en todos los navegadores debido a restricciones de autoplay.
                # audio_html = f"""
                # <audio controls autoplay style="width: 100%;">
                #     <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                #     Tu navegador no soporta el elemento de audio.
                # </audio>
                # """
                # st.markdown(audio_html, unsafe_allow_html=True)
                # Nota: Streamlit no tiene un botón de "reproducir" nativo para audio como el que puedes incrustar manualmente.
                # El control "autoplay" intentará reproducir el audio automáticamente, pero el navegador
                # puede bloquearlo si no ha habido interacción previa del usuario.
                # Los controles "controls" añaden la barra de reproducción estándar.

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- Configuración de la barra lateral y opciones adicionales ---
#st.image(image, caption='Chatbot de Ciclos Formativos', use_column_width=True)

# Mostrar información del chatbot
st.sidebar.header("Chatbot de Ciclos Formativos en Extremadura")
st.sidebar.markdown("""
    Este chatbot te permite hacer preguntas sobre los ciclos formativos en Extremadura basándose en datos extraídos del PDF indicado. \n
    Puedes preguntar sobre:
    - Ciclos formativos disponibles
    - Institutos y centros educativos
    - Familias profesionales
    - Grados y niveles de formación
    - Información sobre turnos y modalidades (diurno, vespertino, bilingüe, etc.)
    - Y mucho más relacionado con la oferta formativa en Extremadura.
    \n\n
""")

# Mostrar información del archivo PDF y Excel
show_datos = st.sidebar.checkbox("¿Mostrar datos utilizados?")
if show_datos:
    #st.sidebar.subheader("Información utilizada")
    st.sidebar.write(f"- Fuente: `{FilePDF}`")
    #st.sidebar.write(f"- `{FileExcel}`")

    if st.session_state.excel_data is not None:
        st.sidebar.write(f"- Nº Ciclos Formativos: {len(st.session_state.excel_data):,.0f}".replace(",", "."))

    #if st.session_state.model is not None:
    #    st.sidebar.write(f"- Modelo: `{ModeloEmbeddings}`")

new_pdf = st.sidebar.checkbox("¿Cargar nuevo PDF de datos?")
if new_pdf:
    # Cargar PDF
    pdf_obj = st.sidebar.file_uploader("Carga el documento PDF fuente", type="pdf")
    # Si se carga un PDF, procesarlo
    if pdf_obj is not None:
        # Guardar el PDF en un archivo temporal
        with open(FilePDF, "wb") as f:
            f.write(pdf_obj.getbuffer())
        # Extraer información del PDF y crear el DataFrame
        df = extraer_informacion_pdf(FilePDF)
        st.session_state.excel_data = df
        st.session_state.faiss_index, st.session_state.corpus = create_faiss_index(df, st.session_state.model)
        st.success("¡Datos cargados, embeddings e índice FAISS creados correctamente! Ahora puedes hacer preguntas.")
        # Limpiar historial de chat al cargar un nuevo archivo
        st.session_state.chat_history = []

# Mostrar el DataFrame cargado desde el PDF
# if st.session_state.excel_data is not None:
#     st.subheader("Datos Cargados desde el PDF")
#     st.dataframe(st.session_state.excel_data.head())  # Mostrar las primeras filas del DataFrame
# else:
#     st.info("Por favor, sube un archivo PDF para empezar a interactuar con el chatbot.")

#show_historial = st.sidebar.checkbox("¿Mostrar el Historial del Chat?")
#if show_historial:
#    st.sidebar.subheader("Historial de Chat")
#    if st.session_state.chat_history:
#        st.sidebar.write(f"Total de mensajes en el historial: {len(st.session_state.chat_history)}")
#        if len(st.session_state.chat_history) > 0:
#            st.sidebar.write("Último mensaje:")
#            last_message = st.session_state.chat_history[-1]
#            st.sidebar.write(f"{last_message['role']}: {last_message['content']}")
#    else:
#        st.sidebar.write("No hay mensajes en el historial de chat.")

# Opcional: Botón para limpiar el historial de chat
if st.sidebar.button("Reiniciar Chat"):
    st.session_state.clear()  # Borra todas las variables de sesión
    #st.session_state.messages = []
    # if not st.session_state.get("messages"):
    #     st.write("Historial de chat vacío. 🎉")
    st.rerun()

# Footer
st.sidebar.markdown("""
    ---
    **Desarrollado por:**
    - Antonio J. Abasolo Sierra
    - José David Honrado García
""")
