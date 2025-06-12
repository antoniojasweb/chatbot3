# -------------------------------------------------------------------
#pip install streamlit pandas faiss-cpu sentence-transformers requests
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Importar las librerías necesarias
# -------------------------------------------------------------------
import fitz  # PyMuPDF
import pandas as pd
import os
import requests

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
import os

import streamlit as st
from gtts import gTTS
import io  # Necesario para manejar el buffer de audio
import base64 # Necesario para incrustar audio directamente en HTML
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Definición de rutas y ficheros de datos
url = "https://raw.githubusercontent.com/antoniojasweb/chatbot/main/pdf/"
FilePDF = "25_26_OFERTA_por_Familias.pdf"
FileExcel = "oferta_formativa_completa.xlsx"

# Otros Parámetros de configuración
ModeloEmbeddings = 'paraphrase-multilingual-MiniLM-L12-v2'
# -------------------------------------------------------------------

# -------------------------------------------------------------------
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
    with st.expander("Ver información recuperada: " + str(top_k) + " opciones más relevantes"):
        st.write(retrieved_docs_df[['Nombre Ciclo', 'Municipio', 'Provincia', 'Grado', 'Familia Profesional']])

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

#if os.path.exists(FileExcel):
#     os.remove(FileExcel)

# Comprobar si el archivo Excel ya existe, si no, extraer información del PDF y crear el DataFrame
# Si el archivo Excel ya existe, cargarlo directamente
if not os.path.exists(FileExcel):
    df = extraer_informacion_pdf(FilePDF)
else:
    df = pd.read_excel(FileExcel)

    # Mostrar las primeras filas del DataFrame para verificar que se ha cargado correctamente
    #st.write(df.head())
    #st.dataframe(df.head())  # Alternativa para mostrar el DataFrame de forma interactiva
    #st.write("Datos cargados desde el archivo Excel existente.")

#df.head()
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Configuración de la aplicación Streamlit ---
st.set_page_config(page_title="Chatbot de Ciclos Formativos", layout="centered")

st.title("📚 Chatbot de Ciclos Formativos")
#st.subheader("Trabajando según los datos del fichero: " + FilePDF)
#st.markdown("¡Sube un archivo Excel con información de ciclos formativos y pregúntame lo que quieras sobre ellos!")

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
    #st.write("Cargando modelo de embeddings (esto puede tardar unos segundos)...")
    st.write("Preparando el entorno (esto puede tardar unos segundos)...")
    st.session_state.model = load_embedding_model()

#st.write("Trabajando según los datos del fichero: " + FilePDF)
#st.markdown("Trabajando según los datos del fichero: " + FilePDF)

# Carga de archivo Excel
#uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
#if uploaded_file is not None and st.session_state.excel_data is None:

if st.session_state.excel_data is None:
    #st.write("Archivo Excel subido. Procesando...")
    try:
        #df = pd.read_excel(uploaded_file)
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
        with st.spinner("Pensando..."):
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

# else:
#     st.info("Por favor, sube un archivo PDF para empezar a interactuar con el chatbot.")
#     st.markdown("""
#     **Formato de ejemplo para el Excel:**
#     Tu archivo Excel debe tener, al menos, las siguientes columnas para un funcionamiento óptimo:
#     - `Nombre Ciclo`
#     - `Grado` (Ej: Grado Medio, Grado Superior)
#     - `Familia Profesional`
#     - `Instituto`
#     - `Municipio`
#     - `Provincia`
#     - `Turno`
#     """)

# --- Configuración de la barra lateral y opciones adicionales ---
# Configuración de la barra lateral
#st.sidebar.title("Opciones del Chatbot")
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
show_datos = st.sidebar.checkbox("¿Mostrar datos utilizados?", value=True)
if show_datos:
    if st.session_state.excel_data is not None:
        st.sidebar.subheader("Fuente de Datos")
        st.sidebar.write(f"Archivo PDF fuente: `{FilePDF}`")
        st.sidebar.write(f"Archivo Excel generado: `{FileExcel}`")
        st.sidebar.write(f"Total de ciclos formativos: {len(st.session_state.excel_data)}")
        # st.sidebar.markdown("""
        #     El archivo PDF contiene información sobre los ciclos formativos en Extremadura, incluyendo detalles sobre familias profesionales, grados, centros educativos y más.
        #     \n\n
        #     Puedes hacer preguntas específicas sobre los ciclos formativos y el chatbot te proporcionará respuestas basadas en esta información.
        # """)
        #st.sidebar.write("Primeras filas del DataFrame:")
        #st.sidebar.dataframe(st.session_state.excel_data.head())
        # Mostrar información del modelo de embeddings
        st.sidebar.subheader("Modelo de Embeddings")
        if st.session_state.model is not None:
            st.sidebar.write(f"Modelo de embeddings cargado: `{ModeloEmbeddings}`")
        #     st.sidebar.write("Este modelo se utiliza para generar representaciones vectoriales de los textos, lo que permite buscar información relevante en el corpus.")
        # else:
        #     st.sidebar.write("Modelo de embeddings no cargado. Asegúrate de que el modelo se ha inicializado correctamente.")
else:
    st.sidebar.write("No se han cargado datos.")


# Mostrar información del índice FAISS
# if st.session_state.faiss_index is not None:
#     st.sidebar.subheader("Índice FAISS")
#     st.sidebar.write("Índice FAISS creado con éxito.")
# else:
#     st.sidebar.write("Índice FAISS no creado. Asegúrate de cargar un archivo Excel válido.")

# Mostrar información del corpus
# if st.session_state.corpus is not None:
#     st.sidebar.subheader("Corpus de Documentos")
#     st.sidebar.write(f"Total de documentos en el corpus: {len(st.session_state.corpus)}")
# else:
#     st.sidebar.write("Corpus no disponible. Asegúrate de cargar un archivo Excel válido.")

# --- Instrucciones de uso ---
# Mostrar instrucciones de uso
# st.sidebar.subheader("Instrucciones de Uso")
# st.sidebar.markdown("""
#     1. **Sube un archivo PDF**: Asegúrate de que el archivo contenga información sobre ciclos formativos en Extremadura.
#     2. **Haz preguntas**: Utiliza el campo de entrada para hacer preguntas sobre los ciclos formativos.
#     3. **Explora el historial de chat**: Puedes ver las preguntas y respuestas anteriores en el historial de chat.
#     4. **Opciones adicionales**: Puedes cargar un nuevo archivo Excel o limpiar el historial de chat desde la barra lateral.
# """)

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

st.sidebar.write("\n")

# Mostrar el DataFrame cargado desde el PDF
# if st.session_state.excel_data is not None:
#     st.subheader("Datos Cargados desde el PDF")
#     st.dataframe(st.session_state.excel_data.head())  # Mostrar las primeras filas del DataFrame
# else:
#     st.info("Por favor, sube un archivo PDF para empezar a interactuar con el chatbot.")


# Sidebar para opciones adicionales
#st.sidebar.header("Opciones del Chatbot")
# st.sidebar.markdown("""
#     - **Cargar un nuevo archivo PDF**: Si quieres cambiar los datos, sube un nuevo archivo.
#     - **Limpiar el historial de chat**: Puedes limpiar el historial de chat si lo deseas.
# """)
# Opcional: Botón para cargar un nuevo archivo Excel
# if st.sidebar.button("Cargar nuevo archivo PDF"):
#     st.session_state.excel_data = None
#     st.session_state.faiss_index = None
#     st.session_state.corpus = None
#     st.session_state.chat_history = []
#     st.rerun()  # Recargar la aplicación para permitir la carga de un nuevo archivo
#     st.info("Por favor, sube un nuevo archivo PDF para empezar a interactuar con el chatbot.")

show_historial = st.sidebar.checkbox("¿Mostrar el Historial del Chat?")
if show_historial:
    st.sidebar.subheader("Historial de Chat")
    if st.session_state.chat_history:
        st.sidebar.write(f"Total de mensajes en el historial: {len(st.session_state.chat_history)}")
        if len(st.session_state.chat_history) > 0:
            st.sidebar.write("Último mensaje:")
            last_message = st.session_state.chat_history[-1]
            st.sidebar.write(f"{last_message['role']}: {last_message['content']}")
    else:
        st.sidebar.write("No hay mensajes en el historial de chat.")

st.sidebar.write("\n")

# Opcional: Botón para limpiar el historial de chat
if st.sidebar.button("Limpiar Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.sidebar.markdown("""
    ---
    **Desarrollado por:**
    - Antonio J. Abasolo Sierra
    - José David Honrado García
""")
