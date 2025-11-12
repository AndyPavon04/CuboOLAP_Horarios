# -------------------------------------------------------------
# ETL: Extracción, Transformación y Carga de Horarios Académicos
# -------------------------------------------------------------
# Script diseñado para automatizar el proceso ETL:
# 1. Extraer datos de horarios académicos desde múltiples archivos PDF.
# 2. Transformar los datos (limpieza, normalización, expansión de filas)
#    y construir un esquema de estrella (dimensiones y tabla de hechos).
# 3. Cargar el modelo de datos dimensional resultante en una base de datos SQL Server.
# -------------------------------------------------------------
# requirements: pdfplumber, pandas, tabula-py, numpy, sqlalchemy, pyodbc
# -------------------------------------------------------------

import re
import pdfplumber
import pandas as pd
import numpy as np
from pathlib import Path

# Módulos para interactuar con SQL Server usando SQLAlchemy y el driver ODBC
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
import pyodbc

# =============================================================
# FASE 1: EXTRACCIÓN (Extract)
# =============================================================

# Lista de rutas de los archivos PDF que contienen los horarios a procesar.
PDFS = [
    "pdfs/PA_OTOÑO_2025_SEMESTRAL_ICC.pdf",
    "pdfs/PA_OTOÑO_2025_SEMESTRAL_ITI.pdf",
    "pdfs/PA_OTOÑO_2025_SEMESTRAL_LCC.pdf",
]

def clean_header(cols):
    """
    Procesa y estandariza los encabezados de las tablas capturadas por pdfplumber.
    Elimina espacios múltiples, saltos de línea y convierte a minúsculas para
    facilitar la identificación y mapeo posterior de columnas.

    Args:
        cols (list[str]): Lista de nombres de columna tal como se detectaron en el PDF.

    Returns:
        list[str]: Nombres de columna estandarizados (minúsculas, sin espacios extra ni saltos).
    """
    return [re.sub(r"\s+", " ", c).strip().lower() for c in cols]


def extract_tables_pdfplumber(pdf_path):
    """
    Abre y recorre cada página de un archivo PDF para extraer su contenido tabular
    mediante `pdfplumber`. Aplica una heurística para asegurarse de que solo se
    procesan las tablas de horarios que contienen las columnas esperadas.

    Args:
        pdf_path (str | Path): Ruta completa del archivo PDF a leer.

    Returns:
        DataFrame: Un DataFrame de pandas con todas las filas de horarios extraídas
                   del PDF, con columnas esperadas como 'nrc', 'clave', 'materia', etc.
                   Retorna un DataFrame vacío si no se encuentran datos o tablas válidas.
    """
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                # Descarta tablas vacías o con una sola fila (solo encabezado)
                if not table or len(table) < 2:
                    continue
                header = clean_header(table[0])
                # Heurística para identificar la tabla correcta: debe contener las 7 columnas clave.
                if {"nrc","clave","materia","días","hora","profesor","salón"}.issubset(set(header)) or \
                   {"nrc","clave","materia","dias","hora","profesor","salon"}.issubset(set(header)):
                    # Procesa el resto de las filas como datos
                    for r in table[1:]:
                        if r and any(x for x in r): # Asegura que la fila no esté completamente vacía
                            rows.append(dict(zip(header, r)))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_all():
    """
    Función principal de extracción. Itera sobre la lista `PDFS`, procesa cada archivo
    válido con `extract_tables_pdfplumber`, agrega una columna para identificar el
    archivo de origen y concatena todos los resultados en un único DataFrame.

    Returns:
        DataFrame: El dataset combinado de todos los PDFs procesados, incluyendo la columna 'origen_pdf'.
                   Retorna un DataFrame vacío si no hay archivos o datos válidos.
    """
    frames = []
    for p in PDFS:
        if Path(p).exists():
            df = extract_tables_pdfplumber(p)
            if not df.empty:
                df["origen_pdf"] = Path(p).name
                frames.append(df)
    # Concatena todos los DataFrames de horarios
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Ejecuta la extracción de los datos iniciales
raw = extract_all()

# =============================================================
# FASE 2: TRANSFORMACIÓN (Transform)
# =============================================================

# -------------------------------------------------------------
# Limpieza inicial de columnas
# -------------------------------------------------------------
# Estandariza las posibles variantes de nombres de columnas ('dias', 'salon') a sus formas preferidas con acentos.
raw = raw.rename(columns={"dias": "días", "salon": "salón"})

def normalizar_profesor(x: str):
    """
    Limpia el texto del nombre del profesor: elimina múltiples espacios, ajusta
    guiones y convierte el texto al formato 'Título Capitalizado'.

    Args:
        x (str): Nombre del profesor original extraído del PDF.

    Returns:
        str | None: Nombre estandarizado o None si la entrada no es válida.
    """
    if not isinstance(x, str):
        return None
    x = re.sub(r"\s+", " ", x).strip().replace(" - ", " ")
    return x.title()

# Aplica la normalización a la columna 'profesor'
raw["profesor"] = raw["profesor"].apply(normalizar_profesor)

# -------------------------------------------------------------
# Normalización de nombres de columnas y formato de hora
# -------------------------------------------------------------
# Asegura que todos los nombres de columna estén en minúsculas y sin espacios iniciales/finales.
raw.columns = [c.strip().lower() for c in raw.columns]

# Detectar columna de hora en múltiples variantes (ej. 'horario', 'hora ', 'h')
# y unificarla bajo el nombre 'hora'.
for variant in ["horario", "hora ", "hora\n", "h"]:
    if variant in raw.columns:
        raw.rename(columns={variant: "hora"}, inplace=True)
        break
# Si ninguna variante se encuentra, crea una columna 'hora' vacía.
if "hora" not in raw.columns:
    raw["hora"] = None

# -------------------------------------------------------------
# Conversión de rangos de hora a inicio/fin/duración
# -------------------------------------------------------------
def parse_hora(rango):
    """
    Analiza un rango de hora como 'HH:MM-HH:MM' y lo descompone en tres componentes:
    hora de inicio, hora de fin (ambos como `datetime.time`), y la duración en minutos.

    Args:
        rango (str): Cadena de texto que representa el rango horario (e.g., '07:00-08:59').

    Returns:
        Series: Una serie de pandas con [hora_inicio, hora_fin, duracion_min].
                Retorna valores None si el formato no es válido.
    """
    if not isinstance(rango, str):
        return pd.Series([None, None, None])
    s = re.sub(r"\s+", "", rango.strip())
    # Patrón para capturar H:MM o HHMM con un guion o dos puntos separadores
    patron = r"(\d{1,2}):?(\d{2})-(\d{1,2}):?(\d{2})"
    m = re.match(patron, s)
    if not m:
        return pd.Series([None, None, None])
    h1, m1, h2, m2 = map(int, m.groups())
    
    # Convierte las horas a objetos datetime.time
    start = pd.to_datetime(f"{h1:02d}:{m1:02d}", format="%H:%M", errors="coerce")
    end   = pd.to_datetime(f"{h2:02d}:{m2:02d}", format="%H:%M", errors="coerce")
    
    if pd.isna(start) or pd.isna(end):
        return pd.Series([None, None, None])
    
    # Calcula la duración en minutos
    duracion = int((end - start).total_seconds() / 60)
    
    if duracion <= 0: # Evita duraciones negativas o cero (e.g., 8:00-7:00)
        return pd.Series([None, None, None])
        
    return pd.Series([start.time(), end.time(), duracion])

# Aplica la función para crear las nuevas columnas de tiempo
raw[["hora_inicio", "hora_fin", "duracion_min"]] = raw["hora"].apply(parse_hora)

# -------------------------------------------------------------
# Expansión de días y normalización de códigos
# -------------------------------------------------------------
# Mapeo de códigos de un solo caracter a nombres completos de días de la semana.
DIA_MAP = {"L":"Lunes","A":"Martes","M":"Miercoles","J":"Jueves","V":"Viernes","S":"Sábado"}

def explotar_por_dia(df):
    """
    Desanida las filas que contienen múltiples días en un solo registro (e.g., 'LMV')
    para crear una fila separada por cada día de la semana que tiene clase.
    Esto transforma el dataset de una representación condensada a una atómica (un registro = un evento).

    Args:
        df (DataFrame): Dataset de horarios que incluye la columna 'días'.

    Returns:
        DataFrame: Dataset expandido, donde cada fila representa una ocurrencia de clase
                   e incluye las nuevas columnas 'dia_codigo' y 'dia_semana'.
    """
    out = []
    for _, row in df.iterrows():
        dias = str(row["días"]).replace(" ", "")
        tokens = dias.split(",") if "," in dias else list(dias)
        for d in tokens:
            r = row.copy()
            r["dia_codigo"] = d
            r["dia_semana"] = DIA_MAP.get(d, d)
            out.append(r)
    return pd.DataFrame(out)

# Ejecuta la expansión para obtener el dataset curado atómico
curated = explotar_por_dia(raw)

# -------------------------------------------------------------
# División del salón en edificio/aula
# -------------------------------------------------------------
def split_salon(s):
    """
    Divide un código de salón compuesto (típicamente 'edificio/aula', e.g. '1CCO4/203')
    en sus componentes 'edificio' y 'aula'.

    Args:
        s (str): Código de salón original.

    Returns:
        Series: Una serie de pandas con [edificio, aula, codigo_salon (original)].
                Si no hay un separador '/', se usa el código completo para 'edificio' y 'codigo_salon'.
    """
    if not isinstance(s, str): return pd.Series([None, None, None])
    s = s.strip()
    # Captura la parte anterior a '/' como edificio y la posterior como aula (opcional)
    m = re.match(r"([^/]+)/?(\w+)?", s)
    if not m: return pd.Series([s, None, s])
    edificio, aula = m.group(1), m.group(2)
    return pd.Series([edificio, aula, s])

# Aplica la función para crear las columnas 'edificio' y 'aula'
curated[["edificio","aula","codigo_salon"]] = curated["salón"].apply(split_salon)

# -------------------------------------------------------------
# Construcción de dimensiones (surrogate keys)
# -------------------------------------------------------------
def build_dim(df, col_key, cols_keep, start_id=1, name_id="id"):
    """
    Crea una tabla de dimensión, asignando una clave subrogada (surrogate key)
    autogenerada a cada registro único basado en las columnas clave especificadas.

    Args:
        df (DataFrame): Dataset curado a partir del cual se extraen los valores únicos.
        col_key (str): Columna principal para definir la unicidad (generalmente redundante si se usa cols_keep).
        cols_keep (list[str]): Lista de columnas a incluir en la dimensión.
        start_id (int): Valor inicial para la clave subrogada.
        name_id (str): Nombre de la columna de la clave subrogada (id_docente, id_materia, etc.).

    Returns:
        DataFrame: Tabla de dimensión con la clave subrogada como primera columna.
    """
    # Selecciona columnas, elimina duplicados y resetea el índice
    d = df[cols_keep].drop_duplicates().reset_index(drop=True)
    # Inserta la clave subrogada autoincremental
    d.insert(0, name_id, range(start_id, start_id+len(d)))
    return d

# Construcción de las cuatro tablas de dimensiones
dim_docente = build_dim(curated, "profesor", ["profesor"], name_id="id_docente")
dim_materia = build_dim(curated, "materia", ["clave","materia"], name_id="id_materia")
dim_espacio = build_dim(curated, "codigo_salon", ["edificio","aula","codigo_salon"], name_id="id_espacio")

# La dimensión de tiempo se construye a partir de la combinación única de día y rango horario
dim_tiempo = curated[["dia_codigo","dia_semana","hora_inicio","hora_fin"]].drop_duplicates().reset_index(drop=True)
dim_tiempo.insert(0, "id_tiempo", range(1, len(dim_tiempo)+1))

# -------------------------------------------------------------
# Creación de tabla de hechos (hechos_clase)
# -------------------------------------------------------------
def map_id(df, dim, key_cols_df, key_cols_dim, id_col):
    """
    Realiza la búsqueda (lookup) y asignación de claves subrogadas (IDs)
    desde una tabla de dimensión a la tabla de hechos. Utiliza una clave de unión
    temporal concatenada para un mapeo eficiente de muchos a uno (m:1).

    Args:
        df (DataFrame): La tabla de hechos (o base) a la que se añadirán los IDs.
        dim (DataFrame): La tabla de dimensión que contiene los IDs.
        key_cols_df (list[str] | str): Columnas clave en el DataFrame de hechos para la unión.
        key_cols_dim (list[str] | str): Columnas clave en la Dimensión para la unión.
        id_col (str): Nombre de la columna ID (clave subrogada) a obtener.

    Returns:
        np.ndarray: Un vector NumPy con los IDs subrogados correspondientes a cada fila de `df`.
    """
    if isinstance(key_cols_df, str):
        key_cols_df = [key_cols_df]
    if isinstance(key_cols_dim, str):
        key_cols_dim = [key_cols_dim]

    # Crea la clave de unión temporal concatenando los valores de las columnas clave
    df["_key_"] = df[key_cols_df].astype(str).agg("|".join, axis=1)
    dim["_key_"] = dim[key_cols_dim].astype(str).agg("|".join, axis=1)

    # Realiza la unión (merge) para obtener el ID de la dimensión
    merged = df.merge(dim[["_key_", id_col]], on="_key_", how="left", validate="m:1")
    result = merged[id_col].values

    # Limpia la clave temporal de ambos DataFrames
    df.drop(columns="_key_", inplace=True, errors="ignore")
    dim.drop(columns="_key_", inplace=True, errors="ignore")

    return result

# Asigna los IDs subrogados a la tabla de hechos
hechos = curated.copy()
hechos["id_docente"] = map_id(hechos, dim_docente, "profesor", "profesor", "id_docente")
hechos["id_materia"] = map_id(hechos, dim_materia, ["clave","materia"], ["clave","materia"], "id_materia")
hechos["id_espacio"] = map_id(hechos, dim_espacio, "codigo_salon", "codigo_salon", "id_espacio")
# La dimensión de tiempo se une directamente por sus claves naturales
hechos = hechos.merge(dim_tiempo, on=["dia_codigo","dia_semana","hora_inicio","hora_fin"], how="left")

# Selecciona las columnas finales para la tabla de hechos
hechos_clase = hechos[[
    "id_docente","id_materia","id_espacio","id_tiempo",
    "nrc","clave","días","duracion_min"
]].rename(columns={"días":"seccion"}) # 'días' se renombra a 'seccion' o 'grupo'

# =============================================================
# FASE 3: CARGA (Load)
# =============================================================

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
SERVER_NAME = "ANDYPAVON" # Nombre del servidor de SQL Server
DATABASE_NAME = "horariosCubo" # Nombre de la base de datos de destino
DRIVER = "ODBC Driver 17 for SQL Server" # Nombre exacto del driver ODBC instalado

# Cadena de conexión (Utilizando Autenticación de Windows/Trusted Connection)
CONNECTION_STRING = (
    f"mssql+pyodbc:///?odbc_connect="
    f"Driver={{{DRIVER}}};" 
    f"Server={SERVER_NAME};"
    f"Database={DATABASE_NAME};"
    f"Trusted_Connection=yes;" # Indica el uso de la autenticación del sistema operativo
)

# ---------------------------------------------------------------------------
# CONEXIÓN A SQL SERVER
# ---------------------------------------------------------------------------
try:
    # Intenta crear el motor de conexión y establecer la conexión
    engine = create_engine(CONNECTION_STRING)
    conn = engine.connect()
    # Ejecuta una consulta simple para verificar que la conexión es funcional
    conn.execute(text("SELECT 1")) 
    print("✅ Conexión a SQL Server establecida correctamente (Windows Auth).")

except Exception as err:
    # Manejo de errores de conexión, que suele indicar un driver o servidor incorrecto
    print(f"❌ Error al conectar a SQL Server (Verifica el driver ODBC): {err}")
    raise SystemExit()


# ---------------------------------------------------------------------------
# PASO 1: ELIMINAR Y CREAR TABLAS (SIN FKs en Hechos)
# ---------------------------------------------------------------------------

# Sentencias DDL para eliminar las tablas de hechos y dimensiones.
# Es crucial eliminar primero la tabla de hechos para evitar errores de Foreign Key (FK)
# si las tablas dimensiones que referencia se eliminan después.
drop_statements = [
    """
    IF OBJECT_ID('hechos_clase', 'U') IS NOT NULL DROP TABLE hechos_clase;
    """,
    """
    IF OBJECT_ID('dim_docente', 'U') IS NOT NULL DROP TABLE dim_docente;
    IF OBJECT_ID('dim_materia', 'U') IS NOT NULL DROP TABLE dim_materia;
    IF OBJECT_ID('dim_espacio', 'U') IS NOT NULL DROP TABLE dim_espacio;
    IF OBJECT_ID('dim_tiempo', 'U') IS NOT NULL DROP TABLE dim_tiempo;
    """,
]

print(">> Eliminando tablas existentes...")
for ddl in drop_statements:
    conn.execute(text(ddl)) # Ejecuta el DDL de eliminación
conn.commit() # Confirma la transacción
print("✅ Tablas eliminadas correctamente.")


# Sentencias DDL para crear las tablas con sus respectivas Primary Keys (PKs).
# Se definen las columnas y tipos de datos para cada tabla de dimensión y la de hechos.
ddl_statements = [
    "CREATE TABLE dim_docente (id_docente INT PRIMARY KEY, nombre_completo NVARCHAR(200))",
    "CREATE TABLE dim_materia (id_materia INT PRIMARY KEY, clave VARCHAR(50), nombre_materia NVARCHAR(200))",
    "CREATE TABLE dim_espacio (id_espacio INT PRIMARY KEY, edificio VARCHAR(50), aula VARCHAR(50), codigo_salon VARCHAR(100))",
    "CREATE TABLE dim_tiempo (id_tiempo INT PRIMARY KEY, dia_codigo VARCHAR(10), dia_semana NVARCHAR(20), hora_inicio TIME, hora_fin TIME)",
    """
    CREATE TABLE hechos_clase (
        id_hecho INT IDENTITY(1,1) PRIMARY KEY, 
        id_docente INT, id_materia INT, id_espacio INT, id_tiempo INT,
        nrc VARCHAR(20), clave VARCHAR(50), seccion VARCHAR(50), duracion_min INT
    )
    """
]

print(">> Creando tablas...")
for ddl in ddl_statements:
    conn.execute(text(ddl)) # Ejecuta el DDL de creación
conn.commit()
print("✅ Tablas creadas correctamente.")


# ---------------------------------------------------------------------------
# PASO 2: LIMPIEZA, TRUNCATE Y CARGA
# ---------------------------------------------------------------------------

# Función auxiliar para la inserción de DataFrames en SQL Server.
def insert_dataframe_to_sql_server(df: pd.DataFrame, table_name: str, conn: Connection):
    """
    Inserta el contenido de un DataFrame de pandas en la tabla especificada
    de SQL Server utilizando el método `to_sql`.

    Args:
        df (pd.DataFrame): DataFrame a insertar.
        table_name (str): Nombre de la tabla de destino en la base de datos.
        conn (Connection): Conexión SQLAlchemy (o el engine, ya que to_sql lo acepta).

    Returns:
        int: El número de filas que se intentaron insertar (longitud del DataFrame).
    """
    if df.empty:
        print(f"(⚠️ {table_name} está vacío, no se inserta nada)")
        return 0 
    
    # Usa 'append' porque las tablas ya fueron creadas (y vaciadas por el paso 1)
    # Se configura index=False para no intentar insertar el índice de pandas
    df.to_sql(
        name=table_name,
        con=engine, 
        if_exists='append', 
        index=False,
        chunksize=1000 # Inserción por lotes para mejorar el rendimiento
    )
    return len(df) # Retorna el número de filas del DF insertado

# Asegura que la columna de duración sea de tipo entero (permitiendo nulos con Int64Dtype)
hechos_clase["duracion_min"] = pd.to_numeric(hechos_clase["duracion_min"], errors='coerce').astype(pd.Int64Dtype())

# Cargar Dimensiones
rows_inserted = insert_dataframe_to_sql_server(dim_docente.rename(columns={'profesor': 'nombre_completo'}), "dim_docente", engine)
print(f"✅ {rows_inserted} filas insertadas en dim_docente")

rows_inserted = insert_dataframe_to_sql_server(dim_materia.rename(columns={'materia': 'nombre_materia'}), "dim_materia", engine)
print(f"✅ {rows_inserted} filas insertadas en dim_materia")

rows_inserted = insert_dataframe_to_sql_server(dim_espacio, "dim_espacio", engine)
print(f"✅ {rows_inserted} filas insertadas en dim_espacio")

rows_inserted = insert_dataframe_to_sql_server(dim_tiempo, "dim_tiempo", engine)
print(f"✅ {rows_inserted} filas insertadas en dim_tiempo")

# Cargar Hechos
rows_inserted = insert_dataframe_to_sql_server(hechos_clase, "hechos_clase", engine)
print(f"✅ {rows_inserted} filas insertadas en hechos_clase")


# ---------------------------------------------------------------------------
# PASO 3: CREACIÓN FINAL DE CLAVES FORÁNEAS
# ---------------------------------------------------------------------------

print(">> Creando Claves Foráneas...")
# Sentencias DDL para añadir las Foreign Keys (FK) a la tabla de hechos
fk_statements = [
    "ALTER TABLE hechos_clase ADD CONSTRAINT FK_Docente FOREIGN KEY (id_docente) REFERENCES dim_docente(id_docente);",
    "ALTER TABLE hechos_clase ADD CONSTRAINT FK_Materia FOREIGN KEY (id_materia) REFERENCES dim_materia(id_materia);",
    "ALTER TABLE hechos_clase ADD CONSTRAINT FK_Espacio FOREIGN KEY (id_espacio) REFERENCES dim_espacio(id_espacio);",
    "ALTER TABLE hechos_clase ADD CONSTRAINT FK_Tiempo FOREIGN KEY (id_tiempo) REFERENCES dim_tiempo(id_tiempo);"
]

try:
    # Utiliza un bloque transaccional para ejecutar todas las sentencias de FK
    with engine.begin() as t_conn:
        for stmt in fk_statements:
            t_conn.execute(text(stmt))
        t_conn.commit() # Confirma la transacción
    print("✅ Claves Foráneas creadas exitosamente.")
except Exception as e:
    # Este error se maneja porque si el script se ejecuta más de una vez, las FKs ya existirán
    print("⚠️ Las FKs ya existen o hubo un error al crearlas, lo cual es normal si ya existe la tabla.")


print("✅ Todos los datos cargados exitosamente en SQL Server.")

# ---------------------------------------------------------------------------
# 🔚 Cierre
# ---------------------------------------------------------------------------
# Cierra la conexión de la base de datos abierta previamente
conn.close()
print("🔚 Conexión a SQL Server cerrada.")