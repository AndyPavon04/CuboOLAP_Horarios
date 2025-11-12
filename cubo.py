# app/cubo.py
import os
import datetime
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus # Necesario para codificar el nombre del driver en la URL

def get_engine():
    """
    Establece y retorna un objeto de conexión `sqlalchemy.engine.Engine`
    hacia la base de datos SQL Server, utilizando la Autenticación de Windows
    (Trusted Connection) para la conexión.

    Define los parámetros de conexión (servidor, base de datos, driver ODBC)
    y construye una URL robusta con el dialecto `mssql+pyodbc`.

    Returns:
        sqlalchemy.engine.Engine: Objeto de conexión listo para interactuar con la DB.
    """
    # Parámetros fijos, ya que la autenticación de Windows no usa usuario/password en la URL
    DB_HOST = "ANDYPAVON"  # Nombre del servidor SQL Server
    DB_NAME = "horariosCubo" # Nombre de la base de datos
    DRIVER = "ODBC Driver 17 for SQL Server" # Driver que verificaste que tienes instalado

    # Construir la cadena de conexión ODBC.
    # Usamos quote_plus para codificar el nombre del driver con espacios para la URL.
    odbc_connect = quote_plus(
        f"Driver={{{DRIVER}}};"
        f"Server={DB_HOST};"
        f"Database={DB_NAME};"
        f"Trusted_Connection=yes;"
    )

    # Construir la URL de SQLAlchemy usando el dialecto mssql+pyodbc
    url = f"mssql+pyodbc:///?odbc_connect={odbc_connect}"
    
    # Nota: También puedes usar mssql+pyodbc://ANDYPAVON/horariosCubo?driver={ODBC Driver 17 for SQL Server}&trusted_connection=yes
    # pero el formato con odbc_connect es más robusto.

    return create_engine(url)


def cargar_tablas(engine):
    """
    Ejecuta consultas SQL para cargar todas las tablas del modelo dimensional
    (hechos y sus cuatro dimensiones) desde la base de datos en DataFrames de pandas.

    Args:
        engine (sqlalchemy.engine.Engine): Conexión SQLAlchemy a la base de datos.

    Returns:
        tuple: Una tupla con los DataFrames en el orden:
            (hechos_clase, dim_docente, dim_materia, dim_espacio, dim_tiempo)
    """
    # Lectura de cada tabla de dimensión y la tabla de hechos
    dim_docente = pd.read_sql("SELECT * FROM dim_docente", engine)
    dim_materia = pd.read_sql("SELECT * FROM dim_materia", engine)
    dim_espacio = pd.read_sql("SELECT * FROM dim_espacio", engine)
    dim_tiempo  = pd.read_sql("SELECT * FROM dim_tiempo", engine)
    hechos_clase = pd.read_sql("SELECT * FROM hechos_clase", engine)
    return hechos_clase, dim_docente, dim_materia, dim_espacio, dim_tiempo


# -------------------------------------------------------------
# CLASE: CuboHorario
# -------------------------------------------------------------
class CuboHorario:
    """
    Implementa un cubo OLAP en memoria (utilizando pandas DataFrame) para el análisis
    multidimensional de horarios académicos. Esta clase consolida los hechos y las
    dimensiones cargados de la base de datos en un único DataFrame (`self.cubo`)
    y proporciona métodos para realizar consultas OLAP específicas.

    Attributes:
        hechos (DataFrame): Tabla de hechos sin las dimensiones.
        dim_docente (DataFrame): Dimensión de Docentes.
        dim_materia (DataFrame): Dimensión de Materias.
        dim_espacio (DataFrame): Dimensión de Espacios/Salones.
        dim_tiempo (DataFrame): Dimensión de Tiempo (Día y Rango Horario).
        cubo (DataFrame): El DataFrame principal resultante de unir las 5 tablas, listo para consultas.
    """

    def __init__(self, hechos, dim_docente, dim_materia, dim_espacio, dim_tiempo):
        """
        Inicializa el cubo realizando las uniones (JOINs) entre la tabla de hechos
        y cada una de las dimensiones a través de sus respectivas claves foráneas/subrogadas.
        También realiza limpieza y normalización post-carga (ej. tipos de hora, claves duplicadas).

        Args:
            hechos (DataFrame): Tabla de hechos (hechos_clase).
            dim_docente (DataFrame): Dimensión docente.
            dim_materia (DataFrame): Dimensión materia.
            dim_espacio (DataFrame): Dimensión espacio.
            dim_tiempo (DataFrame): Dimensión tiempo.
        """
        self.hechos = hechos
        self.dim_docente = dim_docente
        self.dim_materia = dim_materia
        self.dim_espacio = dim_espacio
        self.dim_tiempo = dim_tiempo

        # 🔹 Unir hechos con dimensiones (Modelo de Estrella)
        self.cubo = (
            hechos
            .merge(dim_docente, on="id_docente", how="left")
            .merge(dim_materia, on="id_materia", how="left")
            .merge(dim_espacio, on="id_espacio", how="left")
            .merge(dim_tiempo, on="id_tiempo", how="left")
        )

        # 🔹 Normalizar columnas duplicadas de 'clave'
        # Resuelve el conflicto que surge si 'clave' existe tanto en hechos como en dim_materia
        if "clave_x" in self.cubo.columns and "clave_y" in self.cubo.columns:
            # Combina 'clave_y' (dimensión) como preferente, usando 'clave_x' (hechos) como respaldo
            self.cubo["clave"] = self.cubo["clave_y"].combine_first(self.cubo["clave_x"])
            self.cubo.drop(columns=["clave_x", "clave_y"], inplace=True)
        elif "clave_x" in self.cubo.columns:
            self.cubo.rename(columns={"clave_x": "clave"}, inplace=True)
        elif "clave_y" in self.cubo.columns:
            self.cubo.rename(columns={"clave_y": "clave"}, inplace=True)

        # 🔹 Conversión robusta de hora_inicio / hora_fin a objetos datetime.time
        # Asegura que las columnas de hora sean del tipo `datetime.time` para comparaciones correctas
        for col in ["hora_inicio", "hora_fin"]:
            if col in self.cubo.columns:
                def to_time_safe(x):
                    # Helper para convertir cualquier representación de hora a datetime.time de forma segura
                    if pd.isna(x) or x in [None, "", "NaT", "None"]:
                        return None
                    if isinstance(x, datetime.time):
                        return x
                    # Maneja conversión de timedelta/Timestamp a time
                    if hasattr(x, "components") and hasattr(x, "total_seconds"):
                        total = int(x.total_seconds())
                        h, m = divmod(total, 3600)
                        m, s = divmod(m, 60)
                        return datetime.time(h, m, s)
                    try:
                        return pd.to_datetime(str(x), errors="coerce").time()
                    except Exception:
                        return None
                self.cubo[col] = self.cubo[col].apply(to_time_safe)

        # 🔹 Calcular duración si no existe
        # Recalcula la duración en minutos si la columna falta, usando hora_inicio y hora_fin
        if "duracion_min" not in self.cubo.columns and \
           {"hora_inicio", "hora_fin"}.issubset(self.cubo.columns):
            def minutos(a, b):
                # Calcula la diferencia entre dos objetos datetime.time en minutos
                if not (isinstance(a, datetime.time) and isinstance(b, datetime.time)):
                    return None
                A = datetime.timedelta(hours=a.hour, minutes=a.minute, seconds=a.second)
                B = datetime.timedelta(hours=b.hour, minutes=b.minute, seconds=b.second)
                return round((B - A).total_seconds() / 60.0, 2)
            self.cubo["duracion_min"] = self.cubo.apply(
                lambda r: minutos(r["hora_inicio"], r["hora_fin"]), axis=1
            )

    # ---------------------------------------------------------
    # 1️. Horario semanal de un docente (OLAP: Slice)
    # ---------------------------------------------------------
    def horario_docente(self, nombre_docente):
        """
        Consulta el cubo para obtener el horario detallado de clases para un docente específico.
        Permite la búsqueda por coincidencia parcial del nombre.

        Tipo de operación OLAP:
            - **Slice**: Realiza un corte o rebanada del cubo sobre la dimensión 'docente'.

        Args:
            nombre_docente (str): Nombre (o una subcadena del nombre) del docente a buscar.

        Returns:
            DataFrame: Clases programadas del docente, ordenadas por día y hora.
        """

        def format_docente_display(x: str) -> str:
            """
            Heurística para reformatear el nombre del docente de 'Apellido1 Apellido2 Nombre[s]'
            a un formato más legible como 'Nombre[s] Apellido1[ Apellido2 ]'.
            También colapsa apellidos duplicados.
            """
            if not isinstance(x, str) or not x.strip():
                return x
            partes = x.split()
            if len(partes) < 3:
                return x  # No intentar inferir si el patrón no es claro
            ap1, ap2, nombres = partes[0], partes[1], " ".join(partes[2:])
            apellidos = ap1 if ap1.lower() == ap2.lower() else f"{ap1} {ap2}"
            return f"{nombres} {apellidos}".strip()

        # 🔹 Filtrar por coincidencia parcial del nombre del docente
        df = self.cubo[self.cubo["nombre_completo"].str.contains(nombre_docente, case=False, na=False)]
        if df.empty:
            return pd.DataFrame()

        # 🔹 Normalizar y ordenar días usando tipo Categórico
        orden_dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"]
        df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=orden_dias, ordered=True)

        # 🔹 Aplicar formato visual al nombre
        df["nombre_completo"] = df["nombre_completo"].apply(format_docente_display)

        # 🔹 Ordenar resultados por día y hora
        df = df.sort_values(["dia_semana", "hora_inicio"])

        # 🔹 Seleccionar columnas de salida
        cols = [
            "nombre_completo",
            "dia_semana",
            "hora_inicio",
            "hora_fin",
            "nombre_materia",
            "clave",
            "codigo_salon",
            "edificio",
            "aula",
        ]

        # Filtra columnas que realmente existen antes de seleccionar
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ---------------------------------------------------------
    # 2️. Docentes que imparten una materia específica (OLAP: Dice)
    # ---------------------------------------------------------
    def docentes_por_materia(self, materia_o_clave):
        """
        Realiza una búsqueda para encontrar todos los docentes asociados a una materia,
        permitiendo la consulta por nombre completo de la materia o por su clave.

        Tipo de operación OLAP:
            - **Dice**: Realiza un filtrado del cubo a través de la dimensión 'materia'.

        Args:
            materia_o_clave (str): Nombre o clave de la materia a consultar (búsqueda parcial).

        Returns:
            DataFrame: Lista de docentes únicos y la materia/clave asociada.
        """
        cols = self.cubo.columns
        condiciones = []
        # Crea condiciones de filtro para buscar en nombre de materia O clave
        if "nombre_materia" in cols:
            condiciones.append(self.cubo["nombre_materia"].str.contains(materia_o_clave, case=False, na=False))
        if "clave" in cols:
            condiciones.append(self.cubo["clave"].str.contains(materia_o_clave, case=False, na=False))
            
        if not condiciones:
            return pd.DataFrame()
            
        # Combina las condiciones con un OR lógico ( | )
        filtro = condiciones[0]
        for cond in condiciones[1:]:
            filtro = filtro | cond
            
        df = self.cubo[filtro]
        if df.empty:
            return pd.DataFrame()
            
        # Retorna los resultados únicos (eliminando duplicados por clase)
        return (df[["clave","nombre_materia","nombre_completo"]]
                .drop_duplicates()
                .sort_values(["clave","nombre_completo"]))

    # ---------------------------------------------------------
    # 3️. Docentes en un edificio a una hora específica (OLAP: Dice)
    # ---------------------------------------------------------
    def docentes_en_edificio_hora(self, edificio, hora):
        """
        Realiza una consulta para identificar qué docentes están activos (dando clase)
        en un edificio específico y a una hora dada.

        Tipo de operación OLAP:
            - **Dice**: Aplica un filtro combinado sobre las dimensiones 'espacio' y 'tiempo'.

        Args:
            edificio (str): Código o nombre del edificio (búsqueda parcial).
            hora (str | datetime.time): La hora de referencia (punto de tiempo a consultar, ej. '10:00').

        Returns:
            DataFrame: Lista de docentes, materia, salón, y día para las clases que están
                       activas en ese edificio y hora.
        """
        def to_time_any(x):
            # Helper para convertir la hora de referencia a `datetime.time`
            if isinstance(x, datetime.time):
                return x
            try:
                t = pd.to_datetime(str(x), errors="coerce")
                return None if pd.isna(t) else t.time()
            except Exception:
                return None

        hora_ref = to_time_any(hora)
        if hora_ref is None:
            return pd.DataFrame()

        # Filtra filas que tienen un rango de hora válido para la comparación
        df = self.cubo.dropna(subset=["hora_inicio", "hora_fin"]).copy()

        # Condición compuesta: Edificio coincide Y (hora_ref >= hora_inicio Y hora_ref <= hora_fin)
        mask = (
            df["edificio"].astype(str).str.contains(edificio, case=False, na=False)
        ) & (
            df["hora_inicio"].apply(lambda h: isinstance(h, datetime.time) and h <= hora_ref)
        ) & (
            df["hora_fin"].apply(lambda h: isinstance(h, datetime.time) and h >= hora_ref)
        )

        df = df.loc[mask].copy()
        if df.empty:
            return pd.DataFrame()

        columnas = [
            "nombre_completo",
            "nombre_materia",
            "clave",
            "codigo_salon",
            "dia_semana",
            "hora_inicio",
            "hora_fin",
        ]
        columnas = [c for c in columnas if c in df.columns]

        # Elimina duplicados que podrían surgir de la misma clase con múltiples registros por si acaso.
        df = df[columnas].drop_duplicates().sort_values(
            ["nombre_completo", "dia_semana", "hora_inicio"]
        )

        return df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 4️. Clases por docente y día (OLAP: Pivot / Roll-Up)
    # ---------------------------------------------------------
    def clases_por_docente_y_dia(self):
        """
        Crea una tabla de contingencia que contabiliza el número de clases
        que imparte cada docente, desglosado por día de la semana.

        Tipo de operación OLAP:
            - **Pivot**: Reorganiza la dimensión 'día' para que sean las columnas de la tabla.
            - **Roll-Up**: Agrega los datos sumando el total de clases por docente (`Total`).

        Returns:
            DataFrame: Tabla dinámica con `nombre_completo` como índice, días de la semana
                       como columnas, y una columna final `Total` con el conteo semanal.
        """
        # Crea la tabla dinámica: filas=docente, columnas=días, valores=conteo de NRC
        tabla = pd.pivot_table(
            self.cubo,
            values="nrc",
            index="nombre_completo",
            columns="dia_semana",
            aggfunc="count",
            fill_value=0 # Rellena los días sin clase con cero
        )

        orden_dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sábado"]

        # Asegura que todos los días estén presentes, rellenando con 0 si faltan
        for dia in orden_dias:
            if dia not in tabla.columns:
                tabla[dia] = 0

        # Reordena las columnas para que los días aparezcan en el orden correcto
        tabla = tabla[orden_dias]
        
        # Agrega la columna de total (Roll-Up)
        tabla["Total"] = tabla.sum(axis=1)
        return tabla

    # ---------------------------------------------------------
    # 5️. Duración promedio por edificio (OLAP: Roll-Up)
    # ---------------------------------------------------------
    def promedio_duracion_por_edificio(self):
        """
        Calcula el valor promedio de la duración de las clases, agrupado a nivel de edificio.
        Esto permite identificar edificios donde las clases tienden a ser más largas o cortas.

        Tipo de operación OLAP:
            - **Roll-Up**: Agregación de la medida 'duracion_min' sobre el nivel 'edificio'
              de la dimensión 'espacio'.

        Returns:
            DataFrame: Edificios y su duración promedio de clases en minutos, ordenados descendentemente.
        """
        if "duracion_min" not in self.cubo.columns:
            return pd.DataFrame()
            
        # Agrupa por edificio y calcula el promedio de 'duracion_min'
        df = self.cubo.groupby("edificio")["duracion_min"].mean().reset_index()
        
        df.rename(columns={"duracion_min":"duracion_promedio_min"}, inplace=True)
        
        # Ordena el resultado por duración promedio
        return df.sort_values("duracion_promedio_min", ascending=False)


# -------------------------------------------------------------
# Helper para construir el cubo desde la base de datos
# -------------------------------------------------------------
def build_cubo_from_db():
    """
    Función de utilidad para instanciar el objeto `CuboHorario` completo.
    Encapsula la secuencia de creación del motor de DB, carga de tablas
    y la inicialización del cubo con los datos.

    Returns:
        CuboHorario: Una instancia de la clase `CuboHorario` con los datos cargados.
    """
    engine = get_engine()
    h, ddoc, dmat, desp, dtime = cargar_tablas(engine)
    return CuboHorario(h, ddoc, dmat, desp, dtime)