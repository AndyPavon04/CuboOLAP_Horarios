# app/main.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from cubo import build_cubo_from_db
import pandas as pd

app = Flask(__name__)
# Configuración de la clave secreta para el manejo de sesiones y mensajes flash
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Cargar el cubo como una función para que se ejecute solo una vez
# Nota: La carga inicial del cubo puede tomar tiempo.
def get_cubo():
    """
    Intenta construir y retornar la instancia de `CuboHorario` cargando los datos
    desde la base de datos. En caso de error (ej. DB inaccesible), retorna un
    objeto simulado (`DummyCubo`) con DataFrames vacíos para prevenir que la
    aplicación Flask colapse.

    Returns:
        CuboHorario | DummyCubo: La instancia del cubo o un sustituto de seguridad.
    """
    try:
        return build_cubo_from_db()
    except Exception as e:
        print(f"ERROR AL CARGAR EL CUBO: {e}")
        # En caso de error, devuelve un DataFrame vacío para que la app no colapse
        class DummyCubo:
            """Clase de emergencia que simula el CuboHorario pero retorna DataFrames vacíos."""
            def __init__(self):
                self.cubo = pd.DataFrame()
            def horario_docente(self, *args): return pd.DataFrame()
            def docentes_por_materia(self, *args): return pd.DataFrame()
            def docentes_en_edificio_hora(self, *args): return pd.DataFrame()
            def clases_por_docente_y_dia(self): return pd.DataFrame()
            def promedio_duracion_por_edificio(self): return pd.DataFrame()
        return DummyCubo()

# Carga la instancia del cubo al iniciar la aplicación (una sola vez)
cubo = get_cubo()

@app.route("/")
def index():
    """Maneja la ruta raíz ('/') y renderiza la plantilla de inicio."""
    return render_template("index.html")

# ------- DOCENTES -------
@app.route("/docentes", methods=["GET", "POST"])
def docentes():
    """
    Maneja la ruta '/docentes'. Permite buscar el horario de un docente
    específico o mostrar la lista completa de horarios si no hay búsqueda.
    Normaliza el orden de los días de la semana para la visualización.
    """
    resultado = None
    nombre_docente = None

    if request.method == "POST":
        # Procesa la solicitud POST para buscar un docente
        nombre_docente = request.form.get("nombre_docente", "").strip()
        if nombre_docente:
            # Llama al método Slice del cubo para obtener el horario del docente
            resultado = cubo.horario_docente(nombre_docente)
        else:
            # Si el campo de búsqueda está vacío, muestra todos los horarios (comportamiento por defecto)
            resultado = cubo.cubo[[
                "nombre_completo", "dia_semana", "hora_inicio", "hora_fin",
                "nombre_materia", "clave", "codigo_salon", "edificio", "aula"
            ]].sort_values(["nombre_completo", "dia_semana", "hora_inicio"])
    else:
        # 🔹 Si es la primera carga (GET), muestra todos los horarios del cubo
        resultado = cubo.cubo[[
            "nombre_completo", "dia_semana", "hora_inicio", "hora_fin",
            "nombre_materia", "clave", "codigo_salon", "edificio", "aula"
        ]].sort_values(["nombre_completo", "dia_semana", "hora_inicio"])

    # 🔹 Agrega el orden correcto de los días de la semana (Lunes a Sábado)
    orden_dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"]
    # Convierte la columna 'dia_semana' a tipo Categórico para ordenar correctamente
    resultado["dia_semana"] = pd.Categorical(resultado["dia_semana"], categories=orden_dias, ordered=True)
    # Ordena nuevamente el resultado asegurando el orden correcto por nombre, día y hora
    resultado = resultado.sort_values(["nombre_completo", "dia_semana", "hora_inicio"])

    # 🔹 Renderiza la plantilla con los resultados
    return render_template(
        "docentes.html",
        resultado=resultado,
        nombre_docente=nombre_docente
    )


# ------- MATERIAS -------
@app.route("/materias", methods=["GET", "POST"])
def materias():
    """
    Maneja la ruta '/materias'. Permite buscar qué docentes imparten una materia
    específica por su nombre o clave. Proporciona una lista de materias para
    facilitar la selección.
    """
    df = None
    query = ""

    # 🔹 Obtener todas las materias únicas para la lista desplegable del formulario
    materias_lista = (
        cubo.cubo[["clave", "nombre_materia"]]
        .drop_duplicates()
        .sort_values("nombre_materia")
        .to_dict("records")
    )

    if request.method == "POST":
        # Procesa la solicitud POST para buscar por materia o clave
        query = request.form.get("materia", "").strip()
        # Llama al método Dice del cubo para obtener los docentes por materia
        df = cubo.docentes_por_materia(query)
        if df is None or df.empty:
            # Muestra un mensaje flash si no se encuentran resultados
            flash(f"No hay docentes para la materia/clave: {query}", "warning")
            return redirect(url_for("materias")) # Redirige para limpiar el POST

    # Renderiza la plantilla, pasando la tabla de resultados (si existe) y la lista de materias
    return render_template("materias.html", tabla=df, query=query, materias_lista=materias_lista)


# ------- EDIFICIOS -------
@app.route("/edificios", methods=["GET", "POST"])
def edificios():
    """
    Maneja la ruta '/edificios'. Permite buscar qué docentes están en un edificio
    específico a una hora determinada. Proporciona listas de edificios y horas
    disponibles para la selección en el formulario.
    """
    df = None
    edificio = ""
    hora = ""

    # Obtiene la lista única de edificios para el formulario de selección
    edificios_lista = (
        cubo.cubo["edificio"].dropna().drop_duplicates().sort_values().tolist()
    )

    # 👉 Obtiene la lista única de horas de inicio y las formatea a "HH:MM" para el selector
    horas_lista = sorted({
        (h.strftime("%H:%M") if hasattr(h, "strftime") else str(h)[:5])
        for h in cubo.cubo["hora_inicio"].dropna().tolist()
    })

    if request.method == "POST":
        # Procesa la solicitud POST para buscar por edificio y hora
        edificio = request.form.get("edificio", "").strip()
        hora = request.form.get("hora", "").strip()   # ej. "09:00"

        if edificio and hora:
            # Llama al método Dice del cubo para obtener los docentes activos
            df = cubo.docentes_en_edificio_hora(edificio, hora)
            if df is None or df.empty:
                # Muestra un mensaje flash si no se encuentran resultados
                flash(f"No hay docentes en {edificio} a las {hora}.", "warning")
                return redirect(url_for("edificios"))

    # Renderiza la plantilla, pasando los datos de selección y los resultados (si existen)
    return render_template(
        "edificios.html",
        tabla=df,
        edificio=edificio,
        hora=hora,
        edificios_lista=edificios_lista,
        horas_lista=horas_lista
    )


if __name__ == "__main__":
    # Comando para ejecutar la aplicación Flask en modo de desarrollo:
    # FLASK_APP=main.py flask run (desde el directorio app/)
    app.run(debug=True)