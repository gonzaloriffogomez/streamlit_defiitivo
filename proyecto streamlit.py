import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as components
from io import BytesIO
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Dashboard Planes de Salud", layout="wide")
st.title("Análisis Comparativo de Planes de Salud")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("df_planes_salud_completo.csv")
    
    # Función para limpiar y convertir a numérico
    def convertir_a_numerico(serie):
        # Intenta convertir directamente
        try:
            return pd.to_numeric(serie)
        except:
            # Si falla, limpia caracteres no numéricos
            serie_limpia = serie.str.replace('[^\d.]', '', regex=True)
            return pd.to_numeric(serie_limpia, errors='coerce')
    
    # Convertir columnas numéricas
    columnas_numericas = [
        'precio_final', 'precio_base', 'variacion_precio_base',
        'reajuste_anual', 'copago_fijo', 'tope_anual',
        'cotizantes_vigentes', 'cargas_vigentes'
    ]
    
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = convertir_a_numerico(df[col])
    
    # Convertir fechas
    columnas_fecha = ['fecha_info', 'fecha_inicio_plan', 'fecha_adecuacion', 'fecha_reajuste']
    for col in columnas_fecha:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

df = cargar_datos()
st.write("Vista previa de los datos:")
st.dataframe(df.head())

# ----------------------------
# SIDEBAR: FILTROS INTERACTIVOS
# ----------------------------
st.sidebar.header("Filtros del Panel")

# Filtros basados en las columnas proporcionadas
tipo_plan = st.sidebar.multiselect(
    "Tipo de Plan", 
    df["tipo_plan"].unique() if 'tipo_plan' in df.columns else [],
    default=df["tipo_plan"].unique() if 'tipo_plan' in df.columns else []
)

modalidad_atencion = st.sidebar.multiselect(
    "Modalidad de Atención", 
    df["modalidad_atencion"].unique() if 'modalidad_atencion' in df.columns else [],
    default=df["modalidad_atencion"].unique() if 'modalidad_atencion' in df.columns else []
)

grupo_objetivo = st.sidebar.multiselect(
    "Grupo Objetivo", 
    df["grupo_objetivo"].unique() if 'grupo_objetivo' in df.columns else [],
    default=df["grupo_objetivo"].unique() if 'grupo_objetivo' in df.columns else []
)

comercializacion = st.sidebar.selectbox(
    "Estado de Comercialización", 
    df["comercializacion"].unique() if 'comercializacion' in df.columns else []
)

# Manejo seguro de precios
if 'precio_final' in df.columns:
    precio_min = float(df["precio_final"].min(skipna=True)) if not df["precio_final"].isnull().all() else 0
    precio_max = float(df["precio_final"].max(skipna=True)) if not df["precio_final"].isnull().all() else 1000
    precio = st.sidebar.slider(
        "Rango de Precio Final", 
        precio_min, 
        precio_max, 
        (precio_min, precio_max)
    )

# Manejo seguro de fechas
if 'fecha_inicio_plan' in df.columns and not df['fecha_inicio_plan'].isnull().all():
    min_date = df["fecha_inicio_plan"].min()
    max_date = df["fecha_inicio_plan"].max()
    if pd.notnull(min_date) and pd.notnull(max_date):
        fecha_inicio = st.sidebar.date_input(
            "Fecha de Inicio del Plan", 
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime()
        )
    else:
        st.sidebar.warning("Datos de fecha incompletos")
else:
    st.sidebar.warning("Columna de fecha no disponible")

# ----------------------------
# APLICAR FILTROS AL DATAFRAME
# ----------------------------
df_filtrado = df.copy()

if 'tipo_plan' in df.columns:
    df_filtrado = df_filtrado[df_filtrado["tipo_plan"].isin(tipo_plan)]

if 'modalidad_atencion' in df.columns:
    df_filtrado = df_filtrado[df_filtrado["modalidad_atencion"].isin(modalidad_atencion)]

if 'grupo_objetivo' in df.columns:
    df_filtrado = df_filtrado[df_filtrado["grupo_objetivo"].isin(grupo_objetivo)]

if 'comercializacion' in df.columns:
    df_filtrado = df_filtrado[df_filtrado["comercializacion"] == comercializacion]

if 'precio_final' in df.columns:
    df_filtrado = df_filtrado[df_filtrado["precio_final"].between(precio[0], precio[1])]

if 'fecha_inicio_plan' in df.columns and 'fecha_inicio' in locals() and len(fecha_inicio) == 2:
    start_date = pd.to_datetime(fecha_inicio[0])
    end_date = pd.to_datetime(fecha_inicio[1])
    df_filtrado = df_filtrado[
        (df_filtrado["fecha_inicio_plan"] >= start_date) &
        (df_filtrado["fecha_inicio_plan"] <= end_date)
    ]

# ----------------------------
# MENÚ PRINCIPAL
# ----------------------------
menu = st.selectbox("Selecciona una sección", 
                   ["Elige un Menú", "Análisis General", "Exploración con PyGWalker"])

# ------------------------------------------------------------------------------------
# MENU SECCION ANALISIS GENERAL
# ------------------------------------------------------------------------------------
if menu == "Análisis General":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    planes_count = len(df_filtrado)
    
    # Manejo seguro de KPIs
    try:
        precio_promedio = df_filtrado["precio_final"].mean()
    except:
        precio_promedio = np.nan
        
    try:
        cotizantes_total = df_filtrado["cotizantes_vigentes"].sum()
    except:
        cotizantes_total = np.nan
        
    try:
        variacion_promedio = df_filtrado["variacion_precio_base"].mean()
    except:
        variacion_promedio = np.nan
    
    col1.metric("Total Planes", planes_count)
    col2.metric("Precio Promedio", 
                f"${precio_promedio:,.0f}" if not np.isnan(precio_promedio) else "N/D", 
                delta=None)
    col3.metric("Cotizantes Vigentes", 
                f"{cotizantes_total:,}" if not np.isnan(cotizantes_total) else "N/D")
    col4.metric("Variación Precio Promedio", 
                f"{variacion_promedio:.2f}%" if not np.isnan(variacion_promedio) else "N/D")

    # Visualización con Seaborn
    if 'precio_final' in df_filtrado.columns and 'tipo_plan' in df_filtrado.columns:
        st.subheader("Distribución de Precios por Tipo de Plan")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filtrar valores nulos
        plot_data = df_filtrado.dropna(subset=['precio_final', 'tipo_plan'])
        
        sns.boxplot(data=plot_data, x="tipo_plan", y="precio_final", palette="viridis")
        plt.xticks(rotation=45)
        plt.ylabel("Precio Final")
        plt.xlabel("Tipo de Plan")
        st.pyplot(fig)
    else:
        st.warning("Datos insuficientes para mostrar gráfico de precios")

    # Visualización de evolución temporal
    if 'fecha_info' in df_filtrado.columns and 'precio_final' in df_filtrado.columns:
        st.subheader("Evolución de Precios en el Tiempo")
        
        # Filtrar valores nulos
        temp_data = df_filtrado.dropna(subset=['fecha_info', 'precio_final'])
        
        if not temp_data.empty:
            df_temp = temp_data.groupby('fecha_info')['precio_final'].mean().reset_index()
            fig2 = px.line(
                df_temp,
                x='fecha_info',
                y='precio_final',
                title='Precio Promedio a lo Largo del Tiempo',
                labels={'precio_final': 'Precio Promedio', 'fecha_info': 'Fecha'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No hay datos suficientes después de filtrar valores nulos")
    else:
        st.warning("Datos insuficientes para mostrar evolución de precios")

    # Tabs para organizar contenido
    tab1, tab2, tab3 = st.tabs(["Tabla Comparativa", "Distribución Geográfica", "Estadísticas"])

    with tab1:
        st.dataframe(df_filtrado)

    with tab2:
        if 'region_comercializa' in df_filtrado.columns and 'nombre_plan' in df_filtrado.columns:
            region_counts = df_filtrado['region_comercializa'].value_counts().reset_index()
            region_counts.columns = ['Región', 'Cantidad de Planes']
            fig3 = px.bar(
                region_counts,
                x='Región',
                y='Cantidad de Planes',
                title='Planes por Región',
                color='Cantidad de Planes'
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Datos insuficientes para mostrar distribución geográfica")

    with tab3:
        st.write("Estadísticas de precios:")
        if 'precio_final' in df_filtrado.columns:
            st.dataframe(df_filtrado["precio_final"].describe())
        else:
            st.write("No hay datos de precios disponibles")
        
        st.write("Estadísticas de cotizantes:")
        if 'cotizantes_vigentes' in df_filtrado.columns:
            st.dataframe(df_filtrado["cotizantes_vigentes"].describe())
        else:
            st.write("No hay datos de cotizantes disponibles")

    # Expander con información
    with st.expander("Glosario de Términos"):
        st.markdown("""
        - **Tipo de Plan**: Categoría del plan (Individual, Familiar, Empresarial, etc.)
        - **Modalidad de Atención**: Libre elección, prestador preferente o cerrado
        - **Grupo Objetivo**: Segmento demográfico al que está dirigido el plan
        - **Precio Final**: Costo total del plan después de ajustes
        - **Cotizantes Vigentes**: Número de personas activamente inscritas en el plan
        - **Variación Precio Base**: Cambio porcentual en el precio base del plan
        """)

    # Formulario para exportar datos
    with st.form("formulario_exportar"):
        st.subheader("Exportar Datos Filtrados")
        formato = st.radio("Formato de exportación", ["CSV", "Excel"])
        nombre_archivo = st.text_input("Nombre del archivo", "planes_salud_filtrados")
        exportar = st.form_submit_button("Exportar Datos")
        
        if exportar:
            if formato == "CSV":
                csv = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name=f"{nombre_archivo}.csv",
                    mime="text/csv"
                )
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_filtrado.to_excel(writer, index=False, sheet_name="Planes_Salud")
                st.download_button(
                    label="Descargar Excel",
                    data=output.getvalue(),
                    file_name=f"{nombre_archivo}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ------------------------------------------------------------------------------------
# MENU SECCION ANALISIS PYGWALKER
# ------------------------------------------------------------------------------------
elif menu == "Exploración con PyGWalker":
    st.subheader("🧭 PyGWalker - Exploración Visual")
    tab_pyg1, tab_pyg2 = st.tabs(["⚙️ PyGWalker dinámico", "📁 Cargar JSON de PyGWalker"])

    with tab_pyg1:
        st.subheader("⚙️ Exploración Dinámica con PyGWalker")
        st.info("Explora interactivamente los datos usando la interfaz de PyGWalker")
        generated_html = pyg.to_html(df_filtrado, return_html=True, dark='light')
        components.html(generated_html, height=800, scrolling=True)

    with tab_pyg2:
        st.subheader("📁 Subir archivo JSON de PyGWalker")
        uploaded_file = st.file_uploader("Selecciona un archivo .json exportado desde PyGWalker", type="json")
        
        if uploaded_file is not None:
            try:
                json_content = uploaded_file.read().decode("utf-8")
                generated_html_json = pyg.to_html(df, return_html=True, dark='light', spec=json_content)
                st.subheader("⚙️ Visualización desde JSON")
                components.html(generated_html_json, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

# ----------------------------
# FOOTER
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard de Planes de Salud**")
st.sidebar.markdown("Versión 1.0 · Julio 2025")