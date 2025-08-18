import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc
import colorsys  # Para conversiones de espacio de color
import warnings
warnings.filterwarnings('ignore')

# Inicializar la app con mejor configuración
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.BOOTSTRAP, 
                              'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'], 
          suppress_callback_exceptions=True, 
          prevent_initial_callbacks=False)

# Configuración para deployment
server = app.server

# Función para generar colores con Chroma y Luminance uniformes
def generate_uniform_colors(n_colors, chroma=0.7, luminance=0.6):
    """
    Genera n colores con Chroma (saturación) y Luminance (brillo) uniformes,
    variando únicamente el Hue (tono).
    
    Parameters:
    - n_colors: número de colores a generar
    - chroma: saturación (0-1), donde 1 es máxima saturación
    - luminance: brillo (0-1), donde 1 es máximo brillo
    
    Returns:
    - Lista de colores en formato hexadecimal
    """
    colors = []
    
    for i in range(n_colors):
        # Muestreo uniforme del Hue (0 a 360 grados)
        hue = i / n_colors
        
        # Convertir HSV a RGB (colorsys usa HSV donde S=chroma, V=luminance)
        r, g, b = colorsys.hsv_to_rgb(hue, chroma, luminance)
        
        # Convertir a valores 0-255 y luego a hexadecimal
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(r * 255), 
            int(g * 255), 
            int(b * 255)
        )
        colors.append(hex_color)
    
    return colors

def create_income_colors_ordinal():
    """
    Escala divergente PuOr (Purple-Orange) con punto neutro beige
    Usa naranja para Low, beige neutro para Middle, y púrpura para High
    El beige es verdaderamente neutral y no interfiere con la progresión
    """
    return {
        "Low income": "#e08214",     # Naranja medio
        "Middle income": "#d4c4a8",  # Beige neutro (perfecto punto medio)
        "High income": "#542788"     # Púrpura oscuro
    }

def create_income_colors(chroma=0.8, luminance=0.65):
    """
    Genera colores específicos para los 3 niveles de ingresos
    con Chroma y Luminance uniformes, distribuyendo el Hue uniformemente.
    DEPRECADO: Mantenido para compatibilidad, usa create_income_colors_ordinal()
    """
    colors = generate_uniform_colors(3, chroma, luminance)
    
    return {
        "Low income": colors[0],     # Primer color del espectro (rojo)
        "Middle income": colors[1],  # Segundo color del espectro (verde)
        "High income": colors[2]     # Tercer color del espectro (azul)
    }

# Cargar los datos
df_vaccine = pd.read_csv("global-vaccination-coverage.csv")

# Para cada fila con NaN, intenta rellenar con el valor de la siguiente o anterior fila del mismo Entity
df_vaccine_sorted = df_vaccine.sort_values(['Entity', 'Year']).reset_index(drop=True)
df_vaccine_filled = df_vaccine_sorted.copy()
cols_to_fill = [col for col in df_vaccine_filled.columns if col not in ['Entity', 'Code', 'Year']]
df_vaccine_filled[cols_to_fill] = (
    df_vaccine_filled.groupby('Entity')[cols_to_fill]
    .transform(lambda group: group.ffill().bfill())
)

# Preparar datos para el bubble chart - SOLO países (excluir regiones)
df_pol3 = df_vaccine_filled[['Entity', 'Code', 'Year', 'Pol3 (% of one-year-olds immunized)']].copy()
df_pol3 = df_pol3.dropna(subset=['Pol3 (% of one-year-olds immunized)'])
df_pol3 = df_pol3[
    df_pol3['Code'].notna() & 
    (df_pol3['Code'] != '') & 
    (df_pol3['Code'].str.len() <= 3)
].copy()

primer_anio = int(df_pol3['Year'].min())
top10_menor = (
    df_pol3[df_pol3['Year'] == primer_anio]
    .sort_values('Pol3 (% of one-year-olds immunized)')
    .head(10)['Entity']
    .tolist()
)
entity_order = (
    df_pol3[df_pol3['Year'] == primer_anio]
    .set_index('Entity')
    .loc[top10_menor]['Pol3 (% of one-year-olds immunized)']
    .sort_values(ascending=True)
    .index.tolist()
)

print(f"Orden de países (menor a mayor cobertura): {entity_order}")
primer_anio_data = df_pol3[df_pol3['Year'] == primer_anio].set_index('Entity').loc[top10_menor]
for entity in entity_order:
    valor = primer_anio_data.loc[entity, 'Pol3 (% of one-year-olds immunized)']
    print(f"{entity}: {valor:.1f}%")

df_pol3_top10 = df_pol3[df_pol3['Entity'].isin(top10_menor)].copy()
df_pol3_top10['size'] = df_pol3_top10['Pol3 (% of one-year-olds immunized)']

# Preparar datos para el análisis por región
regiones_interes = ['Africa', 'Americas', 'Eastern Mediterranean', 'Europe', 'Micronesia', 'South-East Asia', 'Western Pacific']

# --- CORRECCIÓN: Forzar que 'Europe' esté presente y bien escrita en los datos de regiones ---
# A veces, por problemas de espacios o codificación, 'Europe' puede no coincidir exactamente.
# Normalizamos los nombres de las entidades para asegurar coincidencia exacta.
def normalize_region_name(name):
    if isinstance(name, str):
        return name.strip().replace('\u200b', '').replace('\xa0', ' ')
    return name

df_vaccine['Entity'] = df_vaccine['Entity'].apply(normalize_region_name)
regiones_interes_normalized = [normalize_region_name(r) for r in regiones_interes]

df_regions = df_vaccine[
    df_vaccine['Entity'].notna() &
    df_vaccine['Entity'].ne('') &
    (df_vaccine['Code'].isna() | (df_vaccine['Code'] == '')) &
    (df_vaccine['Entity'].isin(regiones_interes_normalized))
].copy()

# --- FIN CORRECCIÓN ---

# Preparar datos para el análisis por nivel de ingresos
income_entities = df_vaccine[df_vaccine['Entity'].str.contains('income', case=False, na=False)]
immun_cols = [col for col in income_entities.columns if '% of one-year-olds immunized' in col]

def short_label(col):
    idx = col.find('(')
    return col[:idx].strip() if idx != -1 else col.strip()

immun_labels = [short_label(col) for col in immun_cols]
label_to_col = dict(zip(immun_labels, immun_cols))

def income_group(entity):
    entity = entity.lower()
    if "middle" in entity:
        return "Middle income"
    elif "low" in entity:
        return "Low income"
    elif "high" in entity:
        return "High income"
    else:
        return entity

income_entities = income_entities.copy()
income_entities['IncomeGroup'] = income_entities['Entity'].apply(income_group)
income_grouped = income_entities.groupby(['Year', 'IncomeGroup'])[immun_cols].mean().reset_index()

# Aplicar la escala de colores ordinal para niveles de ingresos
income_colors = create_income_colors_ordinal()

# Imprimir los colores generados para verificación
print("Colores PuOr generados para niveles de ingresos:")
for level, color in income_colors.items():
    print(f"{level}: {color}")
print("Escala utilizada: PuOr divergente (naranja → beige neutro → púrpura = bajo → medio → alto ingreso)")

# Layout de la aplicación
app.layout = html.Div([
    html.Div([
        html.H1("Dashboard de Cobertura de Vacunación Global", 
                style={'textAlign': 'center', 'marginBottom': '10px', 
                       'fontFamily': 'Inter, sans-serif', 'fontWeight': '600', 'color': '#2c3e50'}),
        html.P("Análisis interactivo de la evolución de la inmunización a nivel mundial", 
               style={'textAlign': 'center', 'marginBottom': '30px', 
                      'fontFamily': 'Inter, sans-serif', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '30px 20px', 'marginBottom': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label="Evolución de aplicación de Pol3 por País", children=[
            html.Div([
                html.H3("💉 Evolución del % de inmunizados de Pol3 por país",
                        style={'textAlign': 'center', 'marginTop': '20px', 'fontFamily': 'Inter, sans-serif'}),
                html.P(""¿Cómo fue la evolución de la inmunización de los 10 países con menor cobertura en 1980?",
                       style={'textAlign': 'center', 'marginBottom': '20px', 'fontFamily': 'Inter, sans-serif', 'color': '#666'}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Año:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Slider(
                    id='year-slider',
                    min=df_pol3_top10['Year'].min(),
                    max=df_pol3_top10['Year'].max(),
                    step=1,
                    value=df_pol3_top10['Year'].min(),
                    marks={str(year): str(year) for year in range(
                        int(df_pol3_top10['Year'].min()), 
                        int(df_pol3_top10['Year'].max())+1, 
                        5
                    )},
                    tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=8),
                        dbc.Col([
                            html.Label("Animación:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                            html.Div([
                                dbc.Button("▶️ Play", id="play-button", color="primary", className="me-2"),
                                dbc.Button("⏸️ Pause", id="pause-button", color="secondary", className="me-2"),
                                dbc.Button("⏹️ Reset", id="reset-button", color="outline-secondary")
                            ])
                        ], width=4)
                    ])
                ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
                dcc.Graph(id='bubble-chart', style={'height': '600px'}),
                dcc.Interval(
                    id='animation-interval',
                    interval=600,
                    n_intervals=0,
                    disabled=True,
                    max_intervals=-1
                )
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="🌍 Análisis por Región", children=[
            html.Div([
                html.H3("¿Cómo fue la evolución de la inmunización por región?",
                        style={'textAlign': 'center', 'marginTop': '20px', 'fontFamily': 'Inter, sans-serif'}),
                html.P("La región destacada aparece en color, el resto en gris.",
                       style={'textAlign': 'center', 'color': '#666', 'fontFamily': 'Inter, sans-serif', 'marginBottom': '30px'}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Vacuna:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='region-vaccine-dropdown',
                            options=[
                                    {'label': short_label(col), 'value': col} for col in df_vaccine.columns 
                                if '% of one-year-olds immunized' in col
                            ],
                            value='Pol3 (% of one-year-olds immunized)',
                            clearable=False
                        )
                        ], width=6),
                        dbc.Col([
                            html.Label("Destacar Región:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                        dcc.Dropdown(
                                id='highlight-region-dropdown',
                                options=[{'label': region, 'value': region} for region in regiones_interes],
                                value='South-East Asia',
                            clearable=False
                        )
                        ], width=6)
                    ])
                ], style={'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
                dcc.Graph(id='region-time-chart', style={'height': '600px'})
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="💸 Análisis por Nivel de Ingresos", children=[
            html.Div([
                html.H3("Cobertura de Vacunación por Nivel de Ingresos",
                        style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div([
                    html.Label("Seleccionar Año:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='income-year-dropdown',
                        options=[{'label': str(year), 'value': year} for year in sorted(income_grouped['Year'].unique())],
                        value=2016,
                        clearable=False,
                        style={'marginBottom': '20px'}
                    ),
                    html.Label("Seleccionar Vacunas:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='income-vaccine-dropdown',
                        options=[{'label': label, 'value': label} for label in immun_labels],
                        value=immun_labels[:3],
                        multi=True,
                        placeholder="Selecciona vacunas...",
                        style={'marginBottom': '20px'}
                    ),
                    html.Div([
                        dbc.Button('Seleccionar Todas', id='select-all', color='success', className='me-2'),
                        dbc.Button('Deseleccionar Todas', id='deselect-all', color='danger')
                    ], style={'marginBottom': '20px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                html.Div([
                    dcc.Graph(id='income-chart')
                ], style={'width': '70%', 'display': 'inline-block'})
            ], style={'padding': '20px'})
        ])
    ])
])

@app.callback(
    [Output('animation-interval', 'disabled'),
     Output('year-slider', 'value')],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('animation-interval', 'n_intervals')],
    prevent_initial_call=True
)
def control_animation(play_clicks, pause_clicks, reset_clicks, n_intervals):
    triggered_id = ctx.triggered_id if ctx.triggered_id else 'No clicks yet'
    years = sorted(df_pol3_top10['Year'].unique())
    min_year = int(min(years))
    max_year = int(max(years))
    if triggered_id == 'play-button':
        return False, min_year
    elif triggered_id == 'pause-button':
        current_value = ctx.states.get('year-slider.value', min_year)
        return True, int(current_value) if current_value else min_year
    elif triggered_id == 'reset-button':
        return True, min_year
    elif triggered_id == 'animation-interval':
        current_year_index = n_intervals % len(years)
        next_year = int(years[current_year_index])
        if current_year_index == len(years) - 1:
            return True, next_year
        else:
            return False, next_year
    return True, min_year

_bubble_chart_cache = {}

PASTEL_COLORS = [
    '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFD700', '#FFDFBA',
    '#E0BBE4', '#FFC9DE', '#B5EAD7', '#C7CEEA', '#FFDAC1'
]

def create_base_bubble_chart():
    global _bubble_chart_cache
    if 'base_fig' in _bubble_chart_cache:
        return _bubble_chart_cache['base_fig'], _bubble_chart_cache['entity_colors']
    primer_anio = int(df_pol3_top10['Year'].min())
    entity_color_map = {}
    for i, entity in enumerate(entity_order):
        entity_color_map[entity] = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    fig_base = go.Figure()
    _bubble_chart_cache['base_fig'] = fig_base
    _bubble_chart_cache['entity_colors'] = entity_color_map
    return fig_base, entity_color_map

@app.callback(
    Output('bubble-chart', 'figure'),
    [Input('year-slider', 'value')],
    prevent_initial_call=False
)
def update_bubble_chart(selected_year):
    try:
        primer_anio = int(df_pol3_top10['Year'].min())
        _, entity_color_map = create_base_bubble_chart()
        df_year = df_pol3_top10[df_pol3_top10['Year'] == selected_year]
        df_initial = df_pol3_top10[df_pol3_top10['Year'] == primer_anio]
        bubble_x, bubble_y, bubble_colors, bubble_sizes = [], [], [], []
        diamond_x, diamond_y, diamond_colors, diamond_sizes = [], [], [], []
        entity_to_color = {}
        for i, entity in enumerate(entity_order):
            entity_to_color[entity] = entity_color_map.get(entity, PASTEL_COLORS[i % len(PASTEL_COLORS)])
        for entity in entity_order:
            color = entity_to_color[entity]
            entity_data = df_year[df_year['Entity'] == entity]
            if not entity_data.empty:
                bubble_x.append(entity)
                pol3_value = entity_data['Pol3 (% of one-year-olds immunized)'].values[0]
                bubble_y.append(pol3_value)
                bubble_colors.append(color)
                bubble_sizes.append(15 + (pol3_value / 100) * 15)
            entity_initial = df_initial[df_initial['Entity'] == entity]
            if not entity_initial.empty:
                diamond_x.append(entity)
                initial_pol3_value = entity_initial['Pol3 (% of one-year-olds immunized)'].values[0]
                diamond_y.append(initial_pol3_value)
                diamond_colors.append(color)
                diamond_sizes.append(18 + (initial_pol3_value / 100) * 15)
        fig = go.Figure()
        for i, entity in enumerate(entity_order):
            if entity in [bubble_x[j] for j in range(len(bubble_x))]:
                bubble_idx = bubble_x.index(entity) if entity in bubble_x else None
                diamond_idx = diamond_x.index(entity) if entity in diamond_x else None
                color = entity_to_color[entity]
                if bubble_idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[bubble_x[bubble_idx]],
                        y=[bubble_y[bubble_idx]],
                        mode='markers',
                        marker=dict(
                            size=bubble_sizes[bubble_idx],
                            color=color,
                            sizemode='area',
                            sizemin=8,
                            line=dict(width=1, color='white')
                        ),
                        name=entity,
                        showlegend=False,
                        hovertemplate=f"<b>País:</b> {entity}<br><b>Año:</b> {selected_year}<br><b>Pol3 (%):</b> {bubble_y[bubble_idx]:.2f}<extra></extra>"
                    ))
                if diamond_idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[diamond_x[diamond_idx]],
                        y=[diamond_y[diamond_idx]],
                        mode='markers',
                        marker=dict(
                            size=diamond_sizes[diamond_idx],
                            color=color,
                            symbol='diamond',
                            line=dict(width=2, color="rgba(0,0,0,0.8)"),
                            sizemode='area'
                        ),
                        name=f"{entity}_diamond",
                        showlegend=False,
                        hovertemplate=f"<b>País:</b> {entity}<br><b>Año inicial:</b> {primer_anio}<br><b>Pol3 (%):</b> {diamond_y[diamond_idx]:.2f}<extra></extra>"
                    ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='#A8A8A8'),
            name='Valor actual (círculo)',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, symbol='diamond', color='#A8A8A8', 
                       line=dict(width=2, color='#A8A8A8')),
            name=f'Valor inicial ({primer_anio}) (diamante)',
            showlegend=True
        ))
        fig.update_layout(
            title=dict(
                text=f"Evolución del % de inmunizados de Pol3 por país",
                font=dict(family="Inter, sans-serif", size=20, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="País (ordenado de menor a mayor cobertura inicial)", font=dict(size=14)),
                categoryorder='array',
                categoryarray=entity_order,
                showgrid=False,
                showline=True,
                linecolor="black",
                mirror=False
            ),
            yaxis=dict(
                title=dict(text="% de inmunizados Pol3", font=dict(size=14)),
                range=[0, 105],
                showgrid=False,
                showline=True,
                linecolor="black",
                mirror=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=dict(text="Tipo", font=dict(color="#757575")),
                font=dict(color="#757575")
            ),
            # Máxima optimización de rendimiento
            uirevision=f"year-{selected_year}",
            transition=dict(duration=300, easing="cubic-in-out"),
            showlegend=True
        )
    
        return fig
        
    except Exception as e:
        # En caso de error, devolver un gráfico vacío con mensaje de error
        fig = go.Figure()
        fig.update_layout(
            title=f"Error al cargar datos: {str(e)}",
            height=600,
            font=dict(family="Inter, sans-serif")
        )
        return fig

# Callback para el gráfico de regiones con destacado (solo spaguetti)
# Callback para el gráfico de regiones con destacado (solo spaguetti)
@app.callback(
    Output('region-time-chart', 'figure'),
    [Input('region-vaccine-dropdown', 'value'),
     Input('highlight-region-dropdown', 'value')],
    prevent_initial_call=False
)
def update_region_charts(selected_vaccine, highlight_region):
    try:
        # Crear gráfico de líneas tipo spaguetti con destacado
        df_vaccine_filtered = df_regions.dropna(subset=[selected_vaccine]).copy()
        
        # Crear figura base
        fig2 = go.Figure()
        
        # Función para ajustar posiciones de etiquetas (mejorada)
        def adjust_positions(regions_data, min_distance=3):
            sorted_regions = sorted(regions_data.items(), key=lambda x: x[1])
            adjusted = {}
            
            for i, (region, y_pos) in enumerate(sorted_regions):
                if i == 0:
                    adjusted[region] = y_pos
                else:
                    prev_y = list(adjusted.values())[-1]
                    if y_pos - prev_y < min_distance:
                        adjusted[region] = prev_y + min_distance
                    else:
                        adjusted[region] = y_pos
            return adjusted
        
        # Obtener posiciones finales para ajustar etiquetas
        final_positions = {}
        regions_with_data = []
        
        # Verificar todas las regiones disponibles en los datos filtrados
        available_regions = df_vaccine_filtered['Entity'].unique()
        print(f"Regiones disponibles en datos: {available_regions}")
        
        # CORRECCIÓN: Verificar datos para todas las regiones incluyendo Europe
        for region in regiones_interes:
            region_data = df_vaccine_filtered[df_vaccine_filtered['Entity'] == region]
            print(f"Verificando {region}: {len(region_data)} filas")
            
            if not region_data.empty and len(region_data) > 0:
                # Buscar el último valor válido (no NaN)
                valid_values = region_data[selected_vaccine].dropna()
                if len(valid_values) > 0:
                    final_val = valid_values.iloc[-1]
                    final_positions[region] = final_val
                    regions_with_data.append(region)
                    print(f"{region}: {final_val:.1f}% (incluida)")
                else:
                    print(f"{region}: todos los valores son NaN (excluida)")
            else:
                print(f"{region}: sin datos (excluida)")
        
        print(f"Regiones con datos para {short_label(selected_vaccine)}: {regions_with_data}")
        
        # Ajustar posiciones de etiquetas
        adjusted_positions = adjust_positions(final_positions, min_distance=3)
        
        # Dibujar líneas solo para regiones con datos
        for region in regions_with_data:
            region_data = df_vaccine_filtered[df_vaccine_filtered['Entity'] == region]
            
            # Estilos condicionales
            if region == highlight_region:
                color = '#ff7f0e'
                width = 3
                opacity = 1.0
                zorder = 3
            else:
                color = 'lightgray'
                width = 2
                opacity = 0.6
                zorder = 2
            
            # Agregar línea
            fig2.add_trace(go.Scatter(
                x=region_data['Year'],
                y=region_data[selected_vaccine],
                mode='lines',
                name=region,
                line=dict(color=color, width=width),
                opacity=opacity,
                showlegend=False,
                hovertemplate=f"<b>Región:</b> {region}<br><b>Año:</b> %{{x}}<br><b>{short_label(selected_vaccine)} (%):</b> %{{y:.2f}}<extra></extra>"
            ))
            
            # CORRECCIÓN: Mejorar la adición de etiquetas
            if region in adjusted_positions:
                x_final = region_data['Year'].max()
                y_original = region_data[region_data['Year'] == x_final][selected_vaccine].iloc[0]
                y_adjusted = adjusted_positions[region]
                x_text = x_final + 0.8  # Más espacio para las etiquetas
                
                # Línea conectora si la posición se ajustó significativamente
                if abs(y_adjusted - y_original) > 3:
                    fig2.add_trace(go.Scatter(
                        x=[x_final, x_text - 0.3],
                        y=[y_original, y_adjusted],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Texto de la etiqueta - CORRECCIÓN: Asegurar que todas las regiones tengan etiqueta
                text_color = color if region == highlight_region else '#666666'
                text_size = 13 if region == highlight_region else 11
                text_weight = 'bold' if region == highlight_region else 'normal'
                
                fig2.add_annotation(
                    x=x_text,
                    y=y_adjusted,
                    text=region,
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=text_size,
                        family="Inter, sans-serif"
                    ),
                    xanchor='left',
                    yanchor='middle'
                )
        
        # CORRECCIÓN: Verificar específicamente Europe y forzar su inclusión si tiene datos
        europe_debug = df_regions[df_regions['Entity'] == 'Europe']
        if not europe_debug.empty:
            europe_vaccine_data = europe_debug.dropna(subset=[selected_vaccine])
            if not europe_vaccine_data.empty:
                print(f"Europe tiene {len(europe_vaccine_data)} registros válidos para {selected_vaccine}")
                if 'Europe' not in regions_with_data:
                    print("ADVERTENCIA: Europe tiene datos pero no se incluyó en regions_with_data")
            else:
                print(f"Europe no tiene datos válidos para {selected_vaccine}")
        else:
            print("Europe no se encuentra en df_regions")
        
        # Configurar layout del gráfico
        x_min = df_vaccine_filtered['Year'].min()
        x_max = df_vaccine_filtered['Year'].max()
        
        fig2.update_layout(
            title=dict(
                text=f"Evolución temporal de la cobertura de {short_label(selected_vaccine)} por región",
                font=dict(family="Inter, sans-serif", size=18, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text='Año', font=dict(size=14)),
                range=[x_min, x_max + 5],  # Más espacio para etiquetas
                showgrid=False,
                showline=True,
                linecolor="black",
                mirror=False
            ),
            yaxis=dict(
                title=dict(text='Cobertura (%)', font=dict(size=14)),
                range=[0, 100],
                showgrid=False,
                showline=True,
                linecolor="black",
                mirror=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
            height=500,
            margin=dict(r=120)  # Más espacio para las etiquetas
        )
    
        return fig2
        
    except Exception as e:
        # En caso de error, devolver gráfico vacío con mensaje de error
        fig = go.Figure()
        fig.update_layout(title=f"Error al cargar datos: {str(e)}", height=600)
        return fig
# Callback para los botones de seleccionar/deseleccionar todas las vacunas
@app.callback(
    Output('income-vaccine-dropdown', 'value'),
    [Input('select-all', 'n_clicks'),
     Input('deselect-all', 'n_clicks')],
    prevent_initial_call=True
)
def update_dropdown(select_all_clicks, deselect_all_clicks):
    triggered_id = ctx.triggered_id if ctx.triggered_id else 'No clicks yet'
    
    if triggered_id == 'select-all':
        return immun_labels
    elif triggered_id == 'deselect-all':
        return []
    
    return immun_labels[:3]

# Callback para el gráfico de ingresos
@app.callback(
    Output('income-chart', 'figure'),
    [Input('income-vaccine-dropdown', 'value'),
     Input('income-year-dropdown', 'value')],
    prevent_initial_call=False
)
def update_income_chart(selected_vaccines, selected_year):
    try:
        if not selected_vaccines:
            # Si no hay vacunas seleccionadas, mostrar gráfico vacío
            fig = go.Figure()
            fig.update_layout(
                title=f"Selecciona al menos una vacuna para visualizar los datos ({selected_year})",
                template="plotly_white",
                xaxis_title="Vacuna",
                yaxis_title="% inmunizados",
                font=dict(family="Inter, sans-serif"),
                height=600,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            return fig
        
        data = []
        df_year = income_grouped[income_grouped['Year'] == selected_year]
        
        for group in ["High income", "Middle income", "Low income"]:
            x_vals = []
            y_vals = []
            
            for vaccine_label in selected_vaccines:
                if vaccine_label in label_to_col:
                    vaccine_col = label_to_col[vaccine_label]
                    group_data = df_year[df_year['IncomeGroup'] == group]
                    if not group_data.empty:
                        val = group_data[vaccine_col].values[0]
                        x_vals.append(vaccine_label)
                        y_vals.append(val)
            
            if x_vals:  # Solo agregar si hay datos
                data.append(go.Bar(
                    x=x_vals,
                    y=y_vals,
                    name=group,
                    marker_color=income_colors[group]
                ))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            barmode="group",
            template="plotly_white",
            title=dict(
                text=f"Porcentaje de inmunizados por nivel de ingresos ({selected_year})",
                font=dict(family="Inter, sans-serif", size=18, color="#2c3e50")
            ),
            xaxis_title="Vacuna",
            yaxis_title="% inmunizados",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(family="Inter, sans-serif"),
            height=600,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        return fig
        
    except Exception as e:
        # En caso de error, devolver un gráfico vacío con mensaje de error
        fig = go.Figure()
        fig.update_layout(
            title=f"Error al cargar datos: {str(e)}",
            height=600,
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
