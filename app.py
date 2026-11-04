# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import gc

st.set_page_config(
    page_title="Mapa de Dose 3D - XII Jornada Acadêmica de Física Médica",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
    <style>
        hr {
            border: 1px solid #b30000;
        }
        .stSlider label, .stRadio label {
            color: #b30000;
            font-weight: bold;
        }
        div[data-baseweb="slider"] div[class*="thumb-value"],
        div[data-baseweb="slider"] output,
        span[data-baseweb="tag"],
        .stSlider [data-testid="stMarkdownContainer"] p {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Sobre o Trabalho")
    st.info(
        "**XII Jornada Acadêmica de Física Médica**\n\n"
        "Visualizador interativo de mapas de dose 3D desenvolvido para a apresentação do trabalho "
        "\"Desenvolvimento e caracterização de um fantoma impresso em 3D para testes de controle de qualidade em tomografia computadorizada\"."
    )
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #b30000; font-size: 13px; font-weight: bold;'>Obrigada pela atenção!</p>", unsafe_allow_html=True)

st.title("Visualizador de Mapa de Dose 3D")
st.markdown("Utilize o controle deslizante abaixo para navegar pelas fatias de profundidade do fantoma.")
st.markdown("<hr>", unsafe_allow_html=True)

ARQUIVO_FRENTE = "3d-dose-map.dat"
ID_FRENTE = "130OQs5XQMsi6_D9gFc1cRckRItDtuXXm"

ARQUIVO_TRAS = "3d-dose-map1.dat"
ID_TRAS = "1TqjHZZxMKNvuM7Ss6XeQ0hCM4hLMzpXv"

def baixar_arquivo(arquivo, id_drive):
    if not os.path.exists(arquivo) or os.path.getsize(arquivo) < 1000000:
        url = f"https://drive.usercontent.google.com/download?id={id_drive}&export=download&confirm=t"
        resposta = requests.get(url, stream=True)
        with open(arquivo, "wb") as f:
            for bloco in resposta.iter_content(chunk_size=32768):
                if bloco:
                    f.write(bloco)

@st.cache_resource(show_spinner="A carregar e processar os mapas de dose... Por favor, aguarde.")
def carregar_dados():
    baixar_arquivo(ARQUIVO_FRENTE, ID_FRENTE)
    baixar_arquivo(ARQUIVO_TRAS, ID_TRAS)

    colunas = ['X', 'Y', 'Z', 'Dose', 'Incerteza', 'IX', 'IY', 'IZ']
    tipos = {'X': np.float32, 'Y': np.float32, 'Z': np.float32, 'Dose': np.float32}

    df = pd.read_csv(
        ARQUIVO_FRENTE,
        sep=r'\s+',
        comment='#',
        names=colunas,
        usecols=['X', 'Y', 'Z', 'Dose', 'IX', 'IY', 'IZ'],
        dtype=tipos,
        on_bad_lines='skip'
    ).dropna()

    ix = df['IX'].astype(int).values - 1
    iy = df['IY'].astype(int).values - 1
    iz = df['IZ'].astype(int).values - 1

    nbx = int(ix.max()) + 1
    nby = int(iy.max()) + 1
    nbz = int(iz.max()) + 1

    dose_3d = np.zeros((nbx, nby, nbz), dtype=np.float32)
    dose_3d[ix, iy, iz] = df['Dose'].values

    x_coords = np.sort(df['X'].unique())
    y_coords = np.sort(df['Y'].unique())

    limites_xy = [
        float(x_coords.min()),
        float(x_coords.max()),
        float(y_coords.min()),
        float(y_coords.max())
    ]

    del df, ix, iy, iz
    gc.collect()

    df_tras = pd.read_csv(
        ARQUIVO_TRAS,
        sep=r'\s+',
        comment='#',
        names=colunas,
        usecols=['Dose', 'IX', 'IY', 'IZ'],
        dtype={'Dose': np.float32},
        on_bad_lines='skip'
    ).dropna()

    ix_t = df_tras['IX'].astype(int).values - 1
    iy_t = df_tras['IY'].astype(int).values - 1
    iz_t = df_tras['IZ'].astype(int).values - 1

    dose_3d[ix_t, iy_t, iz_t] += df_tras['Dose'].values

    del df_tras, ix_t, iy_t, iz_t
    gc.collect()

    doses_validas = dose_3d[dose_3d > 1e-10]
    dose_maxima_referencia = np.percentile(doses_validas, 99.5) if len(doses_validas) > 0 else dose_3d.max()
    dose_normalizada = (dose_3d / dose_maxima_referencia) * 100.0

    return dose_normalizada, limites_xy, nbz

dose_normalizada, limites_xy, nbz = carregar_dados()

opcao_cor = st.radio(
    "Modo de visualização:",
    options=["Escala de Cinza", "Escala Colorida"],
    index=0,
    horizontal=True
)

cmap_escolhido = 'gray_r' if opcao_cor == "Escala de Cinza" else 'jet'

fatia_z = st.slider(
    "Fatia Z:",
    min_value=0,
    max_value=nbz - 1,
    value=nbz // 2,
    step=1,
    label_visibility="visible"
)

corte_xy = dose_normalizada[:, :, fatia_z].T

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(
    corte_xy,
    origin='lower',
    extent=limites_xy,
    cmap=cmap_escolhido,
    vmin=0,
    vmax=100,
    interpolation='nearest',
    aspect='equal'
)

fig.colorbar(im, ax=ax, label='Dose Relativa (%)')
ax.axis('off')
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.markdown("<hr>", unsafe_allow_html=True)
