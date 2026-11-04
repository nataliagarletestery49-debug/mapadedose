# -*- coding: utf-8 -*-
import os
import gc
import threading
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Evita conflito de interface gráfica entre múltiplos usuários
import matplotlib.pyplot as plt

# Trava de segurança para várias pessoas gerarem gráficos simultaneamente sem travar
_plot_lock = threading.Lock()

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

ARQUIVO_CACHE_BINARIO = "dose_normalizada_cache.npz"

def baixar_arquivo(arquivo, id_drive):
    if not os.path.exists(arquivo) or os.path.getsize(arquivo) < 1000000:
        url = f"https://drive.usercontent.google.com/download?id={id_drive}&export=download&confirm=t"
        with requests.Session() as sessao:
            resposta = sessao.get(url, stream=True, timeout=60)
            resposta.raise_for_status()
            with open(arquivo, "wb") as f:
                for bloco in resposta.iter_content(chunk_size=131072):
                    if bloco:
                        f.write(bloco)

@st.cache_resource(show_spinner="A carregar e processar os mapas de dose... Por favor, aguarde.")
def carregar_dados():
    # 1. Se já processou uma vez no servidor, carrega o binário pronto em ~0.02s
    if os.path.exists(ARQUIVO_CACHE_BINARIO):
        dados = np.load(ARQUIVO_CACHE_BINARIO)
        dose_normalizada = dados["dose_normalizada"]
        limites_xy = dados["limites_xy"].tolist()
        nbz = int(dados["nbz"])
        return dose_normalizada, limites_xy, nbz

    # 2. Baixa e processa os arquivos .dat mantendo a sua leitura original
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

    ix = df['IX'].astype(np.int32).values - 1
    iy = df['IY'].astype(np.int32).values - 1
    iz = df['IZ'].astype(np.int32).values - 1

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

    del df, ix, iy, iz, x_coords, y_coords
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

    ix_t = df_tras['IX'].astype(np.int32).values - 1
    iy_t = df_tras['IY'].astype(np.int32).values - 1
    iz_t = df_tras['IZ'].astype(np.int32).values - 1

    dose_3d[ix_t, iy_t, iz_t] += df_tras['Dose'].values

    del df_tras, ix_t, iy_t, iz_t
    gc.collect()

    doses_validas = dose_3d[dose_3d > 1e-10]
    dose_maxima_referencia = float(np.percentile(doses_validas, 99.5)) if len(doses_validas) > 0 else float(dose_3d.max())
    del doses_validas

    # Multiplicação in-place para não duplicar a matriz 3D na memória RAM
    dose_3d *= (100.0 / dose_maxima_referencia)

    # Salva o cache binário comprimido para não precisar reprocessar os .dat
    try:
        np.savez_compressed(
            ARQUIVO_CACHE_BINARIO,
            dose_normalizada=dose_3d,
            limites_xy=np.array(limites_xy, dtype=np.float32),
            nbz=np.int32(nbz)
        )
    except Exception:
        pass

    gc.collect()
    return dose_3d, limites_xy, nb
