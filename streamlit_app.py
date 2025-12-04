# -*- coding: utf-8 -*-
import os
import sys

# >>> BLINDAGEM DE IMPORT: adiciona 'src/' ao sys.path <<<
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# >>> Agora importamos diretamente de 'mta' (arquivo dentro de 'src/') <<<
from mta import (
    generate_synthetic_dataset, build_paths,
    attribution_u_shaped, attribution_time_decay,
    build_transition_matrix, removal_effect_markov,
    logistic_attribution, uplift_two_model,
    compute_roas_cpa, top_converting_paths
)

st.set_page_config(page_title='Demo MTA com IA', layout='wide')
st.title('Demo — Atribuição Multi‑Toque (MTA) com IA')
st.caption('Dados fictícios para apresentação executiva | Por Rafaell Villar')

with st.sidebar:
    st.header('Configuração do Dataset')
    n_users = st.slider('Usuários (fictícios)', 1000, 8000, 3000, 500)
    lookback = st.slider('Janela (dias)', 30, 120, 60, 15)
    lam = st.slider('Lambda do Time‑Decay', 0.01, 0.20, 0.08, 0.01)
    uploaded = st.file_uploader('Carregar Excel (.xlsx) com abas touchpoints/conversions/channel_costs', type=['xlsx'])
    btn_generate = st.button('Gerar Dataset Fictício')

# ... (restante do arquivo permanece igual)
