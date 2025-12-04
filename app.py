# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.mta import (
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

# Carrega ou gera dados
touchpoints = conversions = channel_costs = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        touchpoints = pd.read_excel(xls, 'touchpoints')
        conversions = pd.read_excel(xls, 'conversions')
        channel_costs = pd.read_excel(xls, 'channel_costs')
        st.success('Arquivo carregado com sucesso!')
    except Exception as e:
        st.error(f'Erro ao ler o Excel: {e}')

if (touchpoints is None or conversions is None or channel_costs is None) and btn_generate:
    tp, cv, cc = generate_synthetic_dataset(n_users=n_users, lookback_days=lookback)
    touchpoints, conversions, channel_costs = tp, cv, cc
    st.success('Dataset fictício gerado!')

if touchpoints is None:
    st.info('\uD83D\uDCA1 Carregue um Excel ou clique em **Gerar Dataset Fictício** na barra lateral.')
    st.stop()

# Construção de paths
paths_df = build_paths(touchpoints, conversions)

# Modelos de crédito
st.subheader('Crédito por Canal — Comparativo de Modelos')
col1, col2 = st.columns(2)
with col1:
    u_credit = attribution_u_shaped(paths_df, conversions)
    fig, ax = plt.subplots(figsize=(6,4))
    (u_credit/u_credit.sum()).sort_values().plot(kind='barh', color='#4c78a8', ax=ax)
    ax.set_title('U‑Shaped'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)
with col2:
    d_credit = attribution_time_decay(paths_df, touchpoints, conversions, lam=lam)
    fig, ax = plt.subplots(figsize=(6,4))
    (d_credit/d_credit.sum()).sort_values().plot(kind='barh', color='#f58518', ax=ax)
    ax.set_title('Time‑Decay'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    P = build_transition_matrix(paths_df)
    markov_credit = removal_effect_markov(P, paths_df, conversions)
    fig, ax = plt.subplots(figsize=(6,4))
    markov_credit.sort_values().plot(kind='barh', color='#54a24b', ax=ax)
    ax.set_title('Markov (Efeito de Remoção)'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)
with col4:
    model, imp_df, credit_lr = logistic_attribution(touchpoints, conversions)
    fig, ax = plt.subplots(figsize=(6,4))
    (credit_lr/credit_lr.sum()).sort_values().plot(kind='barh', color='#e45756', ax=ax)
    ax.set_title('IA — Logística (aprox.)'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

# Importância de features
st.subheader('IA — Importância das Variáveis (Logística)')
fig, ax = plt.subplots(figsize=(8,4))
imp_df.head(12).sort_values('importance').plot(kind='barh', x='feature', y='importance', color='#72b7b2', ax=ax)
ax.set_title('Importância por Permutação – Top Features'); ax.set_xlabel('Impacto na Acurácia (Δ)')
st.pyplot(fig)

# Uplift
st.subheader('Incrementalidade — Curva Uplift (LinkedIn Ads – demo)')
uplift, inc = uplift_two_model(touchpoints, conversions, treatment_channel='LinkedIn Ads')
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.linspace(0,1,len(inc)), inc, color='#e45756', lw=2)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Percentil da Audiência (rank por uplift)')
ax.set_ylabel('Incremento Cumulativo (aprox.)')
ax.set_title('Curva Uplift (LinkedIn Ads) — Demo')
st.pyplot(fig)

# ROAS & CPA (Markov)
st.subheader('Eficiência por Canal — ROAS e CPA (Markov)')
rev_markov, roas, cpa = compute_roas_cpa(markov_credit, conversions, channel_costs)
colA, colB = st.columns(2)
with colA:
    fig, ax = plt.subplots(figsize=(6,4))
    roas.fillna(0).sort_values().plot(kind='barh', color='#54a24b', ax=ax)
    ax.set_title('ROAS por Canal (Markov)'); ax.set_xlabel('Receita/Gasto')
    st.pyplot(fig)
with colB:
    fig, ax = plt.subplots(figsize=(6,4))
    cpa.fillna(0).sort_values().plot(kind='barh', color='#b279a2', ax=ax)
    ax.set_title('CPA por Canal (Markov)'); ax.set_xlabel('R$ por Conversão')
    st.pyplot(fig)

# Top paths
st.subheader('Top Jornadas que Convertem')
top_paths = top_converting_paths(paths_df, topn=10)
st.dataframe(top_paths.rename('frequência'))

# Download do dataset atual
st.subheader('Baixar Dataset')
with pd.ExcelWriter('tmp_dataset.xlsx', engine='openpyxl') as writer:
    touchpoints.to_excel(writer, index=False, sheet_name='touchpoints')
    conversions.to_excel(writer, index=False, sheet_name='conversions')
    channel_costs.to_excel(writer, index=False, sheet_name='channel_costs')
with open('tmp_dataset.xlsx', 'rb') as f:
    st.download_button('Baixar Excel do Dataset', f, file_name='mta_dataset.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.caption('© 2025 Demo MTA — Este aplicativo é ilustrativo e usa dados fictícios.')
