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

# -------------------------------
# Utilidades de saneamento/robustez
# -------------------------------
def _standardize_channel_names(df, col='channel'):
    """Padroniza nomes de canais: remove espaços, normaliza tipo e mantém caso."""
    if df is not None and col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def _ensure_series(obj, preferred_name=None):
    """Garante que o retorno seja uma Series. Se vier DataFrame, tenta pegar a 1ª coluna."""
    if isinstance(obj, pd.Series):
        return obj
    elif isinstance(obj, pd.DataFrame):
        if preferred_name and preferred_name in obj.columns:
            return obj[preferred_name]
        elif len(obj.columns) >= 1:
            return obj.iloc[:, 0]
        else:
            return pd.Series(dtype='float64')
    else:
        return pd.Series(dtype='float64')

def _clean_series_for_plot(s):
    """Remove inf/NaN, estados especiais e entradas sem índice."""
    if s is None:
        return pd.Series(dtype='float64')
    s = _ensure_series(s)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    # Remove estados padrão caso venham na saída de Markov
    s = s[~s.index.isin(['Start', 'Conversion', 'Null'])]
    # Remove índices nulos
    s = s.loc[s.index.notnull()]
    return s

# -------------------------------
# App
# -------------------------------
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

if touchpoints is None or conversions is None or channel_costs is None:
    st.info('💡 Carregue um Excel ou clique em **Gerar Dataset Fictício** na barra lateral.')
    st.stop()

# Padroniza nomes de canais (evita mismatch)
touchpoints = _standardize_channel_names(touchpoints, 'channel')
channel_costs = _standardize_channel_names(channel_costs, 'channel')

# Verificações básicas
if conversions.empty:
    st.warning('Não há conversões no dataset. Os gráficos de Markov e ROAS ficarão vazios.')
if 'value' not in conversions.columns:
    st.info('A aba **conversions** não possui coluna de valor (ex.: "value"). '
            'O ROAS pode sair 0/NaN dependendo da implementação em compute_roas_cpa.')

# Construção de paths
paths_df = build_paths(touchpoints, conversions)

# Modelos de crédito — U-Shaped e Time-Decay
st.subheader('Crédito por Canal — Comparativo de Modelos')
col1, col2 = st.columns(2)
with col1:
    u_credit = attribution_u_shaped(paths_df, conversions)
    u_clean = _clean_series_for_plot(u_credit)
    fig, ax = plt.subplots(figsize=(6,4))
    if u_clean.empty or u_clean.sum() == 0:
        st.warning('U‑Shaped sem dados suficientes para plotar.')
    else:
        (u_clean / max(u_clean.sum(), 1e-9)).sort_values().plot(kind='barh', color='#4c78a8', ax=ax)
        ax.set_title('U‑Shaped'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

with col2:
    d_credit = attribution_time_decay(paths_df, touchpoints, conversions, lam=lam)
    d_clean = _clean_series_for_plot(d_credit)
    fig, ax = plt.subplots(figsize=(6,4))
    if d_clean.empty or d_clean.sum() == 0:
        st.warning('Time‑Decay sem dados suficientes para plotar.')
    else:
        (d_clean / max(d_clean.sum(), 1e-9)).sort_values().plot(kind='barh', color='#f58518', ax=ax)
        ax.set_title('Time‑Decay'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

# Markov e Logística
col3, col4 = st.columns(2)
with col3:
    P = build_transition_matrix(paths_df)

    # Chamada robusta — diferentes assinaturas possíveis
    try:
        markov_credit = removal_effect_markov(P, paths_df, conversions)
    except TypeError:
        try:
            markov_credit = removal_effect_markov(P)
        except TypeError:
            markov_credit = removal_effect_markov(paths_df, conversions)

    mc = _clean_series_for_plot(markov_credit)
    fig, ax = plt.subplots(figsize=(6,4))
    if mc.empty or mc.sum() == 0:
        st.warning('Markov (Efeito de Remoção) sem crédito calculado — verifique se há paths com conversão e o estado "Conversion" na matriz.')
        # Opcional: mostrar estados da matriz para diagnóstico
        st.caption(f"Estados (linhas) na matriz: {list(P.index)}")
        st.caption(f"Estados (colunas) na matriz: {list(P.columns)}")
    else:
        mc.sort_values().plot(kind='barh', color='#54a24b', ax=ax)
        ax.set_title('Markov (Efeito de Remoção)'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

with col4:
    # IA Logística
    model, imp_df, credit_lr = logistic_attribution(touchpoints, conversions)
    lr_clean = _clean_series_for_plot(credit_lr)
    fig, ax = plt.subplots(figsize=(6,4))
    if lr_clean.empty or lr_clean.sum() == 0:
        st.warning('IA — Logística sem crédito calculado.')
    else:
        (lr_clean / max(lr_clean.sum(), 1e-9)).sort_values().plot(kind='barh', color='#e45756', ax=ax)
        ax.set_title('IA — Logística (aprox.)'); ax.set_xlabel('Share do Crédito')
    st.pyplot(fig)

# Importância de features
st.subheader('IA — Importância das Variáveis (Logística)')
fig, ax = plt.subplots(figsize=(8,4))
try:
    imp_plot = imp_df.copy()
    if 'feature' in imp_plot.columns and 'importance' in imp_plot.columns:
        imp_plot.head(12).sort_values('importance').plot(kind='barh', x='feature', y='importance', color='#72b7b2', ax=ax)
        ax.set_title('Importância por Permutação – Top Features'); ax.set_xlabel('Impacto na Acurácia (Δ)')
    else:
        st.warning('Tabela de importância sem colunas esperadas ("feature", "importance").')
except Exception as e:
    st.warning(f'Falha ao plotar importância: {e}')
st.pyplot(fig)

# Uplift
st.subheader('Incrementalidade — Curva Uplift (LinkedIn Ads – demo)')
uplift, inc = uplift_two_model(touchpoints, conversions, treatment_channel='LinkedIn Ads')
fig, ax = plt.subplots(figsize=(8,4))
try:
    x = np.linspace(0, 1, len(inc)) if len(inc) > 0 else np.array([])
    if len(x) == 0:
        st.warning('Curva Uplift sem pontos para plotar.')
    else:
        ax.plot(x, inc, color='#e45756', lw=2)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Percentil da Audiência (rank por uplift)')
        ax.set_ylabel('Incremento Cumulativo (aprox.)')
        ax.set_title('Curva Uplift (LinkedIn Ads) — Demo')
except Exception as e:
    st.warning(f'Falha ao plotar uplift: {e}')
st.pyplot(fig)

# ROAS & CPA (Markov)
st.subheader('Eficiência por Canal — ROAS e CPA (Markov)')
# Chamada robusta — diferentes assinaturas possíveis
try:
    rev_markov, roas, cpa = compute_roas_cpa(markov_credit, conversions, channel_costs)
except TypeError:
    try:
        rev_markov, roas, cpa = compute_roas_cpa(markov_credit, channel_costs)
    except TypeError:
        # Último fallback: sem conversions
        rev_markov, roas, cpa = compute_roas_cpa(markov_credit, pd.DataFrame(), channel_costs)

# Padroniza e alinha canais com channel_costs
channels_ref = list(channel_costs['channel'].dropna().astype(str).str.strip().unique())
roas = _ensure_series(roas).copy()
cpa = _ensure_series(cpa).copy()

# Alinha ao conjunto de canais de custos (evita NaN generalizado)
roas = roas.reindex(channels_ref)
cpa = cpa.reindex(channels_ref)

colA, colB = st.columns(2)
with colA:
    fig, ax = plt.subplots(figsize=(6,4))
    roas_clean = roas.replace([np.inf, -np.inf], np.nan)
    if roas_clean.dropna().empty:
        st.warning('ROAS por Canal (Markov) está vazio — confira nomes de canais, custos e valores de conversão.')
        # Diagnóstico
        st.caption(f"Canais nos custos: {channels_ref}")
        st.caption(f"Canais no crédito Markov: {list(_clean_series_for_plot(markov_credit).index)}")
    else:
        roas_clean.fillna(0).sort_values().plot(kind='barh', color='#54a24b', ax=ax)
        ax.set_title('ROAS por Canal (Markov)'); ax.set_xlabel('Receita/Gasto')
    st.pyplot(fig)

with colB:
    fig, ax = plt.subplots(figsize=(6,4))
    cpa_clean = cpa.replace([np.inf, -np.inf], np.nan)
    if cpa_clean.dropna().empty:
        st.warning('CPA por Canal (Markov) está vazio — confira nomes de canais e coluna de custos.')
    else:
        cpa_clean.fillna(0).sort_values().plot(kind='barh', color='#b279a2', ax=ax)
        ax.set_title('CPA por Canal (Markov)'); ax.set_xlabel('R$ por Conversão')
    st.pyplot(fig)

# Top paths
st.subheader('Top Jornadas que Convertem')
try:
    top_paths = top_converting_paths(paths_df, topn=10)
    if isinstance(top_paths, pd.Series):
        st.dataframe(top_paths.rename('frequência'))
    elif isinstance(top_paths, pd.DataFrame):
        st.dataframe(top_paths)
    else:
        st.warning('Função top_converting_paths não retornou dados tabulares.')
except Exception as e:
    st.warning(f'Falha ao gerar top paths: {e}')

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
