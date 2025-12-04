# -*- coding: utf-8 -*-
"""
Biblioteca de Atribuição Multi‑Toque (MTA) — funções para geração de dados, atribuição e métricas.
"""
import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# Config padrão
CHANNELS = [
    'Email', 'Instagram', 'Google Ads', 'LinkedIn Ads',
    'Google Search', 'Site IBGC', 'Direct', 'Referral'
]

NP_SEED = 42

# Probabilidades base para simulação
P_FIRST = {
    'Instagram': 0.20,
    'Google Ads': 0.25,
    'LinkedIn Ads': 0.15,
    'Google Search': 0.30,
    'Referral': 0.07,
    'Site IBGC': 0.03,
    'Email': 0.00,
    'Direct': 0.00
}

P_MIDDLE = {
    'Email': 0.25,
    'Instagram': 0.10,
    'Google Ads': 0.15,
    'LinkedIn Ads': 0.08,
    'Google Search': 0.15,
    'Site IBGC': 0.15,
    'Direct': 0.07,
    'Referral': 0.05
}

P_LAST = {
    'Email': 0.20,
    'Google Search': 0.35,
    'Direct': 0.20,
    'Google Ads': 0.10,
    'LinkedIn Ads': 0.05,
    'Instagram': 0.05,
    'Site IBGC': 0.04,
    'Referral': 0.01
}

BETA = {
    'Intercept': -2.2,
    'Email': 0.20, 'Instagram': 0.08, 'Google Ads': 0.22, 'LinkedIn Ads': 0.12,
    'Google Search': 0.45, 'Site IBGC': 0.15, 'Direct': 0.10, 'Referral': 0.05,
    'RecencyDays': -0.015, 'Diversity': 0.05
}

MONTHLY_SPEND = {
    'Email': 15000, 'Instagram': 30000, 'Google Ads': 80000, 'LinkedIn Ads': 45000,
    'Google Search': 0, 'Site IBGC': 0, 'Direct': 0, 'Referral': 0
}

# Util
np.random.seed(NP_SEED)

def _choice(d):
    labels = list(d.keys()); probs = np.array(list(d.values()))
    probs = probs / probs.sum()
    return np.random.choice(labels, p=probs)

# ----------------
# Geração de dados
# ----------------

def generate_synthetic_dataset(n_users=3000, lookback_days=60, val_mean=900, val_sd=250, max_touches=15, base_date=None):
    """Gera DataFrames: touchpoints, conversions, channel_costs."""
    if base_date is None:
        base_date = datetime.today()
    touch_rows, conv_rows = [], []
    for uid in range(1, n_users+1):
        n_t = min(np.random.poisson(4) + 1, max_touches)
        if n_t < 2:
            n_t = 2
        days = np.sort(np.random.randint(1, lookback_days+1, size=n_t))
        ts = [base_date - timedelta(days=int(d)) - timedelta(minutes=np.random.randint(0, 24*60)) for d in days]
        chs = []
        if n_t >= 1: chs.append(_choice(P_FIRST))
        for i in range(1, n_t-1): chs.append(_choice(P_MIDDLE))
        if n_t >= 2: chs.append(_choice(P_LAST))
        eng = np.clip(np.random.beta(a=2, b=5, size=n_t), 0, 1)
        counts = Counter(chs)
        diversity = len(counts)
        recency_days = (base_date - ts[-1]).days
        s = BETA['Intercept'] + BETA['RecencyDays']*recency_days + BETA['Diversity']*diversity
        for c, cnt in counts.items(): s += BETA[c]*cnt
        p_conv = 1/(1+math.exp(-s))
        converted = np.random.rand() < p_conv
        conv_time = ts[-1] + timedelta(minutes=np.random.randint(5, 240)) if converted else None
        value = float(np.clip(np.random.normal(val_mean, val_sd), 150, 3000)) if converted else 0.0
        for i in range(n_t):
            touch_rows.append({'user_id': uid, 'timestamp': ts[i], 'channel': chs[i], 'event_type': 'click', 'engagement_score': eng[i]})
        conv_rows.append({'user_id': uid, 'converted': int(converted), 'timestamp': conv_time, 'conversion_type': 'Inscrição', 'value': value})
    touchpoints = pd.DataFrame(touch_rows)
    conversions = pd.DataFrame(conv_rows)
    # custos por clique
    channel_clicks = touchpoints['channel'].value_counts().to_dict()
    cpc = {ch: (MONTHLY_SPEND.get(ch,0) / max(channel_clicks.get(ch,1), 1)) for ch in CHANNELS}
    touchpoints['cost'] = touchpoints['channel'].map(cpc)
    # custos por dia
    start_date = (base_date - timedelta(days=lookback_days)).date()
    dates = pd.date_range(start_date, base_date.date(), freq='D')
    rows = []
    for d in dates:
        for ch, mspend in MONTHLY_SPEND.items():
            daily = (mspend/30.0) * np.random.uniform(0.8, 1.2)
            rows.append({'date': d, 'channel': ch, 'campaign_id': f'{ch}-GEN', 'spend': daily})
    channel_costs = pd.DataFrame(rows)
    return touchpoints, conversions, channel_costs

# ----------
# Paths
# ----------

def build_paths(touchpoints, conversions):
    paths = []
    for uid, df_u in touchpoints.sort_values(['user_id','timestamp']).groupby('user_id'):
        seq = df_u['channel'].tolist()
        conv_flag = int(conversions.loc[conversions['user_id']==uid, 'converted'].values[0])
        seq = seq + (['Conversion'] if conv_flag else ['Null'])
        paths.append({'user_id': uid, 'path': seq})
    return pd.DataFrame(paths)

# ----------------------
# Modelos de atribuição
# ----------------------

def attribution_u_shaped(paths_df, conversions):
    credit = defaultdict(float)
    for _, row in paths_df.iterrows():
        seq = row['path']
        if seq[-1] != 'Conversion': continue
        uid = row['user_id']
        value = float(conversions.loc[conversions['user_id']==uid,'value'].values[0])
        touches = [c for c in seq[:-1]]; n = len(touches)
        if n == 1: credit[touches[0]] += value
        else:
            w = np.zeros(n); w[0] = 0.40; w[-1] = 0.40
            if n > 2: w[1:-1] = 0.20/(n-2)
            for ch, wt in zip(touches, w): credit[ch] += value*wt
    s = pd.Series(credit)
    return s.sort_values(ascending=False)


def attribution_time_decay(paths_df, touchpoints, conversions, lam=0.08):
    credit = defaultdict(float)
    df_t = touchpoints.sort_values(['user_id','timestamp'])
    for _, row in paths_df.iterrows():
        seq = row['path']
        if seq[-1] != 'Conversion': continue
        uid = row['user_id']
        value = float(conversions.loc[conversions['user_id']==uid,'value'].values[0])
        tp_u = df_t[df_t['user_id']==uid]
        conv_time = conversions.loc[conversions['user_id']==uid,'timestamp'].values[0]
        weights, ch_seq = [], []
        for _, r in tp_u.iterrows():
            dt_days = (conv_time - r['timestamp']).total_seconds()/86400.0
            w = math.exp(-lam*max(dt_days,0))
            weights.append(w); ch_seq.append(r['channel'])
        weights = np.array(weights)
        if weights.sum() == 0: continue
        weights = weights/weights.sum()
        for ch, wt in zip(ch_seq, weights): credit[ch] += value*wt
    s = pd.Series(credit)
    return s.sort_values(ascending=False)

# Markov

def build_transition_matrix(paths_df, channels=CHANNELS):
    nodes = ['Start'] + channels + ['Conversion','Null']
    from collections import Counter
    counts = {n: Counter() for n in nodes}
    for _, row in paths_df.iterrows():
        seq = row['path']
        if len(seq) == 0: continue
        counts['Start'][seq[0]] += 1
        for a,b in zip(seq[:-1], seq[1:]): counts[a][b] += 1
    P = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for a in nodes:
        total = sum(counts[a].values())
        if total > 0:
            for b, c in counts[a].items(): P.loc[a,b] = c/total
    return P


def _path_prob(seq, P):
    if len(seq) == 0: return 0.0
    prob = P.loc['Start', seq[0]]
    if prob == 0: return 0.0
    for a,b in zip(seq[:-1], seq[1:]):
        p = P.loc[a,b]
        if p == 0: return 0.0
        prob *= p
    return float(prob)


def removal_effect_markov(P, paths_df, conversions, channels=CHANNELS):
    baseline = 0.0
    for _, row in paths_df.iterrows():
        if row['path'][-1] == 'Conversion': baseline += _path_prob(row['path'], P)
    effects = {}
    for ch in channels:
        removed_sum = 0.0
        for _, row in paths_df.iterrows():
            seq = row['path']
            if seq[-1] != 'Conversion': continue
            seq_removed = [x for x in seq if x != ch]
            removed_sum += _path_prob(seq_removed, P)
        effects[ch] = max(baseline - removed_sum, 0.0)
    s = pd.Series(effects)
    return (s/s.sum()).sort_values(ascending=False) if s.sum() > 0 else s

# IA (logística)

def make_user_features(touchpoints, conversions, base_date=None):
    if base_date is None:
        base_date = datetime.today()
    df_t = touchpoints.copy(); df_c = conversions.copy()
    cnt = df_t.pivot_table(index='user_id', columns='channel', values='event_type', aggfunc='count', fill_value=0)
    cnt.columns = [f'cnt_{c}' for c in cnt.columns]
    eng = df_t.groupby('user_id')['engagement_score'].mean().rename('eng_mean')
    total = df_t.groupby('user_id')['event_type'].count().rename('touches_total')
    diversity = df_t.groupby('user_id')['channel'].nunique().rename('diversity')
    last_touch = df_t.sort_values(['user_id','timestamp']).groupby('user_id').tail(1)[['user_id','channel']].set_index('user_id')
    last_ohe = pd.get_dummies(last_touch['channel'], prefix='last')
    last_ts = df_t.groupby('user_id')['timestamp'].max().rename('last_ts')
    recency_days = ((base_date - last_ts).dt.total_seconds()/86400.0).rename('recency_days')
    X = pd.concat([cnt, eng, total, diversity, last_ohe, recency_days], axis=1).fillna(0)
    y = df_c.set_index('user_id')['converted']
    values = df_c.set_index('user_id')['value']
    return X, y, values


def logistic_attribution(touchpoints, conversions):
    from sklearn.model_selection import train_test_split
    X, y, values = make_user_features(touchpoints, conversions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=NP_SEED)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    perm = permutation_importance(model, X_test, y_test, n_repeats=7, random_state=NP_SEED)
    imp_df = pd.DataFrame({'feature': X_test.columns, 'importance': perm.importances_mean}).sort_values('importance', ascending=False)
    # Crédito por coeficientes * contagem (aprox.)
    coef = pd.Series(model.coef_[0], index=X.columns)
    channel_features = [c for c in X.columns if c.startswith('cnt_')]
    channel_coefs = coef[channel_features]
    credit = defaultdict(float)
    for uid in X.index:
        if y.loc[uid] == 0 or values.loc[uid] <= 0: continue
        w = {}
        for f in channel_features:
            ch = f.replace('cnt_','')
            wt = max(X.loc[uid, f] * channel_coefs.get(f, 0), 0)
            w[ch] = wt
        s = sum(w.values())
        if s == 0:
            last_cols = [c for c in X.columns if c.startswith('last_')]
            last_ch = None
            for lc in last_cols:
                if X.loc[uid, lc] == 1:
                    last_ch = lc.replace('last_',''); break
            credit[last_ch or 'Direct'] += float(values.loc[uid])
            continue
        for ch, wt in w.items(): credit[ch] += float(values.loc[uid]) * (wt/s)
    credit_s = pd.Series(credit).sort_values(ascending=False)
    return model, imp_df, credit_s

# Uplift (two-model) — exemplo por LinkedIn Ads

def uplift_two_model(touchpoints, conversions, treatment_channel='LinkedIn Ads'):
    from sklearn.model_selection import train_test_split
    X, y, _ = make_user_features(touchpoints, conversions)
    T = (X[f'cnt_{treatment_channel}'] > 0).astype(int)
    features_u = [c for c in X.columns if c != f'cnt_{treatment_channel}']
    Xu_treated = X.loc[T==1, features_u]; y_treated = y.loc[T==1]
    Xu_control = X.loc[T==0, features_u]; y_control = y.loc[T==0]
    if Xu_treated.shape[0] < 50 or Xu_control.shape[0] < 50:
        return pd.Series(np.zeros(len(X))), pd.Series([0])
    m_t = LogisticRegression(max_iter=500).fit(Xu_treated, y_treated)
    m_c = LogisticRegression(max_iter=500).fit(Xu_control, y_control)
    p_t = m_t.predict_proba(X[features_u])[:,1]
    p_c = m_c.predict_proba(X[features_u])[:,1]
    uplift = p_t - p_c
    order = np.argsort(-uplift)
    conv_array = y.values; treat_array = T.values
    inc = []; cum = 0
    for i in order:
        if treat_array[i] == 1: cum += conv_array[i]
        else: cum -= conv_array[i]*0.5
        inc.append(cum)
    return pd.Series(uplift, index=X.index), pd.Series(inc)

# Métricas de eficiência com base em um crédito

def compute_roas_cpa(credit_series, conversions, channel_costs):
    total_rev = float(conversions['value'].sum())
    share = (credit_series/credit_series.sum()) if credit_series.sum() > 0 else credit_series
    rev_attr = share * total_rev
    spend = channel_costs.groupby('channel')['spend'].sum()
    ticket = max(conversions['value'].mean(), 1)
    conv_attr = rev_attr / ticket
    roas = (rev_attr / (spend + 1e-9)).fillna(0)
    cpa = (spend / (conv_attr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return rev_attr, roas, cpa

# Top paths

def top_converting_paths(paths_df, topn=10):
    conv_paths = [' > '.join([c for c in seq[:-1]]) for seq in paths_df['path'] if seq[-1]=='Conversion']
    return pd.Series(conv_paths).value_counts().head(topn)
