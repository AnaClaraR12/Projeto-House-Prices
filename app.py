import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Prices",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            text-align: center;
            color: #555;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        }
        div[data-testid="metric-container"] {
            background-color: #f0f4ff;
            border-left: 5px solid #1f4e79;
            border-radius: 8px;
            padding: 10px 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Caminhos
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"

# ──────────────────────────────────────────────────────────────
# Funções de dados / modelos (com cache)
# ──────────────────────────────────────────────────────────────


@st.cache_data
def load_raw():
    return pd.read_csv(TRAIN_PATH)


@st.cache_data
def preprocess():
    base = pd.read_csv(TRAIN_PATH)

    # Variável binária para ar-condicionado
    base["central_air"] = base["CentralAir"].apply(
        lambda x: 1 if x == "Y" else 0)

    # Colunas com mais de 10% de nulos → remover
    eliminar = base.columns[
        (base.isnull().sum() / base.shape[0]) > 0.10
    ].tolist()
    base = base.drop(eliminar, axis=1)

    # BsmtQual: preencher nulos com 'None' + One-Hot Encoding
    base["BsmtQual"] = base["BsmtQual"].fillna("None")
    bsmt_dummies = pd.get_dummies(
        base["BsmtQual"], prefix="BsmtQual").astype(int)
    base = pd.concat([base, bsmt_dummies], axis=1).drop("BsmtQual", axis=1)

    # MSZoning: preencher com moda
    base["MSZoning"] = base["MSZoning"].fillna(base["MSZoning"].mode()[0])

    # KitchenQual: One-Hot com drop_first=True (evitar multicolinearidade)
    kitchen_dummies = pd.get_dummies(
        base["KitchenQual"], prefix="KitchenQual", drop_first=True
    ).astype(int)
    base = pd.concat([base, kitchen_dummies], axis=1).drop(
        "KitchenQual", axis=1)

    # Manter apenas colunas numéricas
    colunas_num = base.columns[base.dtypes != "object"]
    base2 = base[colunas_num].copy()

    # Imputação restante
    base2 = base2.fillna({"GarageYrBlt": -1, "MasVnrArea": 0})

    return base2, eliminar


@st.cache_resource
def train_models():
    base2, eliminar = preprocess()

    X = base2.drop("SalePrice", axis=1)
    y = base2["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    reg_rl = LinearRegression().fit(X_train, y_train)
    reg_ar = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    reg_knn = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)

    modelos = {
        "Regressão Linear": reg_rl,
        "Árvore de Decisão": reg_ar,
        "KNN (k=2)": reg_knn,
    }

    resultados = {}
    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        resultados[nome] = {
            "model":       modelo,
            "predictions": y_pred,
            "MAE":  mean_absolute_error(y_test, y_pred),
            "MSE":  mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

    feature_cols = X.columns.tolist()
    medians = X.median().to_dict()

    return resultados, X_train, X_test, y_train, y_test, feature_cols, medians


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## House Prices")
    st.caption("Competição Kaggle — Previsão de preços de imóveis")
    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão Geral",
            "Comparativo de Modelos",
            "Simulador de Previsão",
        ],
    )

    st.divider()
    st.markdown("**Melhor resultado Kaggle**")
    st.markdown("Score: **0.16529**")
    st.markdown("Modelo: Regressão Linear")
    st.divider()
    st.markdown(
        "[Competição no Kaggle](https://www.kaggle.com/competitions/"
        "house-prices-advanced-regression-techniques)"
    )


# ══════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
if pagina == "Visão Geral":
    st.markdown('<div class="main-header">House Prices — Kaggle</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Previsão de preços de imóveis com Machine Learning</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    raw = load_raw()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registros", f"{raw.shape[0]:,}")
    c2.metric("Features", f"{raw.shape[1]:,}")
    c3.metric("Preço Médio",   f"$ {raw['SalePrice'].mean():,.0f}")
    c4.metric("Preço Mediano", f"$ {raw['SalePrice'].median():,.0f}")
    c5.metric("Score Kaggle",  "0.16529")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Primeiros Registros")
        st.dataframe(raw.head(10), width="stretch")

    with col_r:
        st.markdown("#### Estatísticas Descritivas")
        st.dataframe(
            raw.describe().round(2),
            width="stretch",
        )

# ══════════════════════════════════════════════════════════════
# PÁGINA 2 — COMPARATIVO DE MODELOS
# ══════════════════════════════════════════════════════════════
elif pagina == "Comparativo de Modelos":
    st.markdown("## Comparativo de Modelos de Machine Learning")

    with st.spinner("Treinando modelos…"):
        resultados, X_train, X_test, y_train, y_test, feature_cols, medians = train_models()

    # Tabela de métricas
    st.markdown("### Métricas de Desempenho")
    metricas_rows = [
        {
            "Modelo": nome,
            "MAE":  f"$ {info['MAE']:,.2f}",
            "RMSE": f"$ {info['RMSE']:,.2f}",
            "MSE":  f"{info['MSE']:,.0f}",
        }
        for nome, info in resultados.items()
    ]
    st.dataframe(pd.DataFrame(metricas_rows), width="stretch", hide_index=True)

    # Cards
    cols = st.columns(3)
    for i, (nome, info) in enumerate(resultados.items()):
        with cols[i]:
            st.markdown(f"#### {nome}")
            st.metric("MAE",  f"$ {info['MAE']:,.0f}")
            st.metric("RMSE", f"$ {info['RMSE']:,.0f}")

    st.divider()

    # Scatter Real vs Previsto
    st.markdown("### Real vs Previsto (em milhares $)")
    cores = ["#1f4e79", "#2e75b6", "#9dc3e6"]
    nomes = list(resultados.keys())

    fig = make_subplots(rows=1, cols=3, subplot_titles=nomes)
    for i, (nome, info) in enumerate(resultados.items()):
        y_r = y_test.values / 1000
        y_p = info["predictions"] / 1000
        max_val = max(y_r.max(), y_p.max())

        fig.add_trace(
            go.Scatter(
                x=y_r, y=y_p,
                mode="markers",
                marker=dict(color=cores[i], opacity=0.55, size=5),
                name=nome,
                hovertemplate="Real: $%{x:.0f}k<br>Previsto: $%{y:.0f}k",
            ),
            row=1, col=i + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Ideal",
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )

    fig.update_xaxes(title_text="Real ($ mil)")
    fig.update_yaxes(title_text="Previsto ($ mil)")
    fig.update_layout(height=430, title_text="Valores Reais vs Previstos")
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Interpretação:** Quanto mais próximos os pontos da linha vermelha "
        "tracejada (previsão perfeita), melhor o modelo. A **Regressão Linear** "
        "obteve o menor MAE e RMSE, resultando em score **0.16529** no Kaggle."
    )

# ══════════════════════════════════════════════════════════════
# PÁGINA 3 — SIMULADOR DE PREVISÃO
# ══════════════════════════════════════════════════════════════
elif pagina == "Simulador de Previsão":
    st.markdown("## Simulador de Previsão de Preço")
    st.caption(
        "Ajuste as características do imóvel e veja a estimativa do modelo "
        "de Regressão Linear."
    )

    with st.spinner("Carregando modelo…"):
        resultados, _, _, _, _, feature_cols, medians = train_models()
    reg_rl = resultados["Regressão Linear"]["model"]
    raw = load_raw()

    st.divider()

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown("### Características do Imóvel")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            overall_qual = st.slider("Qualidade Geral (1–10)", 1, 10, 7)
            year_built = st.slider("Ano de Construção", 1872, 2010, 2000)
            full_bath = st.slider("Banheiros Completos", 0, 4, 2)
        with r1c2:
            gr_liv_area = st.number_input(
                "Área Habitável (sq ft)", 300, 6000, 1500, step=50)
            garage_area = st.number_input(
                "Área da Garagem (sq ft)", 0, 1500, 400, step=50)
            bedroom = st.slider("Quartos (acima do solo)", 0, 8, 3)
        with r1c3:
            total_bsmt = st.number_input(
                "Área do Porão (sq ft)", 0, 3000, 800, step=50)
            lot_area = st.number_input(
                "Terreno (sq ft)", 1000, 100000, 8000, step=500)
            central_air_input = st.selectbox(
                "Ar-Cond. Central", ["Sim", "Não"])

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            bsmt_qual_input = st.selectbox("Qualidade do Porão",  [
                                           "Gd", "Ex", "TA", "Fa", "None"])
        with r2c2:
            kitchen_qual_input = st.selectbox(
                "Qualidade da Cozinha", ["Gd", "Ex", "TA", "Fa"])

    # ── Montar linha de entrada ──────────────────────────────
    input_data = dict(medians)  # começa com medianas de treinamento

    # Overrides com valores do usuário
    overrides = {
        "OverallQual":   overall_qual,
        "YearBuilt":     year_built,
        "FullBath":      full_bath,
        "GrLivArea":     gr_liv_area,
        "GarageArea":    garage_area,
        "BedroomAbvGr":  bedroom,
        "TotalBsmtSF":   total_bsmt,
        "LotArea":       lot_area,
        "central_air":   1 if central_air_input == "Sim" else 0,
    }
    for k, v in overrides.items():
        if k in input_data:
            input_data[k] = v

    # BsmtQual dummies
    for cat in ["Ex", "Fa", "Gd", "None", "TA"]:
        key = f"BsmtQual_{cat}"
        if key in input_data:
            input_data[key] = 1 if bsmt_qual_input == cat else 0

    # KitchenQual dummies (drop_first=True → Ex foi removida como baseline)
    for cat in ["Fa", "Gd", "TA"]:
        key = f"KitchenQual_{cat}"
        if key in input_data:
            input_data[key] = 1 if kitchen_qual_input == cat else 0

    # Previsão
    input_df = pd.DataFrame([input_data])[feature_cols]
    predicao = reg_rl.predict(input_df)[0]
    predicao = max(predicao, 0)  # sem preços negativos

    price_median = raw["SalePrice"].median()
    percentile = float((raw["SalePrice"] < predicao).mean() * 100)
    pct_vs_median = (predicao / price_median - 1) * 100

    with col_result:
        st.markdown("### Estimativa de Preço")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1f4e79, #2e75b6);
                color: white; border-radius: 14px;
                padding: 28px 20px; text-align: center;
            ">
                <div style="font-size:0.9rem; opacity:0.8;">Preço Estimado</div>
                <div style="font-size:2.6rem; font-weight:bold; margin:8px 0;">
                    $ {predicao:,.0f}
                </div>
                <div style="font-size:0.85rem; opacity:0.75;">Regressão Linear</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.metric(
            "vs. Mediana do dataset",
            f"$ {predicao:,.0f}",
            f"{pct_vs_median:+.1f}%",
        )
        st.metric("Percentil no dataset", f"{percentile:.0f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Baseado no modelo treinado com os dados do Kaggle. Score: **0.16529**.")
