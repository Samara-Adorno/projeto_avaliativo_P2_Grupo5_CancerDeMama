from pathlib import Path



import joblib

import matplotlib



matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import streamlit as st





ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "dataset.csv"

MODEL_PATH = ROOT / "model" / "modelo_final.joblib"

FEATURES_PATH = ROOT / "model" / "feature_names.joblib"

METRICS_PATH = ROOT / "reports" / "metricas_consolidadas.csv"

FEATURE_IMPORTANCE_PATH = ROOT / "reports" / "features_importantes.csv"

LOGO_PATH = ROOT / "assets" / "logo_outubro_rosa.png"





st.set_page_config(

    page_title="Diagnóstico de Câncer de Mama",

    layout="wide",

)





@st.cache_resource

def load_model():

    return joblib.load(MODEL_PATH)





@st.cache_data

def load_dataset():

    return pd.read_csv(DATA_PATH)





@st.cache_data

def load_feature_names():

    return joblib.load(FEATURES_PATH)





@st.cache_data

def load_metrics():

    return pd.read_csv(METRICS_PATH)





@st.cache_data

def load_feature_importance():

    return pd.read_csv(FEATURE_IMPORTANCE_PATH)





def diagnosis_label(value):

    return "Maligno" if value in ["M", 1] else "Benigno"





def prepare_model_frame(raw_df, feature_names):

    model_df = raw_df.drop(columns=["id", "Unnamed: 32"]).copy()

    model_df["diagnosis"] = model_df["diagnosis"].map({"M": 1, "B": 0})

    return model_df[feature_names], model_df["diagnosis"]





def make_count_plot(df):

    plot_df = df.copy()

    plot_df["diagnosis_label"] = plot_df["diagnosis"].map(diagnosis_label)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(

        data=plot_df,

        x="diagnosis_label",

        hue="diagnosis_label",

        order=["Benigno", "Maligno"],

        palette={"Benigno": "#79c8c3", "Maligno": "#d63384"},

        legend=False,

        ax=ax,

    )

    ax.set_title("Distribuição dos diagnósticos")

    ax.set_xlabel("Diagnóstico")

    ax.set_ylabel("Quantidade")

    return fig





def make_distribution_plot(df, feature):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(

        data=df,

        x=feature,

        hue="diagnosis",

        kde=True,

        bins=28,

        palette={"B": "#79c8c3", "M": "#d63384"},

        alpha=0.45,

        ax=ax,

    )

    ax.set_title(f"Distribuição por classe: {feature}")

    ax.set_xlabel(feature)

    ax.set_ylabel("Frequência")

    return fig





def make_boxplot(df, feature):

    plot_df = df.copy()

    plot_df["diagnosis_label"] = plot_df["diagnosis"].map(diagnosis_label)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(

        data=plot_df,

        x="diagnosis_label",

        y=feature,

        hue="diagnosis_label",

        order=["Benigno", "Maligno"],

        palette={"Benigno": "#79c8c3", "Maligno": "#d63384"},

        legend=False,

        ax=ax,

    )

    ax.set_title(f"Dispersão e outliers: {feature}")

    ax.set_xlabel("Diagnóstico")

    ax.set_ylabel(feature)

    return fig





def make_corr_plot(df, features):

    fig, ax = plt.subplots(figsize=(9, 7))

    corr = df[features].corr()

    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, ax=ax)

    ax.set_title("Correlação entre variáveis selecionadas")

    return fig





def make_importance_plot(feature_importance):

    fig, ax = plt.subplots(figsize=(8, 5))

    top = feature_importance.sort_values("Importance", ascending=False).head(10)

    sns.barplot(data=top, x="Importance", y="Feature", color="#d63384", ax=ax)

    ax.set_title("Top 10 variáveis mais relevantes")

    ax.set_xlabel("Importância relativa")

    ax.set_ylabel("")

    return fig





def make_metrics_plot(metrics):

    fig, ax = plt.subplots(figsize=(9, 4))

    final_rows = metrics[metrics["Conjunto"].isin(["Teste final", "Teste final apos retreino"])].copy()

    final_rows["Conjunto exibido"] = final_rows["Conjunto"].replace(

        {"Teste final apos retreino": "Teste final após retreino"}

    )

    final_rows["Cenário"] = final_rows["Modelo"] + " - " + final_rows["Conjunto exibido"]

    plot_df = final_rows.melt(

        id_vars="Cenário",

        value_vars=["Acuracia", "Precisao", "Recall", "F1", "AUC"],

        var_name="Métrica",

        value_name="Valor",

    )

    sns.barplot(data=plot_df, x="Métrica", y="Valor", hue="Cenário", ax=ax)

    ax.set_ylim(0, 1.05)

    ax.set_title("Métricas no conjunto de teste")

    ax.set_xlabel("")

    ax.set_ylabel("Valor")

    ax.legend(loc="lower right", fontsize=8)

    return fig





def nearest_cases(input_df, base_X, raw_df, feature_stats, n=5):

    scaled_base = (base_X - feature_stats["mean"]) / feature_stats["std"].replace(0, 1)

    scaled_input = (input_df - feature_stats["mean"]) / feature_stats["std"].replace(0, 1)

    distances = ((scaled_base - scaled_input.iloc[0]) ** 2).sum(axis=1) ** 0.5

    nearest = raw_df.loc[distances.nsmallest(n).index, ["id", "diagnosis"]].copy()

    nearest["diagnosis"] = nearest["diagnosis"].map(diagnosis_label)

    nearest["distancia_padronizada"] = distances.nsmallest(n).round(3).values

    return nearest.rename(columns={"id": "ID", "diagnosis": "Diagnóstico"})





def prediction_band(probability):

    if probability >= 0.75:

        return "Alta probabilidade de malignidade", "error"

    if probability >= 0.45:

        return "Zona intermediária de atenção", "warning"

    return "Baixa probabilidade de malignidade", "success"





st.markdown(

    """

    <style>

    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}

    .app-subtitle {color: #6f2147; font-size: 1.02rem; margin-top: -0.5rem;}

    .notice {

        border-left: 4px solid #d63384;

        background: #fff0f6;

        padding: 0.85rem 1rem;

        border-radius: 0.35rem;

        color: #4a1730;

    }

    .small-note {color: #8a3a60; font-size: 0.9rem;}

    h1, h2, h3 {color: #7a1f4b;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0f6 0%, #ffffff 55%, #f8f1ef 100%);
        border-right: 1px solid #f3c0d8;
    }

    div[data-testid="stMetric"] {
        background: #fff7fb;
        border: 1px solid #f3c0d8;
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
    }

    div.stButton > button {
        background: #d63384;
        color: #ffffff;
        border: 1px solid #d63384;
    }

    div.stButton > button:hover {
        background: #ad1f63;
        color: #ffffff;
        border-color: #ad1f63;
    }

    </style>

    """,

    unsafe_allow_html=True,

)





model = load_model()

df = load_dataset()

feature_names = load_feature_names()

metrics = load_metrics()

feature_importance = load_feature_importance()

X, y = prepare_model_frame(df, feature_names)

feature_stats = X.agg(["min", "mean", "max", "std"]).T

top_features = feature_importance["Feature"].head(10).tolist()

final_metrics = metrics[metrics["Conjunto"] == "Teste final apos retreino"].iloc[0]





with st.sidebar:

    st.header("Menu")

    if LOGO_PATH.exists():

        st.image(str(LOGO_PATH), width=160)

    else:

        st.warning("Logo não encontrada em assets/logo_outubro_rosa.png.")



    menu_option = st.radio(

        "Navegação",

        ["Painel", "Predição", "Análise exploratória", "Modelo", "Dados"],

        label_visibility="collapsed",

    )



    st.divider()

    st.caption("Projeto acadêmico de Machine Learning.")

    st.caption("O resultado não substitui avaliação médica.")



st.title("Diagnóstico de Câncer de Mama")

st.markdown(

    '<p class="app-subtitle">Painel analítico e simulador de predição com modelo RandomForest treinado no dataset Breast Cancer Wisconsin.</p>',

    unsafe_allow_html=True,

)



st.markdown(

    f'<p class="small-note">Seção atual: <strong>{menu_option}</strong></p>',

    unsafe_allow_html=True,

)





if menu_option == "Painel":

    metric_cols = st.columns(5)

    metric_cols[0].metric("Amostras", f"{len(df):,}".replace(",", "."))

    metric_cols[1].metric("Variáveis do modelo", len(feature_names))

    metric_cols[2].metric("Acurácia final", f"{final_metrics['Acuracia']:.2%}")

    metric_cols[3].metric("Recall maligno", f"{final_metrics['Recall']:.2%}")

    metric_cols[4].metric("AUC final", f"{final_metrics['AUC']:.2%}")



    st.markdown(

        """

        <div class="notice">

        Esta aplicação organiza o projeto em um fluxo profissional: entendimento da base, análise visual,

        predição individual, comparação com casos semelhantes e leitura das métricas do modelo final.

        O uso é acadêmico e não substitui diagnóstico médico.

        </div>

        """,

        unsafe_allow_html=True,

    )



    left, right = st.columns([1, 1])

    with left:

        st.subheader("Composição da base")

        st.pyplot(make_count_plot(df), width="stretch")

    with right:

        st.subheader("Modelo final")

        st.pyplot(make_importance_plot(feature_importance), width="stretch")



    st.subheader("Resumo das métricas")

    display_metrics = metrics.copy()

    for col in ["Acuracia", "Precisao", "Recall", "F1", "AUC"]:

        display_metrics[col] = display_metrics[col].map(lambda value: f"{value:.4f}")

    display_metrics = display_metrics.rename(

        columns={"Acuracia": "Acurácia", "Precisao": "Precisão"}

    )

    display_metrics["Conjunto"] = display_metrics["Conjunto"].replace(

        {"Teste final apos retreino": "Teste final após retreino"}

    )

    st.dataframe(display_metrics, width="stretch", hide_index=True)





if menu_option == "Predição":

    st.subheader("Simulador de predição")

    st.write(

        "Use um exemplo pronto ou altere os valores manualmente. O modelo recebe as 30 variáveis numéricas do exame."

    )



    controls, result_area = st.columns([1.15, 0.85])



    with controls:

        preset = st.selectbox(

            "Perfil inicial",

            ["Média da base", "Caso benigno real", "Caso maligno real"],

            help="Os perfis reais usam a primeira amostra de cada classe apenas para demonstração.",

        )



        if preset == "Caso benigno real":

            default_row = df[df["diagnosis"] == "B"].iloc[0]

        elif preset == "Caso maligno real":

            default_row = df[df["diagnosis"] == "M"].iloc[0]

        else:

            default_row = feature_stats["mean"]



        mode = st.radio(

            "Modo de preenchimento",

            ["Guiado pelas principais variáveis", "Completo"],

            horizontal=True,

        )



        input_values = {}

        visible_features = top_features if mode == "Guiado pelas principais variáveis" else feature_names



        st.markdown("#### Variáveis exibidas")

        cols = st.columns(2)

        for idx, feature in enumerate(visible_features):

            stats = feature_stats.loc[feature]

            default_value = float(default_row[feature] if feature in default_row.index else stats["mean"])

            min_value = float(stats["min"])

            max_value = float(stats["max"])

            step = max((max_value - min_value) / 100, 0.0001)

            with cols[idx % 2]:

                input_values[feature] = st.number_input(

                    feature,

                    min_value=min_value,

                    max_value=max_value,

                    value=min(max(default_value, min_value), max_value),

                    step=step,

                    format="%.5f",

                )



        if mode == "Guiado pelas principais variáveis":

            with st.expander("Ajustar as demais variáveis"):

                hidden_features = [name for name in feature_names if name not in visible_features]

                hidden_cols = st.columns(2)

                for idx, feature in enumerate(hidden_features):

                    stats = feature_stats.loc[feature]

                    default_value = float(

                        default_row[feature] if feature in default_row.index else stats["mean"]

                    )

                    min_value = float(stats["min"])

                    max_value = float(stats["max"])

                    step = max((max_value - min_value) / 100, 0.0001)

                    with hidden_cols[idx % 2]:

                        input_values[feature] = st.number_input(

                            feature,

                            min_value=min_value,

                            max_value=max_value,

                            value=min(max(default_value, min_value), max_value),

                            step=step,

                            format="%.5f",

                            key=f"hidden_{feature}",

                        )



        input_df = pd.DataFrame([{feature: input_values[feature] for feature in feature_names}])

        run_prediction = st.button("Executar predição", type="primary", width="stretch")



    with result_area:

        st.markdown("#### Resultado")

        if run_prediction:

            pred = int(model.predict(input_df)[0])

            prob_malignant = float(model.predict_proba(input_df)[0, 1])

            label = "Maligno" if pred == 1 else "Benigno"

            band_text, band_type = prediction_band(prob_malignant)



            st.session_state["last_input_df"] = input_df

            st.session_state["last_prediction"] = {

                "label": label,

                "prob_malignant": prob_malignant,

                "band_text": band_text,

                "band_type": band_type,

            }



        if "last_prediction" not in st.session_state:

            st.info("Preencha os campos e execute a predição para ver o resultado.")

        else:

            result = st.session_state["last_prediction"]

            if result["band_type"] == "error":

                st.error(result["band_text"])

            elif result["band_type"] == "warning":

                st.warning(result["band_text"])

            else:

                st.success(result["band_text"])



            st.metric("Classe prevista", result["label"])

            st.metric("Probabilidade de malignidade", f"{result['prob_malignant']:.2%}")

            st.progress(min(max(result["prob_malignant"], 0), 1))



            if result["label"] == "Maligno":

                st.write(

                    "Interpretação: o padrão informado se aproxima mais dos casos malignos da base. "

                    "Em uso real, esse resultado exigiria avaliação médica e exames complementares."

                )

            else:

                st.write(

                    "Interpretação: o padrão informado se aproxima mais dos casos benignos da base. "

                    "Ainda assim, o modelo não substitui avaliação clínica."

                )



            last_input = st.session_state["last_input_df"]

            similar = nearest_cases(last_input, X, df, feature_stats)

            st.markdown("#### Casos mais semelhantes na base")

            st.dataframe(similar, width="stretch", hide_index=True)



            st.download_button(

                "Baixar dados da predição",

                data=last_input.to_csv(index=False).encode("utf-8"),

                file_name="predicao_cancer_mama.csv",

                mime="text/csv",

                width="stretch",

            )



    if "last_input_df" in st.session_state:

        st.subheader("Comparação do caso informado com a base")

        compare_feature = st.selectbox(

            "Variavel para comparar",

            top_features,

            key="compare_feature_prediction",

        )

        fig = make_distribution_plot(df, compare_feature)

        ax = fig.axes[0]

        ax.axvline(

            st.session_state["last_input_df"][compare_feature].iloc[0],

            color="#7a1f4b",

            linestyle="--",

            linewidth=2,

            label="Caso informado",

        )

        ax.legend()

        st.pyplot(fig, width="stretch")



        with st.expander("Ver todas as variáveis enviadas ao modelo"):

            st.dataframe(st.session_state["last_input_df"], width="stretch")





if menu_option == "Análise exploratória":

    st.subheader("Análise exploratória")

    st.write(

        "Explore distribuições, outliers e correlações. Esses gráficos ajudam a entender por que certas variáveis"

        " são mais informativas para diferenciar tumores benignos e malignos."

    )



    col_a, col_b = st.columns([0.35, 0.65])

    with col_a:

        selected_feature = st.selectbox("Variável analisada", feature_names, index=feature_names.index("worst perimeter"))

        corr_features = st.multiselect(

            "Variáveis para correlação",

            feature_names,

            default=top_features[:6],

        )

    with col_b:

        stats_table = df.groupby("diagnosis")[selected_feature].agg(["count", "mean", "median", "std", "min", "max"])

        stats_table.index = stats_table.index.map(diagnosis_label)

        st.dataframe(stats_table.round(4), width="stretch")



    chart_left, chart_right = st.columns(2)

    with chart_left:

        st.pyplot(make_distribution_plot(df, selected_feature), width="stretch")

    with chart_right:

        st.pyplot(make_boxplot(df, selected_feature), width="stretch")



    if len(corr_features) >= 2:

        st.pyplot(make_corr_plot(df, corr_features), width="stretch")

    else:

        st.info("Selecione pelo menos duas variáveis para visualizar correlação.")





if menu_option == "Modelo":

    st.subheader("Análise do modelo")



    model_cols = st.columns(4)

    model_cols[0].metric("Modelo", "RandomForest")

    model_cols[1].metric("Features selecionadas", model.named_steps["selector"].k)

    model_cols[2].metric("Precisão final", f"{final_metrics['Precisao']:.2%}")

    model_cols[3].metric("F1 final", f"{final_metrics['F1']:.2%}")



    left, right = st.columns([0.55, 0.45])

    with left:

        st.pyplot(make_metrics_plot(metrics), width="stretch")

    with right:

        st.pyplot(make_importance_plot(feature_importance), width="stretch")



    st.markdown("#### Leitura clínica dos erros")

    st.write(

        "O recall mede quantos casos malignos foram recuperados pelo modelo. Em oncologia, falsos negativos"

        " merecem atenção especial porque podem atrasar investigação e tratamento. A precisão também é relevante,"

        " pois falsos positivos podem gerar ansiedade e exames adicionais."

    )



    st.markdown("#### Tabela consolidada")

    st.dataframe(

        metrics.round(4)

        .replace({"Conjunto": {"Teste final apos retreino": "Teste final após retreino"}})

        .rename(columns={"Acuracia": "Acurácia", "Precisao": "Precisão"}),

        width="stretch",

        hide_index=True,

    )





if menu_option == "Dados":

    st.subheader("Dataset utilizado")



    data_cols = st.columns(4)

    data_cols[0].metric("Linhas", len(df))

    data_cols[1].metric("Colunas", len(df.columns))

    data_cols[2].metric("Benignos", int((df["diagnosis"] == "B").sum()))

    data_cols[3].metric("Malignos", int((df["diagnosis"] == "M").sum()))



    st.dataframe(df, width="stretch", hide_index=True)

    st.download_button(

        "Baixar dataset",

        data=df.to_csv(index=False).encode("utf-8"),

        file_name="dataset_cancer_mama.csv",

        mime="text/csv",

    )



    st.markdown(

        '<p class="small-note">Aplicação acadêmica baseada no Breast Cancer Wisconsin. O resultado não deve ser usado como diagnóstico médico.</p>',

        unsafe_allow_html=True,

    )

