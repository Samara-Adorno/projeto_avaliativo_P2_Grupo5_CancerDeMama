from pathlib import Path
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "model" / "modelo_final.joblib"
METRICS_PATH = ROOT / "reports" / "metricas_consolidadas.csv"
FEATURES_PATH = ROOT / "model" / "feature_names.joblib"
FIGURES_DIR = ROOT / "reports" / "figures"


def ensure_dirs() -> None:
    for path in [DATA_PATH.parent, MODEL_PATH.parent, METRICS_PATH.parent, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_or_create_dataset() -> pd.DataFrame:
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["diagnosis"] = np.where(cancer.target == 0, "M", "B")
    df["Unnamed: 32"] = np.nan
    df.to_csv(DATA_PATH, index=False)
    return df


def metric_row(model_name: str, split: str, y_true, y_pred, y_prob) -> dict:
    return {
        "Modelo": model_name,
        "Conjunto": split,
        "Acuracia": accuracy_score(y_true, y_pred),
        "Precisao": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"],
    )
    plt.title(title)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_or_create_dataset()

    clean_df = df.drop(columns=["id", "Unnamed: 32"])
    clean_df["diagnosis"] = clean_df["diagnosis"].map({"M": 1, "B": 0})

    X = clean_df.drop(columns=["diagnosis"])
    y = clean_df["diagnosis"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )

    models = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("classifier", SVC(probability=True, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        ),
        "NaiveBayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("classifier", GaussianNB()),
            ]
        ),
    }

    rows = []
    fitted_initial_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_initial_models[name] = model
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        rows.append(metric_row(name, "Validacao inicial", y_val, y_pred_val, y_prob_val))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grids = {
        "SVM": (
            models["SVM"],
            {
                "selector__k": [10, 15, 20],
                "classifier__C": [0.1, 1, 10],
                "classifier__gamma": [0.001, 0.01, 0.1],
                "classifier__kernel": ["rbf"],
            },
        ),
        "RandomForest": (
            models["RandomForest"],
            {
                "selector__k": [10, 15, 20],
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5],
            },
        ),
    }

    best_models = {}
    for name, (pipeline, params) in grids.items():
        search = GridSearchCV(
            pipeline,
            params,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        y_pred_val = search.predict(X_val)
        y_prob_val = search.predict_proba(X_val)[:, 1]
        rows.append(metric_row(name, "Validacao otimizada", y_val, y_pred_val, y_prob_val))

        y_pred_test = search.predict(X_test)
        y_prob_test = search.predict_proba(X_test)[:, 1]
        rows.append(metric_row(name, "Teste final", y_test, y_pred_test, y_prob_test))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_PATH, index=False)

    final_row = (
        metrics_df[metrics_df["Conjunto"] == "Validacao otimizada"]
        .sort_values(["F1", "Recall", "AUC"], ascending=False)
        .iloc[0]
    )
    final_model_name = final_row["Modelo"]
    final_model = best_models[final_model_name]

    # O modelo final e salvo depois de ajustar treino+validacao, mantendo teste intocado.
    final_model.fit(X_train_val, y_train_val)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    y_final_test_pred = final_model.predict(X_test)
    y_final_test_prob = final_model.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(
        y_test,
        y_final_test_pred,
        f"Matriz de Confusao - {final_model_name} final",
        "matriz_confusao_modelo_final.png",
    )

    plt.figure(figsize=(7, 5))
    for name, model in best_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    plt.title("Curvas ROC - Teste final")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "curva_roc_teste.png", dpi=160)
    plt.close()

    pca = PCA(n_components=2)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_pca = pca.fit_transform(X_train_scaled)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette=["#2ca25f", "#de2d26"])
    plt.title("PCA dos dados de treino")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend(title="Diagnostico", labels=["Benigno", "Maligno"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_treino.png", dpi=160)
    plt.close()

    selected_mask = final_model.named_steps["selector"].get_support()
    selected_features = X.columns[selected_mask]
    if final_model_name == "RandomForest":
        importances = final_model.named_steps["classifier"].feature_importances_
    else:
        scores = final_model.named_steps["selector"].scores_[selected_mask]
        importances = scores / scores.sum()

    importance_df = (
        pd.DataFrame({"Feature": selected_features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    importance_df.to_csv(ROOT / "reports" / "features_importantes.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", color="#3182bd")
    plt.title("Top 10 features selecionadas")
    plt.xlabel("Importancia relativa")
    plt.ylabel("Caracteristica")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "features_importantes.png", dpi=160)
    plt.close()

    final_metrics = metric_row(
        final_model_name,
        "Teste final apos retreino",
        y_test,
        y_final_test_pred,
        y_final_test_prob,
    )
    metrics_df = pd.concat([metrics_df, pd.DataFrame([final_metrics])], ignore_index=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("Modelo final:", final_model_name)
    print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))
    print("Modelo salvo em:", MODEL_PATH)
    print("Metricas salvas em:", METRICS_PATH)


if __name__ == "__main__":
    main()
