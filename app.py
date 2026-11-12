"""
SojaPredict GO — Servidor Flask
Serve a LP + API completa para todos os recursos Python.

Rotas:
  GET  /                    → LP (index.html)
  POST /api/upload          → Upload de CSV/XLSX
  GET  /api/dataset         → Carrega dataset padrão
  POST /api/treinar         → Treina modelos ML
  POST /api/prever          → Previsão por inputs climáticos
  GET  /api/estatisticas    → Análise estatística completa (ANOVA, DM, Tukey, KS, VIF...)
  GET  /api/vif             → VIF de multicolinearidade
  GET  /api/grafico/<tipo>  → Gera gráficos (base64 PNG)
  GET  /api/exportar        → Exporta resultados CSV
  GET  /api/modelos/status  → Status dos modelos treinados

Autor: Gabriel Sá da Silveira | UEG Eng. Agrícola 2025/2026
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io, os, json, base64, traceback
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Adiciona o diretório raiz ao path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.ml_models import (
    treinar_todos, tabela_comparativa, XGBOOST_OK, validacao_cruzada_folds,
)
from utils.preprocessor import (
    carregar_dataset, gerar_dados_sinteticos, get_X_y,
    FEATURES_MODELO, resumo_outliers,
)
from utils.statistical_analysis import (
    estatisticas_descritivas,
    teste_normalidade_shapiro, teste_normalidade_ks,
    teste_homogeneidade_levene, teste_homogeneidade_bartlett,
    anova_entre_modelos, tukey_hsd,
    kruskal_wallis,
    diebold_mariano_todos_pares,
    calcular_vif,
    correlacao_pearson_spearman, matriz_correlacao,
)

# ── Config ─────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data"
UPL    = BASE / "uploads"
SAVED  = BASE / "models_saved"
for d in [DATA, UPL, SAVED]:
    d.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

import json

class SafeEncoder(json.JSONEncoder):
    """Converte tipos numpy/bool que o Flask não serializa."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = SafeEncoder

# Estado em memória (para a sessão do servidor)
STATE = {
    "df":           None,
    "resultados":   None,
    "y_test":       None,
    "fn":           None,
    "X_test":       None,
    "X_train":      None,
    "y_train":      None,
    "test_size":    0.2,
    "seed":         42,
    "cv_fold_data": None,   # por-fold CV para boxplot e testes estatísticos
    "tuning":       False,
}

# ── Helpers ────────────────────────────────────────────────
CORES = {
    "Regressão Linear Múltipla": "#3B9EFF",
    "Random Forest":             "#2DBF5A",
    "XGBoost":                   "#FFD600",
    "Rede Neural (MLP)":         "#9B51E0",
}
LABELS_IMP = {
    "temp_media_anual_C":         "Temp. média (°C)",
    "temp_max_media_anual_C":     "Temp. máx. (°C)",
    "temp_min_media_anual_C":     "Temp. mín. (°C)",
    "precipitacao_total_mm":      "Precipitação (mm)",
    "umidade_relativa_media_pct": "Umidade (%)",
    "amplitude_termica_C":        "Amplitude térm.",
    "pet_anual_mm":               "PET (mm/ano)",
    "etr_anual_mm":               "ETR (mm/ano)",
    "def_hidrico_mm":             "Déficit hídr. (mm)",
    "exc_hidrico_mm":             "Excedente hídr. (mm)",
    "arm_final_mm":               "ARM solo (mm)",
    "razao_etr_pet":              "ETR/PET",
    "indice_aridez":              "Índice aridez",
    "chuva_por_grau":             "Chuva/grau",
}

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def safe_json(obj):
    """Converte tipos numpy/bool para tipos nativos do Python."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def fig_base(w=9, h=4.5, n=1, dark=False):
    bg  = "#0D1A10" if dark else "#FFFFFF"
    bg2 = "#122018" if dark else "#F8FCF8"
    tc  = "#C8E6CB" if dark else "#1C2B1C"
    gc  = "#1E3A22" if dark else "#D0EDD8"
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor(bg)
    axs = [axes] if n == 1 else list(axes)
    for ax in axs:
        ax.set_facecolor(bg2)
        ax.tick_params(colors=tc, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color(gc)
    return fig, axes, tc, gc

def style_ax(ax, title="", xlabel="", ylabel="", tc="#1C2B1C", gc="#D0EDD8"):
    ax.set_title(title, color=tc, fontsize=11, fontweight="700", pad=10)
    ax.set_xlabel(xlabel, color=tc, fontsize=9)
    ax.set_ylabel(ylabel, color=tc, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(gc)
    ax.grid(True, alpha=0.25, color=gc, linestyle="--")
    ax.tick_params(colors=tc)

def err(msg, code=400):
    return jsonify({"ok": False, "erro": msg}), code

# ══════════════════════════════════════════════════════════
# ROTA PRINCIPAL — Serve a LP
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

# ══════════════════════════════════════════════════════════
# API — UPLOAD DE ARQUIVO
# ══════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return err("Nenhum arquivo enviado.")
    f = request.files["file"]
    if f.filename == "":
        return err("Nome de arquivo vazio.")

    ext = Path(f.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        return err("Formato não suportado. Use CSV ou XLSX.")

    try:
        if ext == ".csv":
            # Tenta diferentes encodings
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(f, encoding=enc)
                    break
                except Exception:
                    f.seek(0)
        else:
            df = pd.read_excel(f)

        # Normaliza nomes de colunas
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

        # Verifica se tem coluna de produtividade
        prod_cols = [c for c in df.columns if "produtiv" in c or "rendimento" in c or "kg_ha" in c]
        if not prod_cols:
            return err("Arquivo não contém coluna de produtividade (kg/ha, rendimento, produtividade).")

        # Renomeia para padrão se necessário
        if "produtividade_kg_ha" not in df.columns and prod_cols:
            df = df.rename(columns={prod_cols[0]: "produtividade_kg_ha"})

        from utils.preprocessor import _normalizar_colunas
        df = _normalizar_colunas(df)
        STATE["df"] = df
        dest = UPL / f.filename
        df.to_csv(dest, index=False, encoding="utf-8-sig")

        return jsonify({
            "ok": True,
            "arquivo": f.filename,
            "registros": len(df),
            "colunas": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
            "stats": {
                "n_municipios": int(df["municipio"].nunique()) if "municipio" in df.columns else None,
                "n_anos": int(df["ano"].nunique()) if "ano" in df.columns else None,
                "produt_media": round(float(df["produtividade_kg_ha"].mean()), 1),
                "produt_min":   round(float(df["produtividade_kg_ha"].min()), 1),
                "produt_max":   round(float(df["produtividade_kg_ha"].max()), 1),
            }
        })
    except Exception as e:
        return err(f"Erro ao ler arquivo: {str(e)}")

# ══════════════════════════════════════════════════════════
# API — CARREGAR DATASET PADRÃO
# ══════════════════════════════════════════════════════════
@app.route("/api/dataset")
def dataset():
    modo = request.args.get("modo", "csv")
    try:
        if modo == "demo":
            df = gerar_dados_sinteticos()
        else:
            df = carregar_dataset(DATA)
            if df.empty:
                df = gerar_dados_sinteticos()

        # Normaliza coluna municipio
        for col in ["municipio_nome", "municipio"]:
            if col in df.columns and "municipio" not in df.columns:
                df = df.rename(columns={col: "municipio"})

        STATE["df"] = df
        col_mun = "municipio" if "municipio" in df.columns else (
                  "municipio_nome" if "municipio_nome" in df.columns else None)

        return jsonify({
            "ok": True,
            "registros": len(df),
            "colunas": list(df.columns),
            "n_municipios": int(df[col_mun].nunique()) if col_mun else None,
            "n_anos": int(df["ano"].nunique()) if "ano" in df.columns else None,
            "produt_media": round(float(df["produtividade_kg_ha"].mean()), 1),
            "features_disponiveis": [f for f in FEATURES_MODELO if f in df.columns],
            "preview": df.head(8).fillna("").to_dict(orient="records"),
        })
    except Exception as e:
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — TREINAR MODELOS
# ══════════════════════════════════════════════════════════
@app.route("/api/treinar", methods=["POST"])
def treinar():
    if STATE["df"] is None:
        return err("Carregue os dados primeiro (/api/dataset ou /api/upload).")

    body        = request.get_json(silent=True) or {}
    modelos_sel = body.get("modelos", ["Regressão Linear", "Random Forest", "XGBoost", "MLP"])
    test_size   = float(body.get("test_size", 0.2))
    seed        = int(body.get("seed", 42))
    tuning      = bool(body.get("tuning", False))
    cv_folds    = int(body.get("cv_folds", 10))
    STATE["test_size"] = test_size
    STATE["seed"]      = seed
    STATE["tuning"]    = tuning

    try:
        df = STATE["df"]
        X, y, fn = get_X_y(df)
        if len(X) < 20:
            return err("Dados insuficientes (mínimo 20 registros).")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        resultados = treinar_todos(Xtr, ytr, Xte, yte, fn, modelos_sel,
                                   tuning=tuning, cv_folds=cv_folds)

        STATE["resultados"] = resultados
        STATE["y_test"]     = yte
        STATE["y_train"]    = ytr
        STATE["X_test"]     = Xte
        STATE["X_train"]    = Xtr
        STATE["fn"]         = fn

        df_c   = tabela_comparativa(resultados)
        melhor = df_c.iloc[0]

        # ── Validação cruzada k-fold por modelo (armazena por-fold) ──────
        # Usa Pipelines completos para evitar data-leakage
        cv_fold_data = validacao_cruzada_folds(
            X, y, fn, modelos_sel,
            n_splits=cv_folds, seed=seed,
        )
        STATE["cv_fold_data"] = cv_fold_data

        # Monta resumo de métricas com CV
        metricas = []
        best_params_map = {}
        for nome, r in resultados.items():
            m = r["metricas"].copy()
            cv_info = cv_fold_data.get(nome, {})
            m["cv_r2_mean"]   = cv_info.get("r2_mean")
            m["cv_r2_std"]    = cv_info.get("r2_std")
            m["cv_rmse_mean"] = cv_info.get("rmse_mean")
            m["cv_mae_mean"]  = cv_info.get("mae_mean")
            metricas.append(m)
            best_params_map[nome] = r.get("best_params", {})

        return jsonify(safe_json({
            "ok":               True,
            "n_treino":         int(len(ytr)),
            "n_teste":          int(len(yte)),
            "features":         fn,
            "modelos_treinados": list(resultados.keys()),
            "melhor_modelo":    str(melhor["modelo"]),
            "melhor_r2":        float(melhor["R²"]),
            "metricas":         metricas,
            "cv_fold_data":     {
                nome: {
                    "r2_por_fold":   d.get("fold_r2", []),
                    "rmse_por_fold": d.get("fold_rmse", []),
                    "r2_mean":       d.get("r2_mean"),
                    "r2_std":        d.get("r2_std"),
                    "rmse_mean":     d.get("rmse_mean"),
                }
                for nome, d in cv_fold_data.items()
            },
            "best_params":       best_params_map,
            "tuning_ativado":    tuning,
        }))
    except Exception as e:
        traceback.print_exc()
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — PREVISÃO INTERATIVA
# ══════════════════════════════════════════════════════════
@app.route("/api/prever", methods=["POST"])
def prever():
    if STATE["resultados"] is None:
        return err("Treine os modelos primeiro (/api/treinar).")

    body = request.get_json(silent=True) or {}
    fn   = STATE["fn"]

    try:
        tm     = float(body.get("temp_media", 23.5))
        tx     = float(body.get("temp_max",   29.0))
        tn     = float(body.get("temp_min",   17.5))
        precip = float(body.get("precipitacao", 1400))
        umid   = float(body.get("umidade",     65.0))

        amp = abs(tx - tn)
        # PET Hargreaves-Samani com Ra médio Goiás (FAO-56) — igual ao dataset
        Ra_mm_d = 37.5 * 0.408
        pet = 0.0023 * (tm + 17.8) * ((max(tx - tn, 0)) ** 0.5) * Ra_mm_d * 365
        ia  = precip / pet if pet > 0 else 0
        cpg = precip / tm  if tm  > 0 else 0

        # Balanço hídrico Thornthwaite-Mather (CAD = 100 mm, padrão soja)
        CAD = 100.0
        P, EP = max(precip, 0), max(pet, 0)
        if P >= EP:
            _exc = P - EP
            _arm = min(CAD, _exc)
            etr  = round(EP, 1)
            arm  = round(_arm, 1)
            defh = 0.0
            exc  = round(max(_exc - _arm, 0), 1)
        else:
            _arm = CAD * np.exp((P - EP) / CAD)
            _arm = max(0, min(CAD, _arm))
            etr  = round(min(P + (CAD - _arm), EP), 1)
            arm  = round(_arm, 1)
            defh = round(EP - etr, 1)
            exc  = 0.0
        razao = round(etr / pet, 3) if pet > 0 else 0.0

        inp = {
            "temp_media_anual_C":         tm,
            "temp_max_media_anual_C":     tx,
            "temp_min_media_anual_C":     tn,
            "precipitacao_total_mm":      precip,
            "umidade_relativa_media_pct": umid,
            "amplitude_termica_C":        round(amp, 2),
            "pet_anual_mm":               round(pet, 1),
            "etr_anual_mm":               etr,
            "arm_final_mm":               arm,
            "def_hidrico_mm":             defh,
            "exc_hidrico_mm":             exc,
            "razao_etr_pet":              razao,
            "indice_aridez":              round(ia, 3),
            "chuva_por_grau":             round(cpg, 2),
        }

        X_inp = np.array([[inp.get(f, 0) for f in fn]])
        previsoes = {}
        for nome, r in STATE["resultados"].items():
            try:
                pred = float(r["modelo"].prever(X_inp)[0])
                previsoes[nome] = {
                    "kg_ha": round(pred),
                    "sc_ha": round(pred / 60, 1),
                }
            except Exception as e:
                previsoes[nome] = {"erro": str(e)}

        media = np.mean([v["kg_ha"] for v in previsoes.values() if "kg_ha" in v])
        return jsonify({
            "ok":          True,
            "inputs":      inp,
            "derivadas":   {"amplitude_C": round(amp,2), "pet_mm": round(pet,1), "indice_aridez": round(ia,3)},
            "previsoes":   previsoes,
            "media_kg_ha": round(media),
            "media_sc_ha": round(media / 60, 1),
            "alertas": {
                "precipitacao": "ok" if 1200 <= precip <= 1800 else "atencao",
                "temperatura":  "ok" if 20    <= tm    <= 26   else "atencao",
                "umidade":      "ok" if umid  >= 55            else "atencao",
            }
        })
    except Exception as e:
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — ANÁLISE ESTATÍSTICA COMPLETA (dissertação seção 4.7)
# ══════════════════════════════════════════════════════════
@app.route("/api/estatisticas")
def estatisticas():
    if STATE["df"] is None:
        return err("Carregue os dados primeiro.")

    df       = STATE["df"]
    vars_num = [v for v in FEATURES_MODELO + ["produtividade_kg_ha"] if v in df.columns]

    try:
        # ── 1. Estatísticas descritivas ───────────────────────────────
        desc = [estatisticas_descritivas(df[v], v) for v in vars_num]

        # ── 2. Outliers ───────────────────────────────────────────────
        feats_num = [v for v in vars_num if v != "produtividade_kg_ha"]
        outliers  = resumo_outliers(df, feats_num)

        # ── 3. Correlações Pearson e Spearman ─────────────────────────
        corrs = [
            correlacao_pearson_spearman(df, v, "produtividade_kg_ha")
            for v in feats_num
        ]

        # ── 4. VIF — multicolinearidade ───────────────────────────────
        feats_vif = [v for v in feats_num if df[v].notna().sum() > 5]
        vif = calcular_vif(df, feats_vif) if len(feats_vif) >= 2 else []

        # ── 5. Normalidade dos resíduos + testes sobre modelos ────────
        shapiro_resultados = []
        ks_resultados      = []
        levene_result      = None
        bartlett_result    = None
        anova_result       = None
        kruskal_result     = None
        dm_comparacoes     = []
        diag_linear        = None

        if STATE["resultados"] and STATE["y_test"] is not None:
            yte      = STATE["y_test"]
            res      = STATE["resultados"]
            res_dict = {}   # erros absolutos por modelo (para ANOVA/KW)

            for nome, r in res.items():
                residuos   = yte - r["y_pred"]
                erros_abs  = np.abs(residuos)
                res_dict[nome] = erros_abs

                sw = teste_normalidade_shapiro(pd.Series(residuos))
                sw["modelo"] = nome
                shapiro_resultados.append(sw)

                ks = teste_normalidade_ks(pd.Series(residuos))
                ks["modelo"] = nome
                ks_resultados.append(ks)

            # Diagnóstico de premissas do modelo linear (T1)
            nome_linear = "Regressão Linear Múltipla"
            if nome_linear in res and STATE["X_test"] is not None:
                try:
                    diag_linear = res[nome_linear]["modelo"].diagnostico_residuos(
                        STATE["X_test"], yte
                    )
                except Exception:
                    pass

            # Homogeneidade de variâncias (pré-requisito ANOVA)
            grupos_erros = list(res_dict.values())
            levene_result   = teste_homogeneidade_levene(*grupos_erros)
            bartlett_result = teste_homogeneidade_bartlett(*grupos_erros)

            # ANOVA paramétrica + Tukey HSD
            anova_result = anova_entre_modelos(res_dict)

            # Kruskal-Wallis (não-paramétrico) + Mann-Whitney
            kruskal_result = kruskal_wallis(res_dict)

            # Diebold-Mariano par-a-par
            dm_comparacoes = diebold_mariano_todos_pares(res, yte)

        return jsonify(safe_json({
            "ok":                  True,
            "descritivas":         desc,
            "outliers":            outliers,
            "correlacoes":         corrs,
            "vif":                 vif,
            "shapiro_wilk":        shapiro_resultados,
            "kolmogorov_smirnov":  ks_resultados,
            "levene":              levene_result,
            "bartlett":            bartlett_result,
            "anova":               anova_result,
            "kruskal_wallis":      kruskal_result,
            "diebold_mariano":     dm_comparacoes,
            "diagnostico_linear":  diag_linear,
        }))
    except Exception as e:
        traceback.print_exc()
        return err(str(e))


# ══════════════════════════════════════════════════════════
# API — VIF (endpoint dedicado)
# ══════════════════════════════════════════════════════════
@app.route("/api/vif")
def vif_endpoint():
    if STATE["df"] is None:
        return err("Carregue os dados primeiro.")
    df       = STATE["df"]
    features = [v for v in FEATURES_MODELO if v in df.columns]
    try:
        resultado = calcular_vif(df, features)
        severas   = [r for r in resultado if r.get("VIF") and r["VIF"] > 10]
        return jsonify(safe_json({
            "ok":            True,
            "vif":           resultado,
            "n_features":    len(features),
            "n_severas":     len(severas),
            "features_severas": [r["feature"] for r in severas],
            "interpretação": "VIF > 10 indica multicolinearidade severa (dissertação seção 4.3)",
        }))
    except Exception as e:
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — GERADOR DE GRÁFICOS
# ══════════════════════════════════════════════════════════
@app.route("/api/grafico/<tipo>")
def grafico(tipo):
    dark = request.args.get("dark", "0") == "1"
    bg   = "#0D1A10" if dark else "#FFFFFF"

    try:
        # ── distribuição ─────────────────────────────────
        if tipo == "distribuicao":
            if STATE["df"] is None:
                return err("Sem dados.")
            df = STATE["df"]
            fig, ax, tc, gc = fig_base(8, 4.5, dark=dark)
            ax.hist(df["produtividade_kg_ha"].dropna(), bins=28,
                    color="#2DBF5A", alpha=0.82, edgecolor=bg, linewidth=0.6)
            ax.axvline(df["produtividade_kg_ha"].mean(),
                       color="#FFD600", ls="--", lw=2.2,
                       label=f"Média: {df['produtividade_kg_ha'].mean():,.0f} kg/ha")
            ax.axvline(df["produtividade_kg_ha"].median(),
                       color="#3B9EFF", ls=":", lw=2.2,
                       label=f"Mediana: {df['produtividade_kg_ha'].median():,.0f} kg/ha")
            ax.legend(facecolor=bg, edgecolor=gc, labelcolor=tc, fontsize=9)
            style_ax(ax, "Distribuição da Produtividade", "Produtividade (kg/ha)", "Frequência", tc, gc)
            plt.tight_layout(pad=1.2)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── dispersão precip × produtividade ─────────────
        elif tipo == "dispersao":
            if STATE["df"] is None:
                return err("Sem dados.")
            df = STATE["df"]
            if "precipitacao_total_mm" not in df.columns:
                return err("Coluna precipitacao_total_mm não encontrada.")
            fig, ax, tc, gc = fig_base(8, 4.5, dark=dark)
            sc = ax.scatter(df["precipitacao_total_mm"], df["produtividade_kg_ha"],
                            c=df["ano"] if "ano" in df.columns else "#2DBF5A",
                            cmap="Greens", alpha=0.6, s=32, edgecolors="white", linewidths=0.4)
            if "ano" in df.columns:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.ax.tick_params(colors=tc, labelsize=7.5)
                cbar.set_label("Ano", color=tc, fontsize=8)
            style_ax(ax, "Precipitação × Produtividade", "Precipitação (mm)", "Produtividade (kg/ha)", tc, gc)
            plt.tight_layout(pad=1.2)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── boxplot regiões ───────────────────────────────
        elif tipo == "boxplot":
            if STATE["df"] is None:
                return err("Sem dados.")
            df = STATE["df"]
            if "regiao" not in df.columns:
                return err("Coluna 'regiao' não encontrada.")
            regioes = sorted(df["regiao"].unique())
            fig, ax, tc, gc = fig_base(10, 4.5, dark=dark)
            data_b = [df[df["regiao"]==r]["produtividade_kg_ha"].dropna().values for r in regioes]
            cores_b = ["#1A8C3E","#2DBF5A","#3B9EFF","#9B51E0","#FFD600"]
            bp = ax.boxplot(data_b, labels=regioes, patch_artist=True, widths=0.5,
                            medianprops=dict(color="#FF6B2B",linewidth=2.5),
                            whiskerprops=dict(color=gc,linewidth=1.2),
                            capprops=dict(color=gc,linewidth=1.2),
                            flierprops=dict(marker="o",markerfacecolor=gc,markersize=3.5,alpha=0.5))
            for patch, cor in zip(bp["boxes"], cores_b[:len(regioes)]):
                patch.set_facecolor(cor); patch.set_alpha(0.55); patch.set_edgecolor(gc)
            style_ax(ax, "Produtividade por Região Produtora", "", "Produtividade (kg/ha)", tc, gc)
            ax.tick_params(axis="x", colors=tc, labelsize=9.5)
            plt.tight_layout(pad=1.2)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── heatmap correlação ────────────────────────────
        elif tipo == "correlacao":
            if STATE["df"] is None:
                return err("Sem dados.")
            df = STATE["df"]
            vars_num = [v for v in FEATURES_MODELO + ["produtividade_kg_ha"] if v in df.columns]
            LABELS = {"produtividade_kg_ha":"Produtividade","temp_media_anual_C":"Temp. média",
                      "temp_max_media_anual_C":"Temp. máx.","temp_min_media_anual_C":"Temp. mín.",
                      "precipitacao_total_mm":"Precipitação","umidade_relativa_media_pct":"Umidade",
                      "pet_anual_mm":"PET","etr_anual_mm":"ETR","def_hidrico_mm":"Déficit hídr.",
                      "exc_hidrico_mm":"Excedente","arm_final_mm":"ARM solo","razao_etr_pet":"ETR/PET",
                      "indice_aridez":"Aridez","amplitude_termica_C":"Amplitude","chuva_por_grau":"Chuva/grau"}
            df_corr = matriz_correlacao(df[vars_num])
            fig, ax = plt.subplots(figsize=(10, 6.5))
            fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
            mask = np.triu(np.ones_like(df_corr, dtype=bool))
            lbls = [LABELS.get(v, v) for v in df_corr.columns]
            cmap = sns.diverging_palette(10, 145, s=85, l=40, as_cmap=True)
            tc2  = "#C8E6CB" if dark else "#1C2B1C"
            sns.heatmap(df_corr, annot=True, fmt=".2f", cmap=cmap, center=0, mask=mask, ax=ax,
                        linewidths=0.6, linecolor="#1E3A22" if dark else "#D0EDD8",
                        xticklabels=lbls, yticklabels=lbls,
                        annot_kws={"size":9,"color":tc2,"weight":"600"},
                        cbar_kws={"shrink":0.8})
            ax.tick_params(colors=tc2, labelsize=9)
            ax.collections[0].colorbar.ax.tick_params(colors=tc2, labelsize=8)
            plt.xticks(rotation=30, ha="right", color=tc2)
            plt.yticks(rotation=0, color=tc2)
            plt.tight_layout(pad=1.2)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── métricas comparativas ─────────────────────────
        elif tipo == "metricas":
            if STATE["resultados"] is None:
                return err("Treine os modelos primeiro.")
            res   = STATE["resultados"]
            nomes = list(res.keys())
            cores = [CORES.get(n, "#888") for n in nomes]
            fig, axes, tc, gc = fig_base(13, 4.5, 3, dark=dark)
            axs = list(axes)
            for i, (met, fmt) in enumerate([("R²",".3f"),("MAE (kg/ha)",".0f"),("RMSE (kg/ha)",".0f")]):
                ax = axs[i]
                vals = [res[n]["metricas"][met] for n in nomes]
                bars = ax.bar(nomes, vals, color=cores, edgecolor="none", width=0.55, alpha=0.9)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.015,
                            format(v, fmt), ha="center", va="bottom",
                            fontsize=9.5, fontweight="700", color=tc)
                ax.set_xticklabels(nomes, rotation=18, ha="right", fontsize=8.5, color=tc)
                style_ax(ax, met, tc=tc, gc=gc)
            plt.tight_layout(pad=1.5)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── real vs previsto ──────────────────────────────
        elif tipo == "real_previsto":
            if STATE["resultados"] is None:
                return err("Treine os modelos primeiro.")
            res  = STATE["resultados"]
            yte  = STATE["y_test"]
            n    = len(res)
            fig, axes, tc, gc = fig_base(4.8*n, 4.8, n, dark=dark)
            axs = [axes] if n == 1 else list(axes)
            for ax, (nome, r) in zip(axs, res.items()):
                cor = CORES.get(nome, "#888")
                ax.scatter(yte, r["y_pred"], color=cor, alpha=0.55, s=32,
                           edgecolors="white", linewidths=0.4)
                lim = [min(yte.min(), r["y_pred"].min())*.93,
                       max(yte.max(), r["y_pred"].max())*1.05]
                ax.plot(lim, lim, "--", color=gc, alpha=0.7, lw=1.5)
                r2 = r["metricas"]["R²"]
                style_ax(ax, f"{nome}\nR²={r2:.3f}", "Real (kg/ha)", "Previsto (kg/ha)", tc, gc)
                ax.set_xlim(lim); ax.set_ylim(lim)
            plt.tight_layout(pad=1.5)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── importância de variáveis ──────────────────────
        elif tipo == "importancia":
            if STATE["resultados"] is None:
                return err("Treine os modelos primeiro.")
            res  = STATE["resultados"]
            n    = len(res)
            fig, axes, tc, gc = fig_base(4.8*n, 4.8, n, dark=dark)
            axs = [axes] if n == 1 else list(axes)
            for ax, (nome, r) in zip(axs, res.items()):
                df_i = r["importancia"].head(9)
                if df_i.empty: continue
                lbls = [LABELS_IMP.get(f, f) for f in df_i["feature"]]
                cor  = CORES.get(nome, "#888")
                ax.barh(lbls[::-1], df_i["importancia"].values[::-1],
                        color=cor, alpha=0.88, edgecolor="none", height=0.65)
                for i, v in enumerate(df_i["importancia"].values[::-1]):
                    ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=8, color=tc)
                style_ax(ax, nome, "Importância (%)", tc=tc, gc=gc)
                ax.tick_params(axis="y", labelsize=9, colors=tc)
            plt.tight_layout(pad=1.5)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── resíduos ──────────────────────────────────────
        elif tipo == "residuos":
            if STATE["resultados"] is None:
                return err("Treine os modelos primeiro.")
            nome_mod = request.args.get("modelo", list(STATE["resultados"].keys())[0])
            if nome_mod not in STATE["resultados"]:
                return err(f"Modelo '{nome_mod}' não encontrado.")
            yte = STATE["y_test"]
            r   = STATE["resultados"][nome_mod]
            res = yte - r["y_pred"]
            cor = CORES.get(nome_mod, "#2DBF5A")
            fig, axes, tc, gc = fig_base(12, 4.5, 3, dark=dark)
            axs = list(axes)
            # Histograma
            axs[0].hist(res, bins=22, color=cor, alpha=0.8, edgecolor=bg, linewidth=0.5)
            axs[0].axvline(0, color="#FFD600", lw=2, ls="--", alpha=0.85)
            style_ax(axs[0], "Histograma dos Resíduos", "Resíduo (kg/ha)", "Frequência", tc, gc)
            # Q-Q
            from scipy import stats as spstats
            (osm, osr), (slope, intercept, _) = spstats.probplot(res, dist="norm")
            axs[1].scatter(osm, osr, color=cor, alpha=0.55, s=22, edgecolors="none")
            axs[1].plot(osm, slope*np.array(osm)+intercept, color="#FFD600", lw=2.2, alpha=0.85)
            style_ax(axs[1], "Q-Q Plot", "Quantis teóricos", "Quantis observados", tc, gc)
            # Resíduos vs Predito
            axs[2].scatter(r["y_pred"], res, color=cor, alpha=0.5, s=22, edgecolors="none")
            axs[2].axhline(0, color="#FFD600", lw=1.8, ls="--", alpha=0.8)
            axs[2].axhline(2*res.std(), color="#FF6B2B", lw=1, ls=":", alpha=0.5)
            axs[2].axhline(-2*res.std(), color="#FF6B2B", lw=1, ls=":", alpha=0.5)
            style_ax(axs[2], "Resíduos × Predito", "Previsto (kg/ha)", "Resíduo", tc, gc)
            plt.tight_layout(pad=1.5)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── evolução temporal ─────────────────────────────
        elif tipo == "evolucao":
            if STATE["df"] is None:
                return err("Sem dados.")
            df = STATE["df"]
            if "ano" not in df.columns:
                return err("Coluna 'ano' não encontrada.")
            fig, ax, tc, gc = fig_base(10, 4.5, dark=dark)
            if "regiao" in df.columns:
                cores_reg = ["#1A8C3E","#3B9EFF","#FF6B2B","#9B51E0","#FFD600"]
                for i, reg in enumerate(sorted(df["regiao"].unique())):
                    med = df[df["regiao"]==reg].groupby("ano")["produtividade_kg_ha"].mean()
                    cor = cores_reg[i % len(cores_reg)]
                    ax.plot(med.index, med.values, lw=2.2, marker="o", ms=5.5,
                            label=f"Região {reg}", color=cor,
                            markerfacecolor="white", markeredgewidth=2)
                ax.legend(facecolor=bg, edgecolor=gc, labelcolor=tc, fontsize=8.5, ncol=3)
            else:
                med = df.groupby("ano")["produtividade_kg_ha"].mean()
                ax.fill_between(med.index, med.values, alpha=0.15, color="#2DBF5A")
                ax.plot(med.index, med.values, color="#2DBF5A", lw=2.5,
                        marker="o", ms=6.5, markerfacecolor="white", markeredgewidth=2.5)
                z = np.polyfit(med.index, med.values, 1)
                ax.plot(med.index, np.poly1d(z)(med.index), "--", color="#FFD600",
                        alpha=0.7, lw=1.8, label="Tendência")
                ax.legend(facecolor=bg, edgecolor=gc, labelcolor=tc, fontsize=9)
            style_ax(ax, "Evolução Temporal da Produtividade", "Ano-safra", "Produtividade (kg/ha)", tc, gc)
            plt.tight_layout(pad=1.2)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        # ── boxplot CV por fold ──────────────────────────────
        elif tipo == "cv_boxplot":
            if STATE["cv_fold_data"] is None:
                return err("Treine os modelos primeiro (CV por fold não disponível).")
            cv_data = STATE["cv_fold_data"]
            nomes   = list(cv_data.keys())
            if not nomes:
                return err("Nenhum dado de CV disponível.")

            fig, axes, tc, gc = fig_base(13, 4.5, 2, dark=dark)
            axs = list(axes)
            cores = [CORES.get(n, "#888") for n in nomes]

            # Subplot 1: RMSE por fold (boxplot)
            data_rmse = [cv_data[n].get("fold_rmse", []) for n in nomes]
            bp1 = axs[0].boxplot(
                [d for d in data_rmse if d],
                labels=[n for n, d in zip(nomes, data_rmse) if d],
                patch_artist=True, widths=0.55,
                medianprops=dict(color="#FFD600", linewidth=2.5),
                whiskerprops=dict(color=gc, linewidth=1.2),
                capprops=dict(color=gc, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=gc, markersize=3.5, alpha=0.5),
            )
            for patch, cor in zip(bp1["boxes"], [c for c, d in zip(cores, data_rmse) if d]):
                patch.set_facecolor(cor); patch.set_alpha(0.6); patch.set_edgecolor(gc)
            axs[0].set_xticklabels(
                [n for n, d in zip(nomes, data_rmse) if d],
                rotation=18, ha="right", fontsize=8.5, color=tc,
            )
            style_ax(axs[0], "RMSE por Fold — Validação Cruzada", "", "RMSE (kg/ha)", tc, gc)

            # Subplot 2: R² por fold (boxplot)
            data_r2 = [cv_data[n].get("fold_r2", []) for n in nomes]
            bp2 = axs[1].boxplot(
                [d for d in data_r2 if d],
                labels=[n for n, d in zip(nomes, data_r2) if d],
                patch_artist=True, widths=0.55,
                medianprops=dict(color="#FFD600", linewidth=2.5),
                whiskerprops=dict(color=gc, linewidth=1.2),
                capprops=dict(color=gc, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=gc, markersize=3.5, alpha=0.5),
            )
            for patch, cor in zip(bp2["boxes"], [c for c, d in zip(cores, data_r2) if d]):
                patch.set_facecolor(cor); patch.set_alpha(0.6); patch.set_edgecolor(gc)
            axs[1].set_xticklabels(
                [n for n, d in zip(nomes, data_r2) if d],
                rotation=18, ha="right", fontsize=8.5, color=tc,
            )
            style_ax(axs[1], "R² por Fold — Validação Cruzada", "", "R²", tc, gc)

            plt.tight_layout(pad=1.5)
            return jsonify({"ok": True, "imagem": fig_to_b64(fig)})

        else:
            return err(f"Tipo de gráfico '{tipo}' não reconhecido.")

    except Exception as e:
        traceback.print_exc()
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — EXPORTAR RESULTADOS
# ══════════════════════════════════════════════════════════
@app.route("/api/exportar")
def exportar():
    tipo = request.args.get("tipo", "dataset")
    try:
        if tipo == "dataset":
            if STATE["df"] is None:
                return err("Sem dados.")
            buf = io.BytesIO()
            STATE["df"].to_csv(buf, index=False, encoding="utf-8-sig")
            buf.seek(0)
            return send_file(buf, mimetype="text/csv",
                             as_attachment=True, download_name="dataset_soja_go.csv")

        elif tipo == "metricas":
            if STATE["resultados"] is None:
                return err("Sem modelos treinados.")
            df_c = tabela_comparativa(STATE["resultados"])
            buf = io.BytesIO()
            df_c.to_csv(buf, index=False, encoding="utf-8-sig")
            buf.seek(0)
            return send_file(buf, mimetype="text/csv",
                             as_attachment=True, download_name="metricas_modelos.csv")

        elif tipo == "previsoes":
            if STATE["resultados"] is None or STATE["y_test"] is None:
                return err("Sem previsões disponíveis.")
            rows = []
            for nome, r in STATE["resultados"].items():
                for real, prev in zip(STATE["y_test"], r["y_pred"]):
                    rows.append({"modelo": nome, "real": round(float(real)),
                                 "previsto": round(float(prev)), "residuo": round(float(real-prev))})
            df_prev = pd.DataFrame(rows)
            buf = io.BytesIO()
            df_prev.to_csv(buf, index=False, encoding="utf-8-sig")
            buf.seek(0)
            return send_file(buf, mimetype="text/csv",
                             as_attachment=True, download_name="previsoes_teste.csv")
        else:
            return err("Tipo de exportação inválido.")
    except Exception as e:
        return err(str(e))

# ══════════════════════════════════════════════════════════
# API — STATUS DOS MODELOS
# ══════════════════════════════════════════════════════════
@app.route("/api/modelos/status")
def status():
    return jsonify({
        "ok":             True,
        "dados_carregados": STATE["df"] is not None,
        "n_registros":    len(STATE["df"]) if STATE["df"] is not None else 0,
        "modelos_treinados": list(STATE["resultados"].keys()) if STATE["resultados"] else [],
        "xgboost_disponivel": XGBOOST_OK,
        "features":       STATE["fn"],
        "test_size":      STATE["test_size"],
    })

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🌱 SojaPredict GO — Servidor Flask")
    print("   Acesse: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
