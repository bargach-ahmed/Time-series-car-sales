from pathlib import Path
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ventes_voitures_quebec.txt"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_series():
    df = pd.read_csv(DATA, sep="\t")
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month").asfreq("MS")
    return df["Sales"].astype(float)


def stationarity_row(name, series):
    clean = series.dropna()
    adf = adfuller(clean, autolag="AIC")
    try:
        kpss_res = kpss(clean, regression="c", nlags="auto")
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    return {
        "serie": name,
        "adf_stat": adf[0],
        "adf_pvalue": adf[1],
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
    }


def make_figures(y, train, test, result, pred, ci_low, ci_up, future, future_ci):
    fig, ax = plt.subplots()
    ax.plot(y.index, y, color="steelblue", linewidth=2)
    ax.set_title("Ventes mensuelles de voitures au Quebec (1960-1968)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes")
    fig.tight_layout()
    fig.savefig(FIG / "serie_complete.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(train.index, train, label="Apprentissage", color="steelblue")
    ax.plot(test.index, test, label="Validation", color="tomato")
    ax.axvline(test.index[0], color="gray", linestyle=":")
    ax.set_title("Decoupage apprentissage / validation")
    ax.set_ylabel("Ventes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "split_train_test.png", dpi=160)
    plt.close(fig)

    monthly = pd.DataFrame({"Sales": y})
    monthly["annee"] = monthly.index.year
    monthly["mois"] = monthly.index.month
    fig, ax = plt.subplots()
    for year, group in monthly.groupby("annee"):
        ax.plot(group["mois"], group["Sales"], marker="o", alpha=0.55, label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_title("Profil saisonnier par annee")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Ventes")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "saisonnalite.png", dpi=160)
    plt.close(fig)

    dec = seasonal_decompose(y, model="multiplicative", period=12)
    fig = dec.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Decomposition multiplicative", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "decomposition.png", dpi=160)
    plt.close(fig)

    log_y = np.log(y)
    transformed = pd.DataFrame({
        "log(Y_t)": log_y,
        "Delta log(Y_t)": log_y.diff(),
        "Delta_12 log(Y_t)": log_y.diff(12),
        "Delta Delta_12 log(Y_t)": log_y.diff().diff(12),
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, col in zip(axes.ravel(), transformed.columns):
        ax.plot(transformed.index, transformed[col], color="steelblue")
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(FIG / "transformations.png", dpi=160)
    plt.close(fig)

    stationary_full = np.log(y).diff().diff(12).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_acf(stationary_full, lags=36, ax=axes[0])
    plot_pacf(stationary_full, lags=36, ax=axes[1], method="ywm")
    axes[0].set_title("ACF de la serie stationnaire (K = 36)")
    axes[1].set_title("PACF de la serie stationnaire (K = 36)")
    fig.tight_layout()
    fig.savefig(FIG / "acf_pacf.png", dpi=160)
    plt.close(fig)

    fig = result.plot_diagnostics(figsize=(12, 8))
    fig.suptitle("Diagnostics des residus du modele SARIMA", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "diagnostics_residus.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(train.index, train, label="Apprentissage", color="steelblue")
    ax.plot(test.index, test, label="Reel validation", color="tomato", linewidth=2)
    ax.plot(pred.index, pred, label="Prevision SARIMA", color="green", linestyle="--")
    ax.fill_between(pred.index, ci_low, ci_up, color="green", alpha=0.15, label="IC 95%")
    ax.axvline(test.index[0], color="gray", linestyle=":")
    ax.set_title("Validation des previsions")
    ax.set_ylabel("Ventes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "prevision_validation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(y.index, y, label="Observe", color="steelblue")
    ax.plot(future.index, future, label="Prevision 1969", color="green", linestyle="--")
    ax.fill_between(
        future.index,
        future_ci.iloc[:, 0],
        future_ci.iloc[:, 1],
        color="green",
        alpha=0.15,
        label="IC 95%",
    )
    ax.set_title("Prevision finale sur 12 mois")
    ax.set_ylabel("Ventes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "prevision_1969.png", dpi=160)
    plt.close(fig)


def main():
    y = load_series()
    train = y.loc[:"1966-12"]
    test = y.loc["1967-01":]

    stationarity = [
        stationarity_row("Serie brute", train),
        stationarity_row("log(serie)", np.log(train)),
        stationarity_row("Delta log(serie)", np.log(train).diff()),
        stationarity_row("Delta Delta_12 log(serie)", np.log(train).diff().diff(12)),
    ]
    pd.DataFrame(stationarity).to_csv(OUT / "stationarity_tests.csv", index=False)

    records = []
    log_train = np.log(train)
    for p, q, p_season, q_season in itertools.product(range(4), range(4), range(2), range(2)):
        try:
            model = SARIMAX(
                log_train,
                order=(p, 1, q),
                seasonal_order=(p_season, 1, q_season, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=300)
            lb = acorr_ljungbox(fitted.resid.dropna(), lags=[12, 24], return_df=True)
            forecast = fitted.get_forecast(steps=len(test))
            pred = np.exp(forecast.predicted_mean)
            pred.index = test.index
            records.append({
                "modele": f"SARIMA({p},1,{q})({p_season},1,{q_season})[12]",
                "p": p,
                "q": q,
                "P": p_season,
                "Q": q_season,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                "LjungBox_p_12": lb["lb_pvalue"].iloc[0],
                "LjungBox_p_24": lb["lb_pvalue"].iloc[1],
                "MAE_validation": mean_absolute_error(test, pred),
                "RMSE_validation": mean_squared_error(test, pred) ** 0.5,
                "MAPE_validation": np.mean(np.abs((test - pred) / test)) * 100,
            })
        except Exception:
            continue

    selection = pd.DataFrame(records).sort_values("AIC").reset_index(drop=True)
    selection.to_csv(OUT / "model_selection.csv", index=False)

    selected_order = (3, 1, 1)
    selected_seasonal = (0, 1, 0, 12)
    selected = SARIMAX(
        log_train,
        order=selected_order,
        seasonal_order=selected_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)

    forecast = selected.get_forecast(steps=len(test))
    pred = np.exp(forecast.predicted_mean)
    ci = np.exp(forecast.conf_int(alpha=0.05))
    pred.index = test.index
    ci.index = test.index
    validation = pd.DataFrame({
        "observe": test,
        "prevision": pred,
        "borne_inf": ci.iloc[:, 0],
        "borne_sup": ci.iloc[:, 1],
        "erreur": test - pred,
    })
    validation.to_csv(OUT / "validation_predictions.csv")

    metrics = pd.DataFrame([{
        "modele_retenu": "SARIMA(3,1,1)(0,1,0)[12]",
        "AIC": selected.aic,
        "BIC": selected.bic,
        "MAE": mean_absolute_error(test, pred),
        "RMSE": mean_squared_error(test, pred) ** 0.5,
        "MAPE": np.mean(np.abs((test - pred) / test)) * 100,
        "LjungBox_p_12": acorr_ljungbox(selected.resid.dropna(), lags=[12], return_df=True)["lb_pvalue"].iloc[0],
        "LjungBox_p_24": acorr_ljungbox(selected.resid.dropna(), lags=[24], return_df=True)["lb_pvalue"].iloc[0],
    }])
    metrics.to_csv(OUT / "validation_metrics.csv", index=False)

    final_model = SARIMAX(
        np.log(y),
        order=selected_order,
        seasonal_order=selected_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)
    future_forecast = final_model.get_forecast(steps=12)
    future = np.exp(future_forecast.predicted_mean)
    future.index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future_ci = np.exp(future_forecast.conf_int(alpha=0.05))
    future_ci.index = future.index
    future_table = pd.DataFrame({
        "prevision": future,
        "borne_inf": future_ci.iloc[:, 0],
        "borne_sup": future_ci.iloc[:, 1],
    })
    future_table.to_csv(OUT / "previsions_1969.csv")

    make_figures(y, train, test, selected, pred, ci.iloc[:, 0], ci.iloc[:, 1], future, future_ci)

    print("Modele retenu: SARIMA(3,1,1)(0,1,0)[12]")
    print(metrics.round(4).to_string(index=False))
    print("\nPrevisions 1969:")
    print(future_table.round(0).to_string())


if __name__ == "__main__":
    main()
