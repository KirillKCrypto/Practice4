# -*- coding: utf-8 -*-
"""
Практическая работа №4: исследование временного ряда UN Tourism SDG 8.9.1.
Скрипт читает исходный Excel-файл, формирует временной ряд для Malaysia за 2008-2023 гг.,
рассчитывает статистики, проверяет стационарность, выделяет компоненты ряда и строит графики.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


ROOT = Path('python/mnt/data')
INPUT_XLSX = "mnt/data/UN_Tourism_8_9_1_TDGDP_04_2025.xlsx"
OUT_DIR = ROOT / 'practice4_outputs'
PLOT_DIR = ROOT / 'practice4_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


COUNTRY = 'Malaysia'
SERIES_CODE = 'ST_GDP_ZS'


def load_series() -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, sheet_name='SDG 8.9.1')
    df = df[df['SeriesCode'].eq(SERIES_CODE)].copy()
    country_df = df[df['GeoAreaName'].eq(COUNTRY)].copy()
    country_df = country_df.sort_values('TimePeriod')
    country_df = country_df[['GeoAreaCode', 'GeoAreaName', 'TimePeriod', 'Value', 'Units', 'SeriesDescription', 'Source', 'FootNote']]
    country_df['Value'] = country_df['Value'].astype(float)
    country_df = country_df.reset_index(drop=True)
    if len(country_df) < 15:
        raise ValueError(f'Временной ряд для {COUNTRY} содержит меньше 15 уровней: {len(country_df)}')
    return country_df


def metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    residuals = y_true - y_pred
    n = len(y_true)
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - sse / sst if sst else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / y_true)) * 100)
    return {'R2': r2, 'Adj_R2': adj_r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SSE': sse}


def fit_models(ts: pd.Series) -> tuple[pd.DataFrame, dict]:
    years = ts.index.to_numpy(dtype=float)
    t = np.arange(1, len(ts) + 1, dtype=float)
    y = ts.to_numpy(dtype=float)

    # Линейная модель: y = a + b*t
    lin_coef = np.polyfit(t, y, 1)
    lin_pred = np.polyval(lin_coef, t)
    # Квадратичная модель: y = a + b*t + c*t^2
    quad_coef = np.polyfit(t, y, 2)
    quad_pred = np.polyval(quad_coef, t)
    # Экспоненциальная модель: ln(y) = a + b*t => y = exp(a)*exp(b*t)
    positive = y > 0
    exp_coef = np.polyfit(t[positive], np.log(y[positive]), 1)
    exp_pred = np.exp(np.polyval(exp_coef, t))

    model_rows = []
    model_rows.append({'model': 'Линейная', 'equation': f'y = {lin_coef[1]:.4f} + {lin_coef[0]:.4f}·t', **metrics(y, lin_pred, 1)})
    model_rows.append({'model': 'Квадратичная', 'equation': f'y = {quad_coef[2]:.4f} + {quad_coef[1]:.4f}·t + {quad_coef[0]:.4f}·t²', **metrics(y, quad_pred, 2)})
    model_rows.append({'model': 'Экспоненциальная', 'equation': f'y = {math.exp(exp_coef[1]):.4f}·e^({exp_coef[0]:.4f}·t)', **metrics(y, exp_pred, 1)})
    fitted = pd.DataFrame({
        'year': years.astype(int),
        'actual': y,
        'linear': lin_pred,
        'quadratic': quad_pred,
        'exponential': exp_pred,
        'residual_quadratic': y - quad_pred,
    })
    return pd.DataFrame(model_rows), {'linear': lin_coef.tolist(), 'quadratic': quad_coef.tolist(), 'exponential_log': exp_coef.tolist(), 'fitted': fitted}


def main() -> None:
    country_df = load_series()
    country_df.to_csv(OUT_DIR / 'prepared_malaysia_sdg_8_9_1.csv', index=False, encoding='utf-8-sig')
    ts = country_df.set_index('TimePeriod')['Value'].astype(float)

    desc = ts.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    # Скользящие средние и экспоненциальное сглаживание
    ma3 = ts.rolling(window=3, center=True).mean()
    ma5 = ts.rolling(window=5, center=True).mean()
    ewm_alpha_03 = ts.ewm(alpha=0.3, adjust=False).mean()

    # Автокорреляции и стационарность
    acf_values = acf(ts, nlags=8, fft=False)
    adf_result = adfuller(ts, autolag='AIC')
    lb = acorr_ljungbox(ts, lags=[4], return_df=True)

    # Декомпозиция: для годового ряда используется условный период 4 года.
    # Календарной сезонности в годовых данных нет, поэтому компонент трактуется как квазипериодические отклонения.
    decomp = seasonal_decompose(ts, model='additive', period=4, extrapolate_trend='freq')

    # Модели тренда
    models_df, fitted_info = fit_models(ts)
    fitted = fitted_info['fitted']
    best_model_name = models_df.sort_values(['RMSE', 'R2'], ascending=[True, False]).iloc[0]['model']

    # Аномальные уровни: по стандартизованным остаткам лучшей квадратичной модели и по IQR исходного ряда.
    residuals = fitted['residual_quadratic'].to_numpy()
    z = (residuals - residuals.mean()) / residuals.std(ddof=1)
    q1, q3 = np.percentile(ts, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    anomalies = []
    for year, value, zscore in zip(ts.index, ts.values, z):
        flags = []
        if abs(zscore) > 2:
            flags.append('остаток |z| > 2')
        if value < lower or value > upper:
            flags.append('правило IQR')
        if flags:
            anomalies.append({'year': int(year), 'value': float(value), 'z_residual': float(zscore), 'criterion': ', '.join(flags)})

    # Остаточная компонента: по модели с лучшим RMSE среди трендов возьмём квадратичную отдельно как гибкую модель.
    res_shapiro = stats.shapiro(residuals)
    dw = durbin_watson(residuals)

    # Таблицы для отчёта
    analysis_table = pd.DataFrame({
        'year': ts.index.astype(int),
        'value': ts.values,
        'ma3_centered': ma3.values,
        'ma5_centered': ma5.values,
        'ewm_alpha_0_3': ewm_alpha_03.values,
        'trend_decomp': decomp.trend.values,
        'seasonal_decomp': decomp.seasonal.values,
        'resid_decomp': decomp.resid.values,
        'quadratic_fit': fitted['quadratic'].values,
        'quadratic_residual': residuals,
        'residual_z': z,
    })
    analysis_table.to_csv(OUT_DIR / 'analysis_results_malaysia.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'lag': np.arange(len(acf_values)), 'acf': acf_values}).to_csv(OUT_DIR / 'autocorrelation.csv', index=False, encoding='utf-8-sig')
    models_df.to_csv(OUT_DIR / 'trend_models.csv', index=False, encoding='utf-8-sig')

    # Графики
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 160

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ts.index, ts.values, marker='o', label='Исходный ряд')
    ax.set_title('SDG 8.9.1: Malaysia, 2008-2023')
    ax.set_xlabel('Год')
    ax.set_ylabel('Доля прямого вклада туризма в ВВП, %')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '01_time_series_line.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(ts.index.astype(str), ts.values)
    ax.set_title('Наглядное представление уровней ряда')
    ax.set_xlabel('Год')
    ax.set_ylabel('Значение, %')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '02_time_series_bar.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.stem(np.arange(len(acf_values)), acf_values)
    conf = 1.96 / np.sqrt(len(ts))
    ax.axhline(conf, linestyle='--', linewidth=1)
    ax.axhline(-conf, linestyle='--', linewidth=1)
    ax.axhline(0, linewidth=0.8)
    ax.set_title('Коррелограмма временного ряда')
    ax.set_xlabel('Лаг')
    ax.set_ylabel('Автокорреляция')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '03_correlogram.png', bbox_inches='tight')
    plt.close(fig)

    comp = pd.DataFrame({
        'Наблюдения': ts,
        'Тренд': decomp.trend,
        'Условная сезонная компонента': decomp.seasonal,
        'Остаток': decomp.resid,
    })
    for col, fname, title, ylabel in [
        ('Тренд', '04_decomposition_trend.png', 'Трендовая компонента ряда', 'Значение, %'),
        ('Условная сезонная компонента', '05_decomposition_seasonal.png', 'Условная сезонная компонента при периоде 4 года', 'Отклонение, п.п.'),
        ('Остаток', '06_decomposition_residual.png', 'Остаточная компонента декомпозиции', 'Отклонение, п.п.'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(comp.index, comp[col], marker='o')
        ax.axhline(0, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('Год')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / fname, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ts.index, ts.values, marker='o', label='Исходный ряд')
    ax.plot(ma3.index, ma3.values, marker='s', label='Скользящая средняя, окно 3')
    ax.plot(ma5.index, ma5.values, marker='^', label='Скользящая средняя, окно 5')
    ax.plot(ewm_alpha_03.index, ewm_alpha_03.values, marker='d', label='Экспоненциальное сглаживание, α=0,3')
    ax.set_title('Сравнение методов сглаживания')
    ax.set_xlabel('Год')
    ax.set_ylabel('Значение, %')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '07_smoothing_comparison.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ts.index, ts.values, marker='o', label='Фактические значения')
    ax.plot(fitted['year'], fitted['linear'], label='Линейная модель')
    ax.plot(fitted['year'], fitted['quadratic'], label='Квадратичная модель')
    ax.plot(fitted['year'], fitted['exponential'], label='Экспоненциальная модель')
    ax.set_title('Сравнение линий тренда')
    ax.set_xlabel('Год')
    ax.set_ylabel('Значение, %')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '08_trend_models.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(fitted['year'].astype(str), residuals)
    ax.axhline(0, linewidth=0.8)
    ax.set_title('Остатки квадратичной модели тренда')
    ax.set_xlabel('Год')
    ax.set_ylabel('Остаток, п.п.')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '09_quadratic_residuals.png', bbox_inches='tight')
    plt.close(fig)

    summary = {
        'country': COUNTRY,
        'n_levels': int(len(ts)),
        'years': [int(ts.index.min()), int(ts.index.max())],
        'description': {k: float(v) for k, v in desc.items()},
        'acf_lag_1': float(acf_values[1]),
        'acf_lag_2': float(acf_values[2]),
        'acf_lag_3': float(acf_values[3]),
        'adf_statistic': float(adf_result[0]),
        'adf_pvalue': float(adf_result[1]),
        'adf_used_lag': int(adf_result[2]),
        'ljung_box_lag4_stat': float(lb['lb_stat'].iloc[0]),
        'ljung_box_lag4_pvalue': float(lb['lb_pvalue'].iloc[0]),
        'models': models_df.to_dict(orient='records'),
        'best_model_by_rmse': best_model_name,
        'anomalies': anomalies,
        'residual_shapiro_W': float(res_shapiro.statistic),
        'residual_shapiro_p': float(res_shapiro.pvalue),
        'durbin_watson_residuals': float(dw),
        'seasonal_period': 4,
        'source_url': 'https://www.unwto.org/tourism-statistics/tourism-statistics-database',
    }
    with open(OUT_DIR / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
