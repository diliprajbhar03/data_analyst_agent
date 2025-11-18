# agent_core.py
import pandas as pd
import numpy as np
from jinja2 import Template
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import base64

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    from statsmodels.tsa.arima.model import ARIMA

def load_data(file):
    if isinstance(file, str):
        if file.lower().endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    else:
        name = getattr(file, 'name', 'uploaded.csv')
        if name.lower().endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

def profile_data(df, top_n=20):
    # ✨ 1. Prevent crashes — handle empty
    if df is None or df.empty:
        return {
            "error": "No data uploaded. Please upload a CSV/Excel file first."
        }

    summary = {}

    # ✨ 2. Basic info
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    # ✨ 3. Missing values
    summary["missing"] = (
        df.isnull().sum().sort_values(ascending=False).head(top_n).to_dict()
    )

    # ✨ 4. Numeric summary (safe)
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        summary["numeric_summary"] = num.describe().T.to_dict()
    else:
        summary["numeric_summary"] = "No numeric columns found."

    # ✨ 5. Unique values
    summary["n_uniques"] = {col: int(df[col].nunique()) for col in df.columns}

    return summary


def cleaning_suggestions(df):
    suggestions = []
    n = len(df)
    miss = df.isnull().sum()
    high_missing = miss[miss > 0.3 * n]
    for col, cnt in high_missing.items():
        suggestions.append({'column': col, 'issue': 'high_missing', 'suggestion': 'Consider dropping or investigating column (>{:.0f}% missing)'.format(cnt/len(df)*100)})
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() <= 20:
            suggestions.append({'column': col, 'issue': 'categorical_small_cardinality', 'suggestion': 'Encode with one-hot or target encoding depending on model'})
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) > 0:
            q1, q3 = np.percentile(series, [25,75])
            iqr = q3 - q1
            outliers = series[(series < q1 - 3*iqr) | (series > q3 + 3*iqr)]
            if len(outliers) > 0.01 * len(series):
                suggestions.append({'column': col, 'issue': 'outliers', 'suggestion': f'{len(outliers)} extreme values detected — consider winsorizing or log-transform'})
    return suggestions

def train_model(df, target, task='regression', random_state=42, n_iter=25):
    if target not in df.columns:
        raise ValueError('target not in dataframe')
    X = df.drop(columns=[target])
    y = df[target]
    X_proc = pd.get_dummies(X, drop_first=True).fillna(0)
    if X_proc.shape[1] == 0:
        raise ValueError('No features after encoding')
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=random_state)
    if task == 'regression':
        base = RandomForestRegressor(random_state=random_state)
        param_dist = {'n_estimators': [100,200,400], 'max_depth': [None,5,10,20], 'min_samples_split': [2,5,10]}
    else:
        base = RandomForestClassifier(random_state=random_state)
        param_dist = {'n_estimators': [100,200,400], 'max_depth': [None,5,10,20], 'min_samples_split': [2,5,10]}
    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter, cv=3, random_state=random_state, n_jobs=-1)
    rs.fit(X_train, y_train)
    model = rs.best_estimator_
    preds = model.predict(X_test)
    if task == 'regression':
        metrics = {'RMSE': float(mean_squared_error(y_test, preds, squared=False)), 'R2': float(r2_score(y_test, preds))}
    else:
        metrics = {'Accuracy': float(accuracy_score(y_test, preds))}
    fi = sorted(zip(X_proc.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:50]
    return {'model': model, 'metrics': metrics, 'feature_importances': fi, 'columns': list(X_proc.columns)}

def forecast_series(df, date_col, value_col, periods=12):
    df2 = df[[date_col, value_col]].dropna()
    df2[date_col] = pd.to_datetime(df2[date_col])
    df2 = df2.sort_values(date_col)
    ds = df2[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    if HAS_PROPHET:
        m = Prophet()
        m.fit(ds)
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)
        return forecast[['ds','yhat','yhat_lower','yhat_upper']]
    else:
        ts = ds.set_index('ds').resample('M').mean().interpolate()
        model = ARIMA(ts['y'], order=(1,1,1))
        res = model.fit()
        pred = res.get_forecast(steps=periods)
        index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq='M')
        out = pd.DataFrame({'ds': index, 'yhat': pred.predicted_mean, 'yhat_lower': pred.conf_int().iloc[:,0], 'yhat_upper': pred.conf_int().iloc[:,1]})
        return out

REPORT_TEMPLATE = '''<html><head><meta charset="utf-8"><title>Agent Report</title></head><body>
<h1>Data Agent Report</h1>
<p>Generated: {{timestamp}}</p>
<p>Rows: {{rows}} | Columns: {{cols}}</p>
<h3>Missing</h3>
<table border="1"><tr><th>Column</th><th>Missing</th></tr>{% for c,m in missing %}<tr><td>{{c}}</td><td>{{m}}</td></tr>{% endfor %}</table>
{% if metrics %}
<h3>Model Metrics</h3>
<ul>{% for k,v in metrics.items() %}<li>{{k}}: {{v}}</li>{% endfor %}</ul>
<h3>Top Features</h3>
<ol>{% for f,i in feature_importances %}<li>{{f}} - {{'%.4f'|format(i)}}</li>{% endfor %}</ol>
{% endif %}
</body></html>'''

def generate_report(df, model_result=None):
    tpl = Template(REPORT_TEMPLATE)
    html = tpl.render(timestamp=datetime.utcnow().isoformat()+"Z",
                      rows=df.shape[0], cols=df.shape[1],
                      missing=list(df.isnull().sum().items())[:50],
                      metrics=(model_result.get('metrics') if model_result else None),
                      feature_importances=(model_result.get('feature_importances') if model_result else None))
    return html

def html_download_link(html, filename="report.html"):
    b64 = base64.b64encode(html.encode()).decode()
    href = f"data:text/html;base64,{b64}"
    return href, filename
