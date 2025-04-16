
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def perform_segment_analysis(df, segment_column, metric_column):
    if segment_column not in df.columns or metric_column not in df.columns:
        return {'error': "Selected column not found in data."}
    try:
        if not pd.api.types.is_numeric_dtype(df[metric_column]):
            df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
        grouped = df.groupby(segment_column)[metric_column].agg(['mean', 'std', 'count']).reset_index()
        grouped['std_error'] = grouped['std'] / np.sqrt(grouped['count'])
        return {'segment_means': grouped}
    except Exception as e:
        return {'error': str(e)}

def generate_basic_summary(df):
    summary = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        summary.append(f"📊 **{col}** — Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}, Count: {df[col].count()}")
    checkbox_cols = [col for col in df.columns if df[col].nunique() <= 2 and df[col].dropna().isin([0, 1]).all()]
    if checkbox_cols:
        summary.append("🧩 **Popular Features Selected:**")
        for col in checkbox_cols:
            percent = (df[col].sum() / len(df)) * 100
            summary.append(f"- {col.replace('_', ' ').title()}: selected by {percent:.1f}%")
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        if df[col].dropna().apply(lambda x: len(str(x).split()) > 3).sum() > 5:
            words = " ".join(df[col].dropna().astype(str)).lower().split()
            top_words = Counter(words).most_common(5)
            summary.append(f"📝 **Top words in {col}:** " + ", ".join([w for w, _ in top_words]))
            break
    return summary

def run_advanced_analysis(df):
    output = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(2)
        output.append(("Correlation Matrix", corr_matrix))
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 5:
                for num_col in numeric_cols:
                    try:
                        groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            output.append((f"ANOVA: {num_col} ~ {cat_col}", pd.DataFrame({
                                'F-statistic': [f_stat],
                                'p-value': [p_val]
                            })))
                    except:
                        continue
        if len(numeric_cols) >= 2:
            y = df[numeric_cols[0]].dropna()
            X = df[numeric_cols[1:]].dropna()
            X = X.loc[y.index]
            if len(X) > 5:
                model = LinearRegression()
                model.fit(X, y)
                coef_df = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': model.coef_
                })
                output.append(("Linear Regression", coef_df))
        if len(numeric_cols) >= 2:
            try:
                km_model = KMeans(n_clusters=2, random_state=42, n_init="auto")
                km_model.fit(df[numeric_cols].dropna())
                df['Cluster'] = km_model.labels_
                cluster_means = df.groupby('Cluster')[numeric_cols].mean()
                output.append(("KMeans Clustering: Cluster Means", cluster_means))
            except:
                pass
    return output

def analyze_checkbox_by_segment(df, checkbox_col, segment_col):
    if checkbox_col not in df.columns or segment_col not in df.columns:
        return {'error': 'Invalid column selection'}
    try:
        group_counts = df.groupby(segment_col)[checkbox_col].agg(['sum', 'count'])
        group_counts['percentage_selected'] = (group_counts['sum'] / group_counts['count']) * 100
        return {'checkbox_segment': group_counts.reset_index()}
    except Exception as e:
        return {'error': str(e)}

def generate_wordcloud(df):
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        if df[col].dropna().apply(lambda x: len(str(x).split()) > 3).sum() > 5:
            full_text = " ".join(df[col].dropna().astype(str)).lower()
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_text)
            return wordcloud, col
    return None, None

def segment_summary(df, segment_col):
    if segment_col not in df.columns:
        return {'error': 'Segment column not found'}
    try:
        summary = df[segment_col].value_counts(dropna=False).reset_index()
        summary.columns = [segment_col, 'count']
        return {'segment_summary': summary}
    except Exception as e:
        return {'error': str(e)}
