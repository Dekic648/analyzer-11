
import streamlit as st
import pandas as pd
from backend import streamlit_survey_app
import matplotlib.pyplot as plt

st.set_page_config(page_title="Survey Analyzer", layout="wide")
st.title("📊 Survey Insights Dashboard")

uploaded_file = st.file_uploader("Upload your survey CSV", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded!")

    st.markdown("### 🔍 Optional Filters")
    filter_cols = st.multiselect("Select Segments to filter by", df.columns)
    for col in filter_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 30:
            selected_vals = st.multiselect(f"Filter {col}", options=unique_vals, key=f"filter_{col}")
            if selected_vals:
                df = df[df[col].isin(selected_vals)]

    st.markdown("### 🧠 Main Summary (Auto-Generated)")
    summary = streamlit_survey_app.generate_basic_summary(df)
    for line in summary:
        st.markdown(line)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        mean_val = df[col].mean()
        st.markdown(f"**{col}** — Average: {mean_val:.2f}")
        fig, ax = plt.subplots(figsize=(6, 3))
        bar = ax.bar([col], [mean_val], color=['#4C78A8'])
        ax.bar_label(bar, fmt='%.2f')
        ax.set_ylim(0, 5)
        st.pyplot(fig)

    st.markdown("### ☁️ WordCloud from Text Responses")
    wordcloud_img, source_col = streamlit_survey_app.generate_wordcloud(df)
    if wordcloud_img:
        st.markdown(f"WordCloud generated from column: **{source_col}**")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_img, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No suitable text column found for wordcloud.")

    st.markdown("### 📈 Segment Comparison (Numeric Metric)")
    segment = st.selectbox("Segment by", df.columns, key="seg_numeric")
    metric = st.selectbox("Numeric Metric", numeric_cols, key="metric_numeric")
    if st.button("Run Numeric Segment Analysis"):
        results = streamlit_survey_app.perform_segment_analysis(df, segment, metric)
        if "error" in results:
            st.error(results["error"])
        else:
            st.dataframe(results["segment_means"])
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(results["segment_means"][segment], results["segment_means"]["mean"], color='#FFA07A')
            ax.bar_label(bars, fmt='%.2f')
            st.pyplot(fig)

    st.markdown("### 🧩 Checkbox Segment Analysis (Multiple Response Questions)")
    checkbox_groups = {}
    for col in df.columns:
        if '_' in col and df[col].dropna().isin([0, 1]).all():
            prefix = col.split('_')[0]
            checkbox_groups.setdefault(prefix, []).append(col)

    if checkbox_groups:
        question_group = st.selectbox("Select Question Group", list(checkbox_groups.keys()))
        seg2 = st.selectbox("Group by segment", df.columns, key="check_seg")

        if st.button("Analyze Checkbox Question Group"):
            cols = checkbox_groups[question_group]
            segment_results = []
            for col in cols:
                temp = streamlit_survey_app.analyze_checkbox_by_segment(df, col, seg2)
                if "checkbox_segment" in temp:
                    df_seg = temp["checkbox_segment"]
                    df_seg["option"] = col
                    segment_results.append(df_seg)

            if segment_results:
                full_df = pd.concat(segment_results)
                full_df["label"] = full_df[seg2].astype(str) + " | " + full_df["option"]
                st.dataframe(full_df[["label", "percentage_selected"]])
                fig, ax = plt.subplots(figsize=(12, 5))
                bars = ax.bar(full_df["label"], full_df["percentage_selected"], color="#86BBD8")
                ax.set_ylabel("% selected")
                ax.set_xticklabels(full_df["label"], rotation=45, ha="right")
                ax.bar_label(bars, fmt='%.1f')
                st.pyplot(fig)
            else:
                st.info("No data found for this question group.")

    st.markdown("### 🧪 Advanced Statistical Analysis")
    if st.button("Run Advanced Analysis"):
        adv_results = streamlit_survey_app.run_advanced_analysis(df)
        if adv_results:
            for title, table in adv_results:
                with st.expander(f"📌 {title}"):
                    st.dataframe(table)
                    if "p-value" in table.columns and table["p-value"].iloc[0] < 0.05:
                        st.markdown(f"**Interpretation:** There is a statistically significant difference in {title.split(':')[0]}.")
                    elif "correlation" in title.lower():
                        st.markdown("**Interpretation:** Review correlation strengths.")
                    elif "coefficient" in table.columns:
                        st.markdown("**Interpretation:** Higher coefficients indicate stronger influence on the target.")
        else:
            st.info("Not enough data for advanced statistical analysis.")

    with st.expander("👤 Segment Overview"):
        segment_choice = st.selectbox("View breakdown of", df.columns, key="seg_overview")
        segment_stats = streamlit_survey_app.segment_summary(df, segment_choice)
        if "error" in segment_stats:
            st.error(segment_stats["error"])
        else:
            st.dataframe(segment_stats["segment_summary"])
