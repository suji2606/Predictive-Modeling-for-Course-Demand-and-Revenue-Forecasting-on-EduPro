import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro Predictive Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #475569;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #f1f5f9;
        border-left: 4px solid #38bdf8; padding-left: 12px; margin: 20px 0 12px;
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; }
    div[data-testid="stSidebar"] { background-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ─── Load Assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)

    # Train if models don't exist
    model_dir = os.path.join(base, "models")
    if not os.path.exists(os.path.join(model_dir, "metadata.pkl")):
        from model_train import train_and_save
        train_and_save(os.path.join(base, "EduPro_Online_Platform.xlsx"), model_dir)

    with open(os.path.join(model_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(model_dir, "EnrollmentCount_model.pkl"), "rb") as f:
        enroll_bundle = pickle.load(f)
    with open(os.path.join(model_dir, "TotalRevenue_model.pkl"), "rb") as f:
        rev_bundle = pickle.load(f)

    return meta, enroll_bundle, rev_bundle

meta, enroll_bundle, rev_bundle = load_assets()
df_raw       = meta["df_raw"]
results      = meta["results"]
importances  = meta["importances"]
cat_revenue  = meta["cat_revenue"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduPro Analytics")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Overview", "🔮 Demand Predictor", "💰 Revenue Forecast",
                                  "📈 Feature Insights", "🏆 Model Performance"])
    st.markdown("---")
    st.caption("Data: EduPro Platform  \nModels: Ridge · RF · GBM")

# ─── OVERVIEW PAGE ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 EduPro Platform Overview")
    st.markdown("##### Exploratory Data Analysis & Key Metrics")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Courses", f"{len(df_raw)}"),
        ("Total Enrollments", f"{int(df_raw['EnrollmentCount'].sum()):,}"),
        ("Total Revenue", f"${df_raw['TotalRevenue'].sum():,.0f}"),
        ("Avg Course Rating", f"{df_raw['CourseRating'].mean():.2f} ★"),
        ("Paid Courses", f"{(df_raw['CourseType']=='Paid').sum()}"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Enrollments by Category</div>", unsafe_allow_html=True)
        cat_enroll = df_raw.groupby("CourseCategory")["EnrollmentCount"].sum().reset_index()
        cat_enroll.columns = ["Category", "Enrollments"]
        fig = px.bar(cat_enroll.sort_values("Enrollments"), x="Enrollments", y="Category",
                     orientation="h", color="Enrollments",
                     color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Revenue by Category</div>", unsafe_allow_html=True)
        fig2 = px.pie(cat_revenue, values="Revenue", names="Category",
                      hole=0.45, template="plotly_dark",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>Course Price vs. Revenue</div>", unsafe_allow_html=True)
        fig3 = px.scatter(df_raw, x="CoursePrice", y="TotalRevenue",
                          color="CourseCategory", size="EnrollmentCount",
                          hover_data=["CourseName"], template="plotly_dark",
                          labels={"CoursePrice": "Course Price ($)", "TotalRevenue": "Revenue ($)"})
        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Course Rating Distribution</div>", unsafe_allow_html=True)
        fig4 = px.histogram(df_raw, x="CourseRating", nbins=20, template="plotly_dark",
                            color_discrete_sequence=["#38bdf8"],
                            labels={"CourseRating": "Rating"})
        fig4.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

# ─── DEMAND PREDICTOR ─────────────────────────────────────────────────────────
elif page == "🔮 Demand Predictor":
    st.title("🔮 Course Demand Predictor")
    st.markdown("Configure a hypothetical course and predict expected enrollments.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📚 Course Details")
        cat = st.selectbox("Category", sorted(df_raw["CourseCategory"].unique()))
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        ctype = st.selectbox("Type", ["Free", "Paid"])
        price = st.slider("Price ($)", 0, 500, 100, step=10)
        duration = st.slider("Duration (hours)", 5, 60, 20)
        rating = st.slider("Expected Course Rating", 1.0, 5.0, 3.5, step=0.1)

    with col2:
        st.markdown("#### 👨‍🏫 Instructor Details")
        exp = st.slider("Years of Experience", 1, 20, 5)
        t_rating = st.slider("Teacher Rating", 1.0, 5.0, 3.5, step=0.1)
        expertise_match = st.radio("Instructor Expertise Matches Category?", ["Yes", "No"])

    with col3:
        st.markdown("#### 📊 Historical Baseline")
        past_enroll = st.number_input("Past Avg Enrollment (similar courses)", 100, 300, 166)
        past_rev = st.number_input("Past Avg Revenue ($)", 0, 50000, 15000, step=500)
        rev_per_enroll = past_rev / max(past_enroll, 1)
        st.info(f"Revenue/Enrollment: **${rev_per_enroll:.2f}**")

    if st.button("🚀 Predict Enrollment Demand", type="primary", use_container_width=True):
        # Build input vector matching training features
        features = enroll_bundle["features"]
        input_data = pd.Series(0.0, index=features)

        # Numeric fills
        for f, v in [("CoursePrice", price), ("CourseDuration", duration),
                     ("CourseRating", rating), ("YearsOfExperience", exp),
                     ("TeacherRating", t_rating), ("ExpertiseMatch", int(expertise_match=="Yes")),
                     ("EnrollmentCount", past_enroll), ("AvgRevenue", past_rev/max(past_enroll,1)),
                     ("RevenuePerEnrollment", rev_per_enroll)]:
            if f in input_data: input_data[f] = v

        # One-hot fills
        for prefix, val in [("CourseCategory_", cat), ("CourseLevel_", level),
                             ("CourseType_", ctype)]:
            key = prefix + val
            if key in input_data: input_data[key] = 1

        # Price band
        band = "Free" if price == 0 else ("Low" if price <= 150 else ("Medium" if price <= 350 else "High"))
        pb_key = f"PriceBand_{band}"
        if pb_key in input_data: input_data[pb_key] = 1

        # Duration bucket
        db = "Short" if duration <= 15 else ("Medium" if duration <= 30 else ("Long" if duration <= 50 else "Extended"))
        dk = f"DurationBucket_{db}"
        if dk in input_data: input_data[dk] = 1

        # Rating tier
        rt = "Low" if rating <= 2 else ("Medium" if rating <= 3 else ("High" if rating <= 4 else "Top"))
        rk = f"RatingTier_{rt}"
        if rk in input_data: input_data[rk] = 1

        # Exp bucket
        eb = "Junior" if exp <= 3 else ("Mid" if exp <= 7 else ("Senior" if exp <= 12 else "Expert"))
        ek = f"ExpBucket_{eb}"
        if ek in input_data: input_data[ek] = 1

        X_in = enroll_bundle["scaler"].transform(input_data.values.reshape(1, -1))
        pred = max(0, int(enroll_bundle["model"].predict(X_in)[0]))

        st.markdown("---")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0c4a6e, #075985);
                    border-radius:16px; padding:32px; text-align:center; border: 1px solid #0ea5e9'>
            <div style='font-size:1rem; color:#7dd3fc; font-weight:600;'>PREDICTED ENROLLMENTS</div>
            <div style='font-size:3.5rem; font-weight:900; color:#38bdf8'>{pred:,}</div>
            <div style='font-size:0.9rem; color:#bae6fd; margin-top:8px;'>
                Estimated Revenue: <strong>${pred * price:,.0f}</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        # Comparison bar
        st.markdown("<br>", unsafe_allow_html=True)
        bench = df_raw[df_raw["CourseCategory"] == cat]["EnrollmentCount"].mean()
        fig_comp = go.Figure(go.Bar(
            x=["Category Avg", "Your Prediction"],
            y=[bench, pred],
            marker_color=["#475569", "#38bdf8"],
            text=[f"{bench:.0f}", f"{pred}"],
            textposition="outside"
        ))
        fig_comp.update_layout(template="plotly_dark", title=f"vs. {cat} Category Average",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40, b=0))
        st.plotly_chart(fig_comp, use_container_width=True)

# ─── REVENUE FORECAST ─────────────────────────────────────────────────────────
elif page == "💰 Revenue Forecast":
    st.title("💰 Revenue Forecast")

    tab1, tab2 = st.tabs(["📦 Course-Level Forecast", "🗂️ Category-Level Analysis"])

    with tab1:
        st.markdown("#### Predict Total Revenue for a New Course")
        c1, c2 = st.columns(2)
        with c1:
            f_cat = st.selectbox("Category", sorted(df_raw["CourseCategory"].unique()), key="rf_cat")
            f_price = st.slider("Course Price ($)", 0, 500, 150, key="rf_price")
            f_dur = st.slider("Duration (hrs)", 5, 60, 25, key="rf_dur")
            f_rating = st.slider("Course Rating", 1.0, 5.0, 3.8, key="rf_rtng")
            f_level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"], key="rf_lvl")
        with c2:
            f_exp = st.slider("Instructor Experience (yrs)", 1, 20, 6, key="rf_exp")
            f_trating = st.slider("Instructor Rating", 1.0, 5.0, 3.8, key="rf_tr")
            f_em = st.radio("Expertise Match?", ["Yes", "No"], key="rf_em")
            f_enroll = st.number_input("Past Avg Enrollments", 100, 300, 166, key="rf_en")
            f_prev_rev = st.number_input("Past Avg Revenue ($)", 0, 50000, 20000, step=500, key="rf_pr")

        if st.button("💰 Forecast Revenue", type="primary", use_container_width=True):
            features = rev_bundle["features"]
            input_data = pd.Series(0.0, index=features)
            f_rev_per = f_prev_rev / max(f_enroll, 1)

            for f, v in [("CoursePrice", f_price), ("CourseDuration", f_dur),
                         ("CourseRating", f_rating), ("YearsOfExperience", f_exp),
                         ("TeacherRating", f_trating), ("ExpertiseMatch", int(f_em=="Yes")),
                         ("EnrollmentCount", f_enroll), ("AvgRevenue", f_prev_rev/max(f_enroll,1)),
                         ("RevenuePerEnrollment", f_rev_per)]:
                if f in input_data: input_data[f] = v

            for prefix, val in [("CourseCategory_", f_cat), ("CourseLevel_", f_level),
                                 ("CourseType_", "Paid" if f_price > 0 else "Free")]:
                key = prefix + val
                if key in input_data: input_data[key] = 1

            band = "Free" if f_price == 0 else ("Low" if f_price<=150 else ("Medium" if f_price<=350 else "High"))
            if f"PriceBand_{band}" in input_data: input_data[f"PriceBand_{band}"] = 1
            db = "Short" if f_dur<=15 else ("Medium" if f_dur<=30 else ("Long" if f_dur<=50 else "Extended"))
            if f"DurationBucket_{db}" in input_data: input_data[f"DurationBucket_{db}"] = 1
            rt = "Low" if f_rating<=2 else ("Medium" if f_rating<=3 else ("High" if f_rating<=4 else "Top"))
            if f"RatingTier_{rt}" in input_data: input_data[f"RatingTier_{rt}"] = 1
            eb = "Junior" if f_exp<=3 else ("Mid" if f_exp<=7 else ("Senior" if f_exp<=12 else "Expert"))
            if f"ExpBucket_{eb}" in input_data: input_data[f"ExpBucket_{eb}"] = 1

            X_in = rev_bundle["scaler"].transform(input_data.values.reshape(1, -1))
            pred_rev = max(0, rev_bundle["model"].predict(X_in)[0])

            col_a, col_b, col_c = st.columns(3)
            for col, label, val, color in [
                (col_a, "Predicted Revenue", f"${pred_rev:,.0f}", "#38bdf8"),
                (col_b, "Revenue per Student", f"${pred_rev/max(f_enroll,1):,.2f}", "#34d399"),
                (col_c, "vs Category Avg", f"{((pred_rev / cat_revenue[cat_revenue.Category==f_cat]['Revenue'].values[0]) * 100):.1f}%", "#f59e0b"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{color}'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Category Revenue Comparison")
        fig_cat = px.bar(cat_revenue.sort_values("Revenue", ascending=False),
                         x="Category", y="Revenue",
                         color="Revenue", color_continuous_scale="Blues",
                         text_auto=True, template="plotly_dark",
                         labels={"Revenue": "Total Revenue ($)"})
        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10))
        st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("#### Revenue vs. Enrollments by Category")
        cat_stats = df_raw.groupby("CourseCategory").agg(
            Revenue=("TotalRevenue","sum"), Enrollments=("EnrollmentCount","sum"),
            Courses=("CourseID","count"), AvgPrice=("CoursePrice","mean")
        ).reset_index()
        fig_bub = px.scatter(cat_stats, x="Enrollments", y="Revenue",
                             size="Courses", color="AvgPrice",
                             hover_data=["CourseCategory"],
                             text="CourseCategory", template="plotly_dark",
                             color_continuous_scale="Viridis",
                             labels={"AvgPrice":"Avg Price"})
        fig_bub.update_traces(textposition="top center")
        fig_bub.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bub, use_container_width=True)

# ─── FEATURE INSIGHTS ─────────────────────────────────────────────────────────
elif page == "📈 Feature Insights":
    st.title("📈 Feature Importance & Insights")

    target_choice = st.radio("Select Target", ["EnrollmentCount", "TotalRevenue"], horizontal=True)
    imp_df = importances[target_choice]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='section-title'>Top Predictive Features</div>", unsafe_allow_html=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         template="plotly_dark")
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Key Takeaways</div>", unsafe_allow_html=True)
        top3 = imp_df["Feature"].head(3).tolist()
        insights = {
            "EnrollmentCount": [
                "**Past enrollment history** is the strongest signal for future demand.",
                "**Course rating** drives student discovery and trust.",
                "**Instructor experience** signals content quality.",
            ],
            "TotalRevenue": [
                "**Course price** directly dominates revenue generation.",
                "**Revenue per enrollment** captures monetization efficiency.",
                "**Instructor rating** correlates with premium pricing power.",
            ],
        }
        for tip in insights[target_choice]:
            st.success(tip)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>Price vs Revenue by Level</div>", unsafe_allow_html=True)
        fig_pv = px.box(df_raw[df_raw["CoursePrice"]>0], x="CourseLevel", y="TotalRevenue",
                        color="CourseLevel", template="plotly_dark",
                        labels={"TotalRevenue": "Revenue ($)"})
        fig_pv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_pv, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Teacher Rating vs Enrollments</div>", unsafe_allow_html=True)
        fig_tr = px.scatter(df_raw, x="TeacherRating", y="EnrollmentCount",
                            color="CourseCategory",
                            template="plotly_dark", labels={"EnrollmentCount": "Enrollments"})
        # Manual trendline (no statsmodels needed)
        x_vals = df_raw["TeacherRating"].values
        y_vals = df_raw["EnrollmentCount"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_tr.add_trace(go.Scatter(x=x_line, y=m * x_line + b,
                                    mode="lines", name="Trend",
                                    line=dict(color="white", width=2, dash="dash")))
        fig_tr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_tr, use_container_width=True)

# ─── MODEL PERFORMANCE ────────────────────────────────────────────────────────
elif page == "🏆 Model Performance":
    st.title("🏆 Model Evaluation & Comparison")

    for target, label in [("EnrollmentCount", "Enrollment Count"), ("TotalRevenue", "Total Revenue")]:
        st.markdown(f"<div class='section-title'>{label} — Model Leaderboard</div>", unsafe_allow_html=True)
        res = results[target].copy()
        res_sorted = res.sort_values("R2", ascending=False)

        col1, col2 = st.columns([1, 2])
        with col1:
            display_df = res_sorted.copy()
            display_df["MAE"]  = display_df["MAE"].map("{:.3f}".format)
            display_df["RMSE"] = display_df["RMSE"].map("{:.3f}".format)
            display_df["R2"]   = display_df["R2"].map("{:.3f}".format)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        with col2:
            fig_m = px.bar(res_sorted, x="Model", y="R2",
                           color="R2", color_continuous_scale="Blues",
                           text_auto=True, template="plotly_dark",
                           title=f"R² Score Comparison — {label}")
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(t=40,b=0))
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")
