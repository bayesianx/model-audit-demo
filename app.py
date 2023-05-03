import streamlit as st
from functionalities.risks import KDE, SHAPEvaluator
from functionalities.load_data import load_data, fit_model
from functionalities.plot_data import *
from st_pages import Page, show_pages, add_page_title
from PIL import Image

st.set_page_config(layout="wide", 
                   page_title="Model Audit", 
                   page_icon=":chart_with_upwards_trend:")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

st.sidebar.image("./icons/logo.png", use_column_width=True)

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("app.py", "Model Audit", icon=":chart_with_upwards_trend:"),
        Page("pages/page2.py", "Explainability", ":mag:"),
    ]
)

add_page_title()

# load data
df, X_train, X_test, y_train, y_test, model1 = load_data()

####### KDE ############
st.header("High-level risk statement")
st.markdown("""""")

threshold = st.slider(label="Select error threshold",
                      min_value=0,
                      max_value=int(df['mape'].max()*100),
                      step=1)

left_column, right_column = st.columns((2, 1))
with left_column:

    cutoff_value = threshold / 100
    x_end = df['mape'].max()
    kde = KDE(res=df["mape"], cutoff_value=cutoff_value, x_end=x_end)
    kde_plot = kde.plot_prob_density(plotname="Expected error distribution",
                                    bandwidth=0.008,
                                    nbins=20)

    st.pyplot(kde_plot)

with right_column:
    # write number in big
    st.write()

    # get probability
    if threshold == 0:
        prob = 1

    else:
        prob = kde.get_probability()

    st.markdown("""
                    <style>
                    .big-font {
                        font-size:65px !important;
                        text-align: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)
    st.markdown(f'<p class="big-font">{prob:.2%}</p>', unsafe_allow_html=True)

    st.markdown("""
                    <style>
                    .small-font {
                        font-size:20px !important;
                        text-align: center; 
                    }
                    </style>
                    """, unsafe_allow_html=True)
    
    st.write(f'<p class="small-font">Probability of model error exceeding {threshold}%</p>', unsafe_allow_html=True)


####### SHAP ERRORS ############
st.markdown("""---""")
st.header("Feature Impact on Model Error")
st.markdown("""""")

(X_train_proc, y_train_res, reg_model) = fit_model(df, X_train=X_train)

explainer = SHAPEvaluator(X_train_proc, reg_model)
explainer.feature_importance()

left_column, right_column = st.columns(2)
with left_column:
    st.markdown('<h3 class="right-sub">Distribution of Impact on Error by Feature</h3>', unsafe_allow_html=True)
    explainer.summary_plot()

with right_column:
    st.markdown('<h3 class="left-sub">Expected Impact on Error by Feature</h3>', unsafe_allow_html=True)
    explainer.plot_feature_importance(y=y_train_res)

###### HEATMAP ############
display_df = df.drop(columns=["res", "abs_res"], axis=1).reset_index(drop=True)
display_df.rename(columns={"model_pred": "Model Prediction", "mape": "Error (%)"}, inplace=True)
num_vars = display_df.select_dtypes(include=['int64','float64']).columns.tolist()[:-1]
num_vars = ["weight", "acceleration", "Y", "Model Prediction"]
# num_vars.remove("displacement")

st.markdown("""---""")
st.header("Model Risk Heatmap")
st.write("")
st.subheader("Select Model Features")


x_var = st.selectbox(label="X axis", 
             options=num_vars,
             index=3)

y_var = st.selectbox(label="Y axis", 
             options=num_vars,
             index=2)

# heatmap_num(display_df, x_var, y_var)

try:
    heatmap_num(display_df, x_var, y_var)
    
except ValueError:
    st.write("Please select two distinct variables")




