import streamlit as st
from st_pages import add_page_title
from functionalities.risks import SHAPEvaluator
from functionalities.load_data import load_data
from functionalities.plot_data import *

st.set_page_config(layout="wide", 
                   page_title="Explainability", 
                   page_icon=":mag:")
add_page_title()

# load data
df, X_train, X_test, y_train, y_test, model1 = load_data()

# preprocess df
shapcols = X_train.columns.values.tolist()
shap_df = df[shapcols]
X_enc = model1['Prep'].transform(shap_df)

explainer = SHAPEvaluator(X_enc=X_enc, model=model1[-1])

####### EXPLAINABILITY ############
display_df = df.drop(columns=["res", "abs_res"], axis=1).reset_index(drop=True)
display_df.rename(columns={"model_pred": "Model Prediction", "mape": "Error (%)"}, inplace=True)
num_vars = display_df.select_dtypes(include=['int64','float64']).columns.tolist()

st.header("Variable comparison")
st.subheader("Select Two Features")

x_var = st.selectbox(label="X variable", 
             options=num_vars,
             index=4)

y_var = st.selectbox(label="Y variable", 
             options=num_vars,
             index=5)

px_scatter(display_df, x_var, y_var)

rownum = st.number_input(label="Please insert observation number to inspect",
                         min_value=0,
                         max_value=display_df.shape[0],
                         value=0)

explainer.waterfall(n_sample=rownum)
