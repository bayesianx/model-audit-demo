import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from sklearn.impute import KNNImputer
import plotly.express as px

def to_discrete(df, vars, n_bins='auto'):
    """
    Function used to discretiza a given variable from a DataFrame
    
    INPUTS:
    
    - df: Pandas DataFrme
        the desired DataFrame containing the variable we wish to discretize

    - var: string
        name of the variable to discretize
    
    RETURNS:

    - df: Pandas DataFrame
        the modified DataFrame containing the binned variable
    """
    df_discrete = df.copy()
    for var in vars:
        varname = f'{var}_binned'

        # drop column if already exists
        if varname in df_discrete.columns:
            df_discrete.drop(varname, axis=1, inplace=True)

        if n_bins == 'auto':
            # calculate number of bins
            lower_edge, upper_edge = df_discrete[var].min(), df_discrete[var].max()
            n = len(df_discrete[var])
            size = math.ceil(2 * n**(1/3))
            n_bins = int(math.ceil(upper_edge - lower_edge) / size)
        else:
            pass
        
        # create binned variable
        var_binned = pd.cut(df_discrete[var], bins=n_bins, include_lowest=True, ordered=True, precision=2) # cut variable into intervals
        # get position of the original variable
        binned_col_position = df_discrete.columns.get_loc(var)+1
        
        # insert binned variable in the DataFrame
        df_discrete.insert(binned_col_position, varname, var_binned, 2)
    
    return df_discrete

def heatmap_num(df, x_var, y_var):
    """Function used to plot a heatmap of the model error by two variables
    
    INPUTS:
        
        - df: Pandas DataFrame
            the DataFrame containing the variables to plot
        - x_var: string
            the name of the variable to plot on the x axis
        - y_var: string
            the name of the variable to plot on the y axis 
            
    """
    # discretize variables
    df_discrete = to_discrete(df, vars=[x_var, y_var], n_bins=15)

    # create pivot table for plotting
    heatmap_df = df_discrete.groupby([f"{x_var}_binned", f"{y_var}_binned"])["Error (%)"].mean().reset_index()
    df_pivot = heatmap_df.pivot(f"{x_var}_binned", f"{y_var}_binned", "Error (%)")

    # use KNN imputer to fill missing values
    knn_imputer = KNNImputer(n_neighbors=4, weights='uniform', metric='nan_euclidean')
    knn_imputer.fit(df_pivot)
    df_pivot_knn = knn_imputer.transform(df_pivot)

    # plot heatmap
    fig, ax = plt.subplots()
    # create a custom colormap that is a gradient between the two colors
    cmap = sns.color_palette("ch:start=.1,rot=-.25", as_cmap=True)
    ax = sns.heatmap(df_pivot_knn, cmap=cmap, cbar=True, cbar_kws={"label" : "Expected Error"})
    x_labels = [int(id.left) for id in df_pivot.index.tolist()]
    y_labels = [int(id.left) for id in df_pivot.columns.tolist()]
    ax.set_yticklabels(x_labels, rotation=0)
    ax.set_xticklabels(y_labels, rotation=90)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel(x_var.title())
    ax.set_ylabel(y_var.title())
    st.pyplot(fig)

def px_scatter(df, x_var, y_var):
    """Function used to plot a scatter plot of the model error by two variables"""
    # Create a custom colormap that is a gradient between the two colors
    cmap = ["#82bed1", "#6db4c9", "#58a9c2" ,"#2f94b3", "#2a85a1","#1c586b", "#444444"]
    
    # Plot the scatter plot
    fig = px.scatter(df, x=x_var, y=y_var,
                     hover_name=df.index,
                     hover_data=[x_var, y_var],
                     color="Error (%)",
                     color_continuous_scale=cmap,
                     width=700, height=700)
    
    fig.update_xaxes(showgrid=False, color="black", linewidth=1.5)
    fig.update_yaxes(showgrid=False, color="black", linewidth=1.5)
    
    st.plotly_chart(fig, use_container_width=True)