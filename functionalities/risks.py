import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from sklearn.neighbors import KernelDensity
import gc
import shap
import streamlit as st

class KDE:
    """
    Class used for applying KDE analysis

    INPUTS:
    -res: Pandas DataFrame, shape=(n_samples, 1)
            residual data to be plotted.

    -cutoff_value: int or float
            initial value for defining probability area of interest within the plot

    -x_end: int or float
            final value for defining probability area of interest within the plot
    """
    def __init__(self, res, cutoff_value, x_end):
        self.res = res
        self.cutoff_value = cutoff_value
        self.x_end = x_end
        self.kd = None
        self.probability = None
        

    def plot_prob_density(self, plotname, bandwidth=0.01, nbins=10):
        """
        Function used to calculate and plot PDF distribution using KDE

        Adapted from: https://towardsdatascience.com/how-to-find-probability-from-probability-density-plots-7c392b218bab

        INPUTS:
        
        -plot_name: string
            col of the type of residual to be plotted. Used for labelling plot
        
        -bandwidth: int
            bandwidth used for KDE. Default value is 0.01

        -nbins: int
            number of bins for the histogram. Default value is 10

        """
        plt.rcParams['axes.facecolor'] = 'white'
        fig, ax = plt.subplots(figsize = (15, 7))
        ax.grid(visible=False)

        x = np.linspace(self.res.min(), self.res.max(), 1000)[:, np.newaxis]

        # Plot the data using a normalized histogram
        plt.hist(self.res, bins=nbins, density=True, label=plotname, color='#92c5de', alpha=1)

        # Perform kernel density estimation
        kd = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.res.values[:, np.newaxis])
        self.kd = kd

        # Plot the estimated densty
        kd_probs = np.exp(kd.score_samples(x))

        plt.plot(x, kd_probs, color='black')

        plt.axvline(x=self.cutoff_value,color='black',linestyle='dashed')
        plt.axvline(x=self.x_end,color='black',linestyle='dashed')
        plt.gca().spines['bottom'].set_color('black') 
        plt.gca().spines['left'].set_color('black') 
        plt.gca().spines['left'].set_linewidth(1.8) 
        plt.gca().spines['bottom'].set_linewidth(1.8) 

        # Show the plots
        plt.xlabel("Expected Error (%)", fontsize=15)
        plt.ylabel('Probability Distribution', fontsize=15)
        plt.legend(loc='upper right', fontsize=12)
        fmt_percent = PercentFormatter(xmax=1.0, decimals=0)
        ax.xaxis.set_major_formatter(fmt_percent)
        return fig
        # plt.show()
        # gc.collect()

    def get_probability(self):
        """
        Function used to calculate probability in a defined region
        from a fitted KDE distribution using integrals.

        Adapted from: https://towardsdatascience.com/how-to-find-probability-from-probability-density-plots-7c392b218bab

        RETURNS:
        -probability: float
            probability of a value being between cutoff_value and x_end
        
        """
        
        # Number of evaluation points 
        N = 100  
        # Step size                                    
        step = (self.x_end - self.cutoff_value) / (N - 1)

        # Generate values in the range
        x = np.linspace(self.cutoff_value, self.x_end, N)[:, np.newaxis]  
        # Get PDF values for each x
        kd_vals = np.exp(self.kd.score_samples(x)) 
        # Approximate the integral of the PDF 
        probability = np.sum(kd_vals * step) 
        self.probability = probability

        return probability
    
class SHAPEvaluator:
    """
    Class for using SHAP on model
    """
    def __init__(self, X_enc, model):
        self.X_enc = X_enc
        self.model = model
        self.shap_values = None
        self.feature_importance_table = None
        self.apply_shap()

    def apply_shap(self):
        """
        Fit SHAP evaluator to model
        """
        # instantiate explainer
        try:
            explainer = shap.TreeExplainer(self.model)
        except:
            explainer = shap.LinearExplainer(model=self.model, masker=self.X_enc)
        # calculate shap_values
        shap_values = explainer(self.X_enc)
        self.shap_values = shap_values

    def summary_plot(self):
        # summary plot         
        fig, ax = plt.subplots() 
        ax.grid(visible=False)
        ax.set_facecolor("white")

        # custom colors
        color1 = '#444444'
        color2 = '#2F94B3'
        
        # Create a custom colormap that is a gradient between the two colors
        cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', [(0, color1), (1, color2)])

        shap.summary_plot(self.shap_values, self.X_enc, cmap = cmap, show = False, max_display=10) 
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['left'].set_linewidth(2)
        plt.xlabel("Impact on Model Error", fontsize = 15)
        st.pyplot(fig)
        
    
    def feature_importance(self):
        """
        Get variables order by feature importance
        """
        # create DataFrame with shap values
        df_shap_values = pd.DataFrame(data=self.shap_values.values,
                                      columns=self.X_enc.columns)
        df_feature_importance = pd.DataFrame(columns=['feature','importance'])

        # calculate absolute value of feature importance
        for col in df_shap_values.columns:
            importance = df_shap_values[col].abs().mean()
            df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
        
        # sort data by importance of features
        df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)
        self.feature_importance_table = df_feature_importance

        return df_feature_importance
    
    def waterfall_bar(self, n_sample):
        """Waterfall plot (using bars)"""

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.grid(visible=False)
        ax.set_facecolor("white")
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['left'].set_linewidth(2)

        # Default SHAP colors
        default_pos_color = "#ff0051"
        default_neg_color = "#008bfb"
        # Custom colors
        positive_color = "#2F94B3" 
        negative_color = "#444444"

        shap.plots.bar(self.shap_values[n_sample], show = False)
        # Change the colormap of the artists
        for fc in plt.gcf().get_children():
            # Ignore last Rectangle
            for fcc in fc.get_children()[:-1]:
                if (isinstance(fcc, matplotlib.patches.Rectangle)):
                    if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                        fcc.set_facecolor(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                        fcc.set_color(negative_color)
                elif (isinstance(fcc, plt.Text)):
                    if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                        fcc.set_color(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                        fcc.set_color(negative_color)
        st.pyplot(fig)

    def waterfall(self, n_sample):
        """Waterfall plot"""

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.grid(visible=False)
        ax.set_facecolor("white")
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['left'].set_linewidth(2)

        # Default SHAP colors
        default_pos_color = "#ff0051"
        default_neg_color = "#008bfb"
        # Custom colors
        positive_color = "#2F94B3" 
        negative_color = "#444444"

        f = shap.plots.waterfall(self.shap_values[n_sample], show = False)

        # Change the colormap of the artists
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                    if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                        fcc.set_facecolor(positive_color)
                        fcc.set_edgecolor(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                        fcc.set_color(negative_color)
                        fcc.set_edgecolor(negative_color)
                elif (isinstance(fcc, plt.Text)):
                    if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                        fcc.set_color(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                        fcc.set_color(negative_color)
        st.pyplot(fig)
    
    def plot_feature_importance(self, y):
        """Plot feature importance"""

        # get top features ordered by importance
        default_pos_color = "#ff0051" 
        default_neg_color = "#008bfb" 
        # Custom colors 
        positive_color = "#2F94B3" 
        negative_color = "#444444" 

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.grid(visible=False)
        ax.set_facecolor("white")
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['left'].set_linewidth(2)

        f = shap.plots.bar(self.shap_values, max_display = 10, show = False) 

        # Change the colormap of the artists 
        for fc in plt.gcf().get_children(): 
            for fcc in fc.get_children()[:-1]: 
                if (isinstance(fcc, matplotlib.patches.Rectangle)): 
                    if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color): 
                        fcc.set_facecolor(positive_color)
                        
                    elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color): 
                        fcc.set_color(negative_color) 
                    
                elif (isinstance(fcc, plt.Text)): 
                    if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color): 
                        fcc.set_color(positive_color)

                    elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color): 
                        fcc.set_color(negative_color) 

        plt.xlabel("Mean Impact on Error", fontsize = 15)
        st.pyplot(fig)
