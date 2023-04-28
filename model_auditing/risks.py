import math
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, RocCurveDisplay
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.vq import vq

# 1. High level risk statement

def calculate_residuals(df, y_true, y_pred, task='R'):
    """
    Simple residual calculator

    INPUTS:
    -df: pd.DataFrame:
        DataFrame in which residual information should be
        included.

    -y_true: array-like, shape=(n_samples, )
        Contains true values for the target variable
    
    -y_pred: array-like, shape=(n_samples, )
        Contains predicted values for the target variable.
        In the case of regression, the array may correspond
        to the point predictions made by the estimator. In
        the case of classification, it corresponds with the 
        posterior probabilities output by the model.

    -task: {'R', 'C'}
        Name of the task for which to calculate the residuals.
        "R" refers to "regression" whilst "C" refers to "classification".
        Default value is "R"

    RETURNS:
    -res: array-like, shape=(n_samples,)
        residual value for each sample, calculated as the 
        difference between Y_actual and Y_Predicted

    """
    if task == "R":
        # calculate residuals
        def res(y_true, y_pred):
            return y_true - y_pred

        # calculate absolute values of residual
        def abs_res(y_true, y_pred):
            return np.abs(res(y_true, y_pred))
        
        # calculate MAPE
        def mape_res(y_true, y_pred):
            return np.abs(res(y_true, y_pred)/y_true)
        
        df["res"] = list(map(res, y_true, y_pred))
        df["abs_res"] = list(map(abs_res, y_true, y_pred))
        df["mape"] = list(map(mape_res, y_true, y_pred))


    elif task == "C":
        # calculate brier score
        def brier(y_true, y_pred):
            y_true = np.array(y_true == 1, int)
            return y_true - y_pred     

        # calculate absolute value of brier score
        def abs_brier(y_true, y_pred):
            return np.abs(brier(y_true, y_pred))

        df["brier"] = list(map(brier, y_true, y_pred))
        df["abs_brier"] = list(map(abs_brier, y_true, y_pred))
    
    else:
        raise ValueError("Unsupported task")

def plot_residuals(df, *columns):
    """
    Plots residuals for specified datasets

    INPUTS:
    
    -df: pd.DataFrame
        DataFrame containing the residuals to be plotted
    
    -*columns:
        Names of the residual columns to be plotted

    """
    sns.set()
    # Set axes
    fig, axs = plt.subplots(nrows=1, ncols=len(columns),
                            figsize=(15,7), sharex=False, 
                            sharey=False)
    plt.tight_layout()
    # Set different colors for different plots
    colors = sns.color_palette("pastel", len(columns))
    edgecolors = sns.color_palette("bright", len(columns))
    i = 0
    for col, ax in zip(columns, axs.ravel()):
        # plot residuals distribution
        plot = sns.histplot(data=df, x=col, ax=ax, 
                           bins=20, color=colors[i], 
                           edgecolor=edgecolors[i], 
                           kde=True, line_kws={'lw': 2})
        plot.lines[0].set_color('darkred')
        plot.set_title(col+' plot', fontsize=20)
        plot.set_xlabel(col.title(), fontsize=15)
        plot.set_ylabel('')
        i+=1
            
    plt.show()

def qq_plots(df, *names):
    """
    Function used for plottig QQplots for residual columns in dataframe

    INPUTS:
    -df: Pandas DataFrame
        DataFrame from which residuals are to be plotted
    -names: 
        Names of residuals to be plotted
    
    RETURNS:
    -plot:
        qqplots for each residual distribution (res, abs_res, mape)
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,7), sharex=False, sharey=False)
    plt.tight_layout()
        
    for name, ax in zip(names, axs.ravel()):
        sm.qqplot(df[name], line ='45', ax=ax)
    plt.show()

# 2. KDE

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
        from sklearn.neighbors import KernelDensity
        import gc

        sns.set_style('whitegrid')
        
        plt.figure(figsize = (15, 7))

        x = np.linspace(self.res.min(), self.res.max(), 1000)[:, np.newaxis]

        # Plot the data using a normalized histogram
        plt.hist(self.res, bins=nbins, density=True, label=plotname, color='orange', alpha=0.2)

        # Do kernel density estimation
        kd = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.res.values[:, np.newaxis])
        self.kd = kd

        # Plot the estimated densty
        kd_probs = np.exp(kd.score_samples(x))

        plt.plot(x, kd_probs, color='orange')

        plt.axvline(x=self.cutoff_value,color='red',linestyle='dashed')
        plt.axvline(x=self.x_end,color='red',linestyle='dashed')

        # Show the plots
        plt.xlabel(plotname, fontsize=15)
        plt.ylabel('Probability Density', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        gc.collect()

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

# 3. Global model metrics

def create_CM(y_true, y_pred, labels=[1,0]):
    """
    Function used for creating confusion matrix for classification

    -y_true: array-like, shape(n_samples, 1)
        Contains true values for the target variable
    
    -y_pred: array-like, shape=(n_samples, 1)
        Contains predicted values for the target variable.
        The array may correspond to the point predictions 
        made by the estimator.
    
    -labels: list
        List of labels to include in the plot. The first label
        corresponds to the positive class, while the second 
        label corresponds to the negative class. Default=[1,0]
    """
    # create confusion matrix
    CM = confusion_matrix(y_true, y_pred, labels=[1,0])
    cm_df = pd.DataFrame(CM, columns=labels)
    cm_df = pd.DataFrame(labels, columns=['Reference']).join(cm_df)
    TP = CM[1,1]
    TN = CM[0,0]
    FP = CM[0,1]
    FN = CM[1,0]

    metrics = {}

    # calculate relevant metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred, normalize=True)
    metrics["Balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Sensitivity"] = round(TP / (TP + FN), ndigits=2)
    metrics["Specificity"] = round(TN / (TN + FP), ndigits=2)
    metrics["Precision"] = precision_score(y_true, y_pred)

    for metric, value in metrics.items():
        print("{}: {:.2f}".format(metric, value))
    
    return metrics

def display_ROC(y_true, y_pred):
    """
    Function used for displaying ROC for classification model.

    -y_true: str, default="Y"
        Contains true values for the target variable
    
    -y_pred: array-like, shape=(n_samples, )
        Contains predicted values for the target variable.
        The array may correspond to the point predictions 
        made by the estimator.
    """
    # calculate fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # get AUC curve
    roc_auc = auc(fpr, tpr)
    # display AUC curve
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.show()

def radial_plot(metrics):
    """
    Radial plot for classification metrics
    
    metrics: dict
        Dictionary containing relevant metrics for plotting. 
    """
    df = pd.DataFrame(dict(r=metrics.values(),
                           theta=metrics.keys()))

    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()

def global_metrics(y_true, y_pred, task):
    """
    Calculate global performance metrics for model at hand

    INPUTS:
    -df: pd.DataFrame:
        DataFrame in which residual information should be
        included.

    -y_true: str, default="Y"
        Contains true values for the target variable
    
    -y_pred: array-like, shape=(n_samples, )
        Contains predicted values for the target variable.
        In the case of regression, the array may correspond
        to the point predictions made by the estimator. In
        the case of classification, it corresponds with the 
        posterior probabilities output by the model.

    -task: {'R', 'C'}
        Name of the task for which to calculate the residuals.
        "R" refers to "regression" whilst "C" refers to "classification".
        Default value is "R"

    RETURNS:
    -res: array-like, shape=(n_samples,)
        Residual value for each sample, calculated as the 
        difference between Y_actual and Y_Predicted

    """

    if task == "R":
        # MAE - Mean Absolute error
        print('MAE:',mean_absolute_error(y_true, y_pred))
        # MSE - Mean squared error
        print('Global MSE:', mean_squared_error(y_true, y_pred))
        # RMSE - Root Mean Square Error
        print('RMSE:',math.sqrt(mean_squared_error(y_true, y_pred)))
        # R^2 
        print('R2:',r2_score(y_true, y_pred))
        # MAPE - Mean Absolute Percentage Error
        print('MAPE:',"{0:.4%}".format(mean_absolute_percentage_error(y_true, y_pred)))

    elif task=="C":
        # Confusion matrix
        cm_metrics = create_CM(y_true, y_pred)
        # ROC curve
        display_ROC(y_true, y_pred)
        # Radial plot
        radial_plot(cm_metrics)

    else:
        raise ValueError("Unsupported task")

# 4. Analysis of residuals

def residual_analysis(df, res, figsize=(15,15), smooth_order=5):
    """
    Function used for plotting residuals for specified DataFrame.
    
    Original code belongs to Jaime Pizarroso 

    INPUTS:
    -df: pd.DataFrame
        DataFrame containing model features and residual information
    
    -res: str
        Name of the column containing residual information
    
    -figsize: tuple
        Desired size of the plot. Default=(15, 15)

    -smooth_order: int
        Polynomial order for regression plot. Defaults to 5.

    
    """
    colnames = df.columns.values.tolist()
    out_num = np.where(df.columns.values == res)[0]
    nplots = df.shape[1]-1
    # Create subplots
    fig, axs = plt.subplots(math.floor(math.sqrt(nplots))+1, math.ceil(math.sqrt(nplots)), figsize=figsize)
    fig.tight_layout(pad=5.0)
    input_num = 0

    for ax in axs.ravel():
        if input_num < nplots:
            # Create plots
            if input_num != out_num:
                if df.iloc[:,input_num].dtype.name == 'category':
                    sns.boxplot(x=colnames[input_num], y=res, data=df, ax=ax)
                    ax.set_title(colnames[input_num] + ' vs residuals')
                else:
                    sns.regplot(x=colnames[input_num], y=res, data=df, ax=ax,\
                                order=smooth_order, ci=None, scatter_kws={"color": "red"},\
                                line_kws={'color':'navy'})
                    ax.set_title(colnames[input_num] + ' vs residuals')

            input_num += 1
        else:
            ax.axis('off')

# 5. Variance and bias
def _draw_bootstrap_sample(rng, X, y):
    """
    Function used for drawing a bootstrap sample from the specified variables

    INPUTS:

    -rng: random seed generator

    -X: array-like, shape=(n_samples, n_features)
        contains input variables
    
    -y: array-like, shape=(n_samples,)
        contains true values for the target variable
    
    RETURNS:
    -X[bootstrap_indices]: array-like, shape=(n_samples, n_features)
        contains input variables for randomly selected bootstrap samples 
    
    -y[bootstrap_indices]: array-like, shape=(n_samples,)
        contains true values for the target variable for randomly selected bootstrap sampless
    
    """
    sample_indices = range(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices, size=X.shape[0], replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]
    
def bias_variance_decomp(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    loss='mse',
    num_rounds=200,
    random_seed=None,
):

    """
    INPUTS:

    estimator : object
        A classifier or regressor object or class implementing both a
        `fit` and `predict` method similar to the scikit-learn API.

    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.

    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.

    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.

    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.

    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.

    num_rounds : int (default=200)
        Number of bootstrap rounds (sampling from the training set)
        for performing the bias-variance decomposition. Each bootstrap
        sample has the same size as the original training set.
        
    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.
    
    RETURNS:
   
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        average bias, and average bias (all floats), where the average
        is computed over the data points in the test set.

    EXAMPLES
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/
    """

    rng = np.random.RandomState(random_seed)

    if loss == "0-1_loss":
        dtype = np.int64
    elif loss == "mse":
        dtype = np.float64

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=dtype)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)
        pred = model.fit(X_boot, y_boot).predict(X_test)
        all_pred[i] = pred
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if loss == "0-1_loss":
        # take predictions for each observation and then
        # take the value that is most common (either 1 or 0)
        main_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred
        )

        # for each prediction round, how many times on average have
        # I been mistaken on the prediction?
        # Then, calculate the global average of that value
        avg_expected_loss = np.apply_along_axis(
            lambda x: (x != y_test).mean(), axis=1, arr=all_pred
        ).mean()

        # Difference between the model prediction (on average) and the
        # ground truth
        avg_bias = (np.sum((main_predictions != y_test)) / y_test.size)**2

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int)
        var /= num_rounds

        avg_var = (var.sum() / y_test.shape[0])

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred
        ).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test)** 2)  / y_test.size
        avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var

# 6,c Feature importance --> use model_auditor.py

# 7. Clustering

class Clusterer:
    """
    Class used for clustering observations with high error
    """
    def __init__(self, df_cluster):
        self.df_cluster = df_cluster
        self.scaler = StandardScaler()
        self.X_transformed = None
        self.n_clusters = None
        self.kmeans = None
        self.labels = None

    def plot_knee(self):
        """
        Plot silhouette and knee plots for determining optimal number
        of clusters to be used
        """
        sns.set()
        # instantiate range of clusters
        range_clusters = np.arange(2, 8)
        # scale X
        X_transformed = self.scaler.fit_transform(X=self.df_cluster.values) 
        self.X_transformed = X_transformed
        SSQ = []
        sil_avg = []
        for n_clusters in range_clusters:

            # Initialize the clusterer with n_clusters value 
            kmeans = KMeans(n_clusters=n_clusters, random_state=2022)
            cluster_labels = kmeans.fit_predict(X_transformed)

            #Obtain Reconstruction error
            _ , err = vq(X_transformed, kmeans.cluster_centers_)
            SSQ.append(np.sum(err**2))

            #Obtain silhouette
            sil_avg.append(silhouette_score(X_transformed, cluster_labels))
        
        # Plot graphs
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(range_clusters, SSQ, marker='o')
        ax1.set_title("Sum of Squares") # es el grafico del codo o knee bend
        ax1.set_xlabel("Number of clusters")
        ax2.plot(range_clusters, sil_avg, marker='o')
        ax2.set_title("Silhouette values")
        ax2.set_xlabel("Number of clusters")
        plt.show()

    def fit_knn(self, n_clusters):
        """
        Plot clusters in PCA space
        """
        # Fit final model and validate
        kmeans = KMeans(n_clusters, n_init="auto", random_state=2022)
        self.n_clusters = n_clusters
        self.kmeans = kmeans
        #Predict on training dataset
        cluster_knn = kmeans.fit_predict(self.X_transformed)
        self.df_cluster['cluster'] = cluster_knn
        self.df_cluster['cluster'] = self.df_cluster['cluster'].astype('category')
        self.labels = np.array(self.df_cluster['cluster'])
        # Cut dendrogram tree by number of clusters
        # print('Clusters specifying number of clusters:')
        # unique, counts = np.unique(cluster_knn, return_counts=True)
        # print(pd.DataFrame(np.asarray(counts), index=unique, columns=['# Samples']))

        # Silhouette
        # silhouette_avg = silhouette_score(self.X_transformed, cluster_knn)
        # print("The average silhouette_score is :", silhouette_avg)

    def plot_clusters(self, show_curves=False, alpha_curves=0.5, scale=False):
        """Plot silhouette and cluster of samples 
        
        Create two plots, one with silhouette score and other with a 2D view of the cluster of self.df_cluster.
        If there are more than 2 features in self.df_cluster, 2D view is made on the first 2 PCAs space.
        labels must have the same samples as self.df_cluster. If there are more than 2 features in self.df_cluster, show_curves 
        argument make a plot assuming you are trying to cluster curve samples, indexed by rows. 
        Args:
            feature1 (str, optional): Name of the x-axis variable to create plot. Defaults to None.
            feature2 (str, optional): Name of the y-axis variable to create plot. Defaults to None.
            centers (np.ndarray, optional): cluster centers. Defaults to None.
            show_curves (bool, optional): show curve plot. Defaults to False.
            figsize (tuple, optional): Size of created figure. Defaults to (12, 9).
            alpha_curves (float, optional): alpha property of curve plot. Defaults to 0.5.
            scale (bool, optional): Scale axis in scatter plot in range [0, 1]. Defaults to False.

        """
        sns.set()
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        plt.tight_layout()
        
        if len(self.labels.shape) > 1:
            self.labels = np.squeeze(self.labels)
        
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters
        ax1.set_ylim([0, self.df_cluster.shape[0] + (self.n_clusters + 1) * 10])
        y_lower = 10
        
        # Compute the silhouette scores for each sample
        pca = PCA(n_components=2,)
        X_pca = pd.DataFrame(pca.fit_transform(self.X_transformed), columns=['PC1','PC2'])
        sample_silhouette_values = silhouette_samples(self.X_transformed, self.labels)
        if scale:
            scalex = 1.0/(X_pca.iloc[:,0].max() - X_pca.iloc[:,0].min())
            scaley = 1.0/(X_pca.iloc[:,1].max() - X_pca.iloc[:,1].min())
            X_pca.iloc[:,0] = X_pca.iloc[:,0] * scalex
            X_pca.iloc[:,1] = X_pca.iloc[:,1] * scaley
            
            
        for i in range(self.n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            samples_cluster = self.labels == i
            ith_cluster_silhouette_values = sample_silhouette_values[samples_cluster]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.nipy_spectral()
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            df_plot = X_pca.iloc[samples_cluster,:]
            sns.scatterplot(x='PC1', y='PC2', data=df_plot, color=color, edgecolor='black', legend='full', ax=ax2)

            ax2.set_title("Visualization of the clustered data.")
            ax2.set_xlabel(f"Feature space for PC1")
            ax2.set_ylabel(f"Feature space for PC2")
            
            if centers is not None:
                center_pca = pca.transform(self.scaler.transform(centers))
                if scale:
                    center_pca[:,0] = center_pca[:,0] * scalex
                    center_pca[:,1] = center_pca[:,1] * scaley
                    
                # Draw white circles at cluster centers
                ax2.scatter(center_pca[:, 0], center_pca[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(center_pca):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

        ax1.set_title(f"Silhouette plot. Mean silhouette: {round(sample_silhouette_values.mean(),3)}")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=sample_silhouette_values.mean(), color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        if self.df_cluster.shape[1] > 2 and show_curves:
            ## Multivariate plots
            for i in range(self.n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                samples_cluster = self.labels == i
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[samples_cluster]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # self.df_cluster.iloc[samples_cluster,:].T.plot(legend=None, color=color, ax=ax3, alpha=alpha_curves)
            
        plt.suptitle(("Silhouette analysis for hierarchical clustering on sample data "
                        "with n_clusters = %d" % self.n_clusters),
                        fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Plot clustering classification
        # plot_clusters(self.df_cluster, labels=cluster_knn)
        
    def calculate_mean_res_cluster(self):
        """
        Calculate mean value of residuals for each cluster
        """
        mean_res_cluster = self.df_cluster.groupby(['cluster'])['abs_res'].mean()

        return mean_res_cluster, self.df_cluster
    
# 7. Visualization

def catplots(df, resname):
    """
    Plot categorical or numerical binned data using boxplot

    INPUTS:

    -df: pd.DataFrame
        DataFrame containing categorical features to be plotted
    -resname: string
        Name of the residuals to be plotted
    """
    for column in df.columns:
        if 'binned' in df[column].name or df[column].dtype == 'category':
            sns.catplot(x=column, y=resname, kind="box", data=df,  height=7, aspect=2)
            plt.show()
        else:
            continue

def scatterplot(df, x, y, resname):
    """
    Plot two variables against each other and color them by value
    of residuals

    INPUTS:
    -df: pd.DataFrame
        DataFrame containing categorical features to be plotted
    
    -x: string
        Name of the variable to be plotted in the X-axis. Must be
        present in df
    -y: string
        Name of the variable to be plotted in the Y-axis. Must be
        present in df
    -resname: string
        Name of the residuals to be plotted
    """
    fig = px.scatter(data=df, x=x, y=y, color=resname,
                    title="{x} vs {y} colored by residual value".format(x, y),
                    width=700, height=700)

    fig.show()