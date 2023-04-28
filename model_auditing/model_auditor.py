# Code for auditing model risks
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, SMOTENC
from .preprocessing_tools import column_encoder
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, brier_score_loss
from lazypredict.Supervised import LazyRegressor, LazyClassifier

random_state = 2022
rng = np.random.default_rng(random_state)

# 6. Feature importance for error distribution

def split_data(df, features, output, test_size=None):
    """
    Split data between training and testing
    """
    # define X and y vars
    X = df[features]
    y = df[output]
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=123)
        
    return X_train, X_test, y_train, y_test

def fix_class_imbalance(X_train, y_train):
    """
    Fix imbalanced classes prior to training
    """
    
    # Fix class imbalance using SMOTE or SMOTENC
    cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
    cat_ids = [X_train.columns.get_loc(col) for col in cat_cols]
    # Use SMOTENC if categorical columns are present
    if len(cat_cols) >= 1:
        imbalance_fixer = SMOTENC(random_state=2022, categorical_features=cat_ids)
        X_balanced, y_balanced = imbalance_fixer.fit_resample(X_train.values, y_train.values)
    
    # if no cat variables are present, use regular SMOTE
    else:
        imbalance_fixer = SMOTE(random_state=2022)
        X_balanced, y_balanced = imbalance_fixer.fit_resample(X_train.values, y_train.values)
    
    # Fix format
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    column_encoder(X_balanced, X_train)
    return X_balanced, y_balanced

def fit_encoder(X_train):
    """
    Fit encoder used for encoding categorical and scaling numerical variables
    """
    ## Inputs of the model. Change accordingly to perform variable selection
    inputs_num = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    inputs_cat = X_train.select_dtypes(include=['category']).columns.tolist()
    inputs = inputs_num + inputs_cat
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False,
                                            drop='if_binary',
                                            handle_unknown='ignore') 
    preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, inputs_num),
            ('cat', categorical_transformer, inputs_cat)],
            verbose_feature_names_out=False
            ).set_output(transform="pandas")

    preprocessor.fit(X_train)

    return preprocessor

class ModelOptimizer:
    """
    Class used for hyperparameter tuning based on Bayesian Optimization

    Functions based on code from: https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec
    """
    def __init__(self, scoring):
        self.optimizer = None
        self.best_optimizer = None
        self.scoring = scoring

    def black_box_function(self, X_train_scale, y_train, model, **params):
        """
        Black box function for optimization algorithm
        """
        model = model.set_params(**params)
        f = cross_val_score(model, X_train_scale, y_train,
                            scoring=self.scoring, cv=10)
        
        return f.mean()

    def optimize_model(self, pbounds, X_train_scale, y_train, model, int_params, n_iter=25):
        """
        Optimize 

        """
        def opt_function(**params):
            """
            Function wrapper
            """
            return self.black_box_function(X_train_scale,
                                           y_train,
                                           model,
                                           **params)
        # create optimizer
        optimizer = BayesianOptimization(f = None,
                                         pbounds=pbounds,
                                         verbose=2,
                                         random_state=2022)

        # declare acquisition function used for getting new values of the
        # hyperparams
        utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)

        # Optimization for loop.
        for i in range(n_iter):
            # Get optimizer to suggest new parameter values to try using the
            # specified acquisition function.
            next_point = optimizer.suggest(utility)
            # Force degree from float to int.
            for int_param in int_params:
                next_point[int_param] = int(next_point[int_param])
            # Evaluate the output of the black_box_function using 
            # the new parameter values.
            target = opt_function(**next_point)
            try:
                # Update the optimizer with the evaluation results. 
                # This should be in try-except to catch any errors!
                optimizer.register(params = next_point, target = target)
            except:
                pass
                   
        print("Best result: {}.".format(optimizer.max["params"]))

        self.optimizer = optimizer
        self.best_optimizer = optimizer.max

        return optimizer.max["params"]

class SHAPEvaluator:
    """
    Class for using SHAP on model
    """
    def __init__(self, X_enc, model):
        self.X_enc = X_enc
        self.model = model
        self.shap_values = None
        self.apply_shap()

    def apply_shap(self):
        """
        Fit SHAP evaluator to model
        """
        # instantiate explainer
        explainer = shap.TreeExplainer(self.model)
        # calculate shap_values
        shap_values = explainer(self.X_enc)
        self.shap_values = shap_values

    def summary_plot(self):
        # summary plot 
        shap.summary_plot(self.shap_values, self.X_enc) 
    
    def scatter_plots(self):
        # scatter plots
        shap.plots.scatter(self.shap_values[:, self.shap_values.abs.mean(0).argsort[:-4:-1]])

    def waterfall(self, n_sample):
        #waterfall plot
        shap.plots.waterfall(self.shap_values[n_sample])

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

        return df_feature_importance

class LazyModels:
    """
    Class used for creating lazy models using LazyPredict
    """
    def __init__(self):
        self.lazy = None
        self.models = None
        self.predictions = None
    
    def get_models(self, X_train, X_test, y_train, y_test):
        """
        Fit models using LazyPredict on training data and get
        predictions for testing data
        """
        self.models, self.predictions = self.lazy.fit(X_train, X_test, y_train, y_test)
        return self.models, self.predictions

    def top_models(self, n):
        """
        Get top n models from LazyPredict, as well as their 
        predictions
        """
        top_models = self.models.iloc[0:n]
        top_models_preds = self.predictions[top_models.index.values]
        return top_models, top_models_preds

class LazyModelsReg(LazyModels):
    """
    Implementation of LazyModels for regression
    """
    def __init__(self):
        super().__init__()
        self.lazy = LazyRegressor(verbose=0, 
                                  ignore_warnings=False, 
                                  custom_metric=mean_absolute_percentage_error,
                                  predictions=True)

class LazyModelsClass(LazyModels):
    """
    Implementation of LazyModels for regression
    """
    def __init__(self):
        super().__init__()
        self.lazy = LazyClassifier(verbose=0,
                                   ignore_warnings=False, 
                                   custom_metric=brier_score_loss,
                                   predictions=True)

def lazy_pred_class(df, features, output, test_size=None):
    from sklearn.metrics import brier_score_loss
    
    X_train, X_test, y_train, y_test = split_data(df, features,
                                                  output=output, test_size=test_size)
    classifiers = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=brier_score_loss,
                        predictions=True)
                        
    models, predictions = classifiers.fit(X_train, X_test, y_train, y_test)

    return models, predictions, y_test

def lazy_pred_reg(df, features, output):
    from sklearn.metrics import mean_absolute_percentage_error
    
    X_train, X_test, y_train, y_test = split_data(df, features, output, test_size=None)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=mean_absolute_percentage_error,
                        predictions=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    return models, predictions, y_test

def lazy_pred_clusters(df, features, output, task, test_size=None):
    cluster_models = {}
    cluster_preds = {}
    cluster_y = {}
    clusters = df['cluster'].value_counts().sort_index(ascending=True).index.tolist()
    for cluster in clusters:
        print("Fitting models for cluster %s" % cluster)
        cluster_data = df.loc[df['cluster'] == cluster, :]
        if task == 'C':
            models, predictions, y_test = lazy_pred_class(cluster_data, features, output, test_size)
        elif task == 'R':
            models, predictions, y_test = lazy_pred_reg(cluster_data, features, output)
        top_models = models.iloc[0:3]
        top_models_preds = predictions[top_models.index.values.tolist()]
        cluster_models[cluster] = top_models
        cluster_preds[cluster] = top_models_preds
        cluster_y[cluster] = y_test
    print("All cluster analyzed")

    return cluster_models, cluster_preds, cluster_y

# 4. Global risk reduction

def model_accuracy(model, X_test, y_test, metric_function):
    """
    Calculate accuracy metric for a given model

    INPUTS:
    -model: object
        Model used for calculating model accuracy

    -X_test: pd.DataFrame
        Data to be used for calculating metric

    -y_test: array-like, shape=(n_samples, )
        Ground truth values for output variable
    
    -metric_function: function
        Function used for calculating accuracy
    """
    # predict data using model
    y_pred = model.predict(X_test)
    # calculate accuracy metric selected
    metric = metric_function(y_test, y_pred)
    # return value of the metric
    return metric

def calculate_hyper_acc(hyper_models, X_test, y_test, metric_function):
    """
    Function used for calculating accuracy metric on 
    hyperparameter tuned models

    INPUTS:
    -hyper_models: list
        List of hyperparameter-tuned model objects
    
    -X_test: pd.DataFrame
        Data to be used for calculating metric

    -y_test: array-like, shape=(n_samples, )
        Ground truth values for output variable
    
    """
    hyper_accs = {}
    for model in hyper_models:
        # calculate accuracy metric
        acc = model_accuracy(model, X_test, y_test, metric_function=metric_function)
        # save it to response dict
        hyper_accs[type(model).__name__] = acc
    return hyper_accs

def calculate_cluster_acc(cluster_models, cluster_preds, cluster_y, metric):
    """
    Function used for calculating accuracy for cluster-specific models

    INPUTS:

    -cluster_models: array
        List of hyperparameter-tuned model objects

    -cluster_preds: array
        List containing predictions for each of the cluster models
    
    -cluster_y: array
        Ground truth values corresponding to the predicted ones for
        each cluster analyzed.
    
    -metric: function
        Metric function to be used when calculating cluster accuracies

    """
    cluster_accs = {}

    for cluster in cluster_models.keys():
        # select top model for predicting clusters
        top_model = cluster_models[cluster].index[0]
        # retrieve its corresponding predictions
        top_preds = cluster_preds[cluster][top_model]
        y_test = cluster_y[cluster]
        # calculate accuracy metric
        top_acc = metric(y_test, top_preds)
        # save to response dict
        cluster_accs[cluster] = top_acc
    
    return cluster_accs

def calculate_weighted_accuracy(df, df_cluster, X_test, y_test, 
                                hyper_models, cluster_models,
                                cluster_preds, cluster_y, 
                                metric_function):
    """
    Function used for computing weighted accuracies calculations

    INPUTS:
    -df: pd.DataFrame
        Original DataFrame containing all data (normal samples and clustered ones)

    -df_cluster: pd.DataFrame
        DataFrame containing data corresponding to clustered samples
    
    -X_test: pd.DataFrame
        Data to be used for calculating metric

    -y_test: array-like, shape=(n_samples, )
        Ground truth values for output variable
    
    -hyper_models: list
        List of hyperparameter-tuned model objects
    
    -cluster_models: array
        List of hyperparameter-tuned model objects
    
    """
    normal_samples = df.drop(df_cluster.index, axis=0)
    id_test = [i for i in X_test.index if i in normal_samples.index]
    X_test_normal = X_test.loc[id_test, :]
    y_test_normal = y_test.loc[id_test, :]

    n_normal = X_test_normal.shape[0]
    cluster_counts = df_cluster["cluster"].value_counts().sort_index(ascending=True).values

    # normalize counts
    n_total = n_normal + cluster_counts.sum()
    normal_weight = n_normal / n_total
    cluster_weights = list(map(lambda x: x/n_total, cluster_counts))
    if normal_weight + sum(cluster_weights) != 1:
        raise ValueError("Weights do not sum up to one")

    # recalculate accuracy for hyper using only normal samples
    hyper_accs = calculate_hyper_acc(hyper_models, X_test_normal, y_test_normal,
                                     metric_function=metric_function)
    # select best hyper model
    best_hyper_acc = max(hyper_accs.values())

    # calculate weighted accuracy
    weighted_acc_hyper = normal_weight * best_hyper_acc

    # calculate cluster accuracies
    cluster_accs = calculate_cluster_acc(cluster_models, cluster_preds, cluster_y,
                                         metric=metric_function)
    
    # calculate weighted accuracy for each of the clusters:
    weighted_acc_clusters = list(map(lambda x,y: x*y, cluster_weights, cluster_accs.values()))

    # sum it all up
    weighted_acc = weighted_acc_hyper + sum(weighted_acc_clusters)
    return weighted_acc

def plot_error_measures(model_df):
    import plotly.express as px
    
    df = pd.DataFrame(dict(
        r=model_df.values,
        theta=model_df.index))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()

def add_figure(fig, models, id):
    import plotly.graph_objects as go

    model = models.iloc[id]

    fig.add_trace(go.Scatterpolar(
      r=model.values,
      theta=model.index,
      fill='toself',
      name=model.name
))

def plot_error_multi(top_models):
    fig = go.Figure()

    for i in range(top_models.shape[0]):
        add_figure(fig, top_models, i)

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1.5]
        )),
    showlegend=False
    )

    fig.show()