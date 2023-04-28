import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from abc import ABC, abstractclassmethod 
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_VIF(X_proc):
    """
    Function used for calculating VIF

    INPUTS:
    -X_proc: pd.DataFrame
        Preprocessed data on which VIF is to be calculated
    
    RETURNS:
    -vif: pd.DataFrame
        Contains VIF Factor mapped to each feature present in X_proc
    
    -vif_corr: pd.DataFrame
        Contain VIF Factor mapped to each feature for which VIF > 10
    """
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X_proc.values, i) for i in range(X_proc.shape[1])]
    vif["features"] = X_proc.columns
    corr_features = vif.loc[vif["VIF Factor"] >= 10, "features"].tolist()
    vif_corr = vif.loc[vif["features"].isin(corr_features), :]
    return vif, vif_corr

def corr_matrix(X_proc, features):
    """
    Function used for plotting correlation matrix

    INPUTS:
    -X_proc: pd.DataFrame
        Preprocessed data on which correlation matrix is to be plotted
    
    -features: list
        List of features for which the correlation matrix is to be 
        plotted. Only numerical features are allowed

    """
    sns.set()
    f = plt.figure()
    plt.matshow(X_proc[features].corr(), fignum=f.number)
    plt.xticks(range(X_proc[features].shape[1]), X_proc[features].columns, fontsize=14, rotation=45)
    plt.yticks(range(X_proc[features].shape[1]), X_proc[features].columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

def features_pairplot(X_test, experiments, metric_name, n_cols): 
    cols = X_test.columns.values.tolist()
    cols.append(metric_name)
    vis_data = pd.DataFrame(experiments[:, n_cols:], columns=cols)
    sns.pairplot(vis_data, x_vars=vis_data.columns.values.tolist()[:-1], y_vars=metric_name,
                 kind="reg", plot_kws=dict(scatter = True, marker = '+', color = "r",
                 scatter_kws=dict(alpha = 0.3, linewidths = 0.1, color = "b")))
    plt.show()

def general_pairplot(X):
    sns.set()
    sns.pairplot(X, vars=X.columns.values.tolist(), diag_kind="hist",
                diag_kws=dict(color="k"),
                kind="reg", plot_kws=dict(scatter = True, color = "blueviolet",
                scatter_kws=dict(alpha = 0.5, color = "k")))
    plt.show()

def generate_nas(X_test, max_samples):
    na_df = X_test.to_numpy()
    n_samples = np.arange(1, max_samples)
    sample_index = np.arange(0, max_samples)
    n_cols = na_df.shape[1]
    # sacar numero aleatorio de muestras tantas veces como cols tengamos
    num_samples = np.random.choice(n_samples, size=n_cols) # nos da el numero de muestras a sacar
    # para cada columna
    for i in np.arange(num_samples.shape[0]):
        # sacamos tantos IDs como nos diga
        na_id = np.random.choice(sample_index, size=num_samples[i])
        # convertimos esas muestras en NAs
        na_df[na_id, i] = np.nan
    return pd.DataFrame(na_df, columns=X_test.columns)

def count_nas(df):
    cols = df.columns.tolist()
    na_counts = [df[i].isnull().sum()  for i in cols]
    return na_counts

def na_imputer(na_df, X_ref):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    INPUTS_NUM = X_ref.select_dtypes(include=['int64','float64']).columns.tolist()
    INPUTS_CAT = X_ref.select_dtypes(include=['category']).columns.tolist()

    numeric_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Pipeline them
    na_imputer = ColumnTransformer(transformers=[
            ('num', numeric_imputer, INPUTS_NUM),
            ('cat', cat_imputer, INPUTS_CAT)],
            verbose_feature_names_out=False).set_output(transform="pandas")
    
    imputed_df = na_imputer.fit_transform(na_df)
    return imputed_df

def calculate_metrics(model, X, y_true, metric_func):
    na_pred = model.predict(X)
    metric = metric_func(y_true, na_pred)
    return metric

def na_pipe(model, X_test, y_true, max_samples, metric_func, n_exp=10):
    results = []
    for i in np.arange(n_exp):
        # generate nas
        na_df = generate_nas(X_test, max_samples)
        # calculate number of nas
        na_counts = count_nas(na_df)
        # calculate percentage of nas
        na_percs = list(map(lambda x: x / na_df.shape[0], na_counts))
        # impute NAs with values
        test_na = na_imputer(na_df, X_test)
        # calculate metric
        metric = calculate_metrics(model, test_na, y_true, metric_func)
        # append results
        results.append(na_counts + na_percs + [metric])
    return np.array(results)

def max_min_vals(X, n_exp):
    values = []
    columns = X.columns.tolist()
    
    for column in columns:
        col = X.loc[:, column]

        # Calculate first quartile
        first_quartile = col.quantile(0.25)

        # Calculate third quartile
        third_quartile = col.quantile(0.75)

        # Calculate IQR
        iqr = stats.iqr(col.dropna())

        # Calculate limits for which outliers are considered
        low_min_val = first_quartile - 3 * iqr
        high_min_val = first_quartile - 1.5 * iqr
        low_max_val = third_quartile + 3 * iqr
        high_max_val = third_quartile + 1.5 * iqr

        # Generate normal distribution in those limits
        low_x = np.linspace(low_min_val, high_min_val, num=1000)
        norm_low = stats.norm(loc = low_x.mean(), scale = np.std(low_x))            
        high_x = np.linspace(low_max_val, high_max_val, num=1000)
        norm_high = stats.norm(loc = high_x.mean(), scale = np.std(high_x))

        # Generate random data for outliers
        min_vals_col = norm_low.rvs(size=(col.shape[0], n_exp))
        max_vals_col = norm_high.rvs(size=(col.shape[0], n_exp))

        values.append(min_vals_col + max_vals_col)

    return np.array(values)

def generate_outliers(X_test, max_samples, values, exp):
    outlier_df = X_test.to_numpy()
    n_samples = np.arange(1, max_samples)
    sample_index = np.arange(0, max_samples)
    n_cols = outlier_df.shape[1]
    # sacar numero aleatorio de muestras tantas veces como cols tengamos
    num_samples = np.random.choice(n_samples, size=n_cols) # nos da el numero de muestras a sacar
    # para cada de las columnas
    for i in np.arange(num_samples.shape[0]):
        # sacamos tantos IDs como nos diga
        outlier_ids = np.random.choice(sample_index, size=num_samples[i])
        # a cada uno le asignamos un valor outlier aleatorio
        outlier_values = np.random.choice(values[i, :, exp], size=num_samples[i])
        for sample_id, value in zip(outlier_ids, outlier_values):
            outlier_df[sample_id, i] = value
    return num_samples.tolist(), pd.DataFrame(outlier_df, columns=X_test.columns)

def outlier_pipe(model, X_test, y_true, max_samples, metric_func, n_exp=10):
    X_out = X_test.reset_index(drop=True)
    num_cols = X_out.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X_out.select_dtypes(include=['category']).columns.tolist()
    outlier_values = max_min_vals(X_out[num_cols], n_exp)
    results = []
    for i in np.arange(n_exp):
        # generate outliers
        outlier_counts, outliers = generate_outliers(X_out[num_cols], max_samples, outlier_values, i)
        outlier_df = pd.concat([outliers, X_out[cat_cols]], axis=1)
        # calculate percentage of nas (which will be turned to outliers)
        na_percs = list(map(lambda x: x / outlier_df.shape[0], outlier_counts))
        # calculate metric
        metric = calculate_metrics(model, outlier_df, y_true, metric_func=metric_func)
        exp_res = [outlier_counts, na_percs, [metric]]
        results.append([item for sublist in exp_res for item in sublist])
    return np.array(results)