# from ordered_set import OrderedSet
import pandas as pd
import math

def remove_weird(df, val):
    """Function used to remove weird values (specific to this dataset)"""
    id = df[df.isin([val]).any(1)].index
    df.drop(id, axis=0, inplace=True)

def to_category(df, *columns):
    """
    Function used for applying categorical encoding 
    to columns of a DataFrame

    INPUTS:
    -df: pd.DataFrame
        DataFrame to encode
    -*columns: 
        Column names to be changed into category
    """
    for col in columns:
        df[col] = df[col].astype('category')

def column_encoder(X_change, X_ref):
    """
    Function used for applying specific column 
    encoding following the guidelines of another
    DataFrame.

    INPUTS:
    -X_change: pd.DataFrame
        DataFrame to encode
    -X_ref: pd.DataFrame
        DataFrame taken as reference
    """
    for col in X_change.columns.tolist():
        dtype = X_ref[col].dtype
        X_change[col] = X_change[col].astype(dtype)

def element_remover(l, *elements):
    # removes elements from list
    for el in elements:
        try:
            l.remove(el)
        except ValueError:
            pass
        
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
        # create name for binned variable
        varname = '{}_binned'.format(var)

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
        var_binned = pd.cut(df_discrete[var], bins=n_bins, include_lowest=True, ordered=True) # cut variable into intervals
        # get position of the original variable
        binned_col_position = df_discrete.columns.get_loc(var)+1
        
        # insert binned variable in the DataFrame
        df_discrete.insert(binned_col_position, varname, var_binned, 2)
    
    return df_discrete
