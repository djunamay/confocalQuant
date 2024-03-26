import numpy as np
import pandas as pd
import statsmodels.api as sm
import numba as nb
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import mixedlm


def compute_nested_anova(resE, score, group, nested_col):
    """
    Compute nested ANOVA statistics and return the results as a formatted text.

    Parameters:
    - resE (pd.DataFrame): DataFrame containing data for analysis.
    - score (str): Dependent variable.
    - group (str): Grouping variable for the ANOVA.
    - nested_col (str): Nested column for grouping.

    Returns:
    - str: Formatted text containing ANOVA coefficients, 95% CI, and p-value.
    """
    # Fit a two-level ANOVA model
    model = mixedlm(score + '~' + group, resE, groups=resE[nested_col])
    result = model.fit()

    p = result.pvalues[1]
    
    p_scientific = "{:e}".format(p)

    coef = result.params[1]
    lower = result.conf_int().iloc[1][0]
    upper = result.conf_int().iloc[1][1]
    text0 = 'Coef: ' + str(np.round(coef,3)) + '\n95% CI: [' + str(np.round(lower,3)) + ',' + str(np.round(upper,3)) + '] \n' + 'p: ' + str(p_scientific)
    
    return text0   

@nb.njit(parallel=True)
def get_expectations(M_unique, masks_sele, probs_sele, vals_sele, E, V, N):
    """
    Calculate expectations, variance, and count for each unique mask value.

    Parameters:
    - M_unique (int): Number of unique mask values.
    - masks_sele (np.ndarray): Selected mask values.
    - probs_sele (np.ndarray): Probabilities associated with the mask values.
    - vals_sele (np.ndarray): Values associated with the mask values.
    - E (np.ndarray): Array to store expectations.
    - V (np.ndarray): Array to store variance.
    - N (np.ndarray): Array to store count.

    Returns:
    - None: Modifies the input arrays in-place.
    """
    for M in nb.prange(M_unique):
        index = masks_sele==M
        temp_probs = probs_sele[index]
        temp_vals = vals_sele[index]

        #E_uniform[M] = np.mean(temp_vals)
        P = temp_probs/np.sum(temp_probs)
        E[M] = np.dot(temp_vals, P)
        V[M] = np.dot((np.power((temp_vals-E[M]),2)), P)/E[M]
        N[M] = np.sum(index)