import numpy as np
import pandas as pd
import statsmodels.api as sm
import numba as nb
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import mixedlm
from tqdm import tqdm 



@nb.njit()
def compute_channel_stats_per_mask(masks_sele, M, probs_sele, vals, temp):
    
    index = masks_sele==M
    temp_probs = probs_sele[index]
    P = temp_probs/np.sum(temp_probs)
    temp_vals = vals[:,index]
        
    E = temp_vals@P.reshape(-1,1)
    E = E.reshape(-1)
    N = np.sum(index)
    
    temp[:len(E)] = E
    temp[-1] = N
    
    return temp
    
@nb.njit(parallel=True)
def compute_stats_all_masks(outputs, M_unique, masks_sele, probs_sele, vals, temp):
    masks = M_unique[1:]
    for i in nb.prange(len(masks)):
        M = masks[i]
        outputs[i] = compute_channel_stats_per_mask(masks_sele, M, probs_sele, vals, temp)

            
def format_stats_as_df(ID, outputs, files, lines, treat, channel_names):
    df = pd.DataFrame(outputs)
    df.columns = channel_names + ['size']
    df['file_ID'] = ID
    df['mask_ID'] = range(1, df.shape[0]+1)
    df['file'] = files[ID]
    df['line'] = lines[ID]
    df['treatment'] = treat[ID]
    return df

def extract_channel(data, discretize = False, thresh = None):
    if discretize:
        return data.ravel()>thresh
    else:
        return data.ravel()
    
def compute_per_cell_stats(file_ids, zi_per_job, Nzi_per_job, probs, all_masks, all_mat, thresh, files, lines, treat, channel_names):

    all_outs = []
    
    for ID in tqdm(file_ids):
        probs_sele, masks_sele, out_float_sele, M_unique = load_im(ID, zi_per_job, Nzi_per_job, probs, all_masks, all_mat)
        nchannels = out_float_sele.shape[-1]
        outputs = np.empty((len(M_unique)-1, nchannels+1))
        nvoxels = np.prod(out_float_sele.shape[:3])
        vals = np.empty((nchannels, nvoxels))
        
        for C in range(nchannels):
            
            T = thresh[C]
            
            if T is not None:
                vals[C] = extract_channel(out_float_sele[:,:,:,C], discretize = True, thresh = T)
            else:
                vals[C] = extract_channel(out_float_sele[:,:,:,C])
        
        temp = np.empty(nchannels+1)
        compute_stats_all_masks(outputs, M_unique, masks_sele, probs_sele, vals, temp)
        
        all_outs.append(format_stats_as_df(ID, outputs, files, lines, treat, channel_names))
        
    return pd.concat(all_outs)


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

#### Add meta-p-values function


# @nb.njit(parallel=True)
# def get_expectations(M_unique, masks_sele, probs_sele, vals_sele, E, V, N):
#     """
#     Calculate expectations, variance, and count for each unique mask value.

#     Parameters:
#     - M_unique (int): Number of unique mask values.
#     - masks_sele (np.ndarray): Selected mask values.
#     - probs_sele (np.ndarray): Probabilities associated with the mask values.
#     - vals_sele (np.ndarray): Values associated with the mask values.
#     - E (np.ndarray): Array to store expectations.
#     - V (np.ndarray): Array to store variance.
#     - N (np.ndarray): Array to store count.

#     Returns:
#     - None: Modifies the input arrays in-place.
#     """
#     for M in nb.prange(M_unique):
#         index = masks_sele==M
#         temp_probs = probs_sele[index]
#         temp_vals = vals_sele[index]

#         #E_uniform[M] = np.mean(temp_vals)
#         P = temp_probs/np.sum(temp_probs)
#         E[M] = np.dot(temp_vals, P)
#         V[M] = np.dot((np.power((temp_vals-E[M]),2)), P)/E[M]
#         N[M] = np.sum(index)