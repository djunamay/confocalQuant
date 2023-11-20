import numpy.ma as ma

def compute_single_expectation_per_cell(Y, P, N):
    return ma.sum(ma.multiply(Y,P))*(1/N)

def compute_all_expectations_per_cell(MASK, mat, probs, N):
    res = []
    for i in range(3):
        Y = ma.masked_array(mat[:,:,:,i], MASK)
        P = ma.masked_array(probs, MASK)
        res.append(compute_single_expectation_per_cell(Y,P,N))
    return res

def get_im_stats(masks, mat, probs):
    res = []
    for cell in tqdm(np.unique(masks)[1:]):
        MASK = masks!=cell
        N = np.sum(np.invert(MASK))
        res.append(np.concatenate((compute_all_expectations_per_cell(MASK, mat, probs, N), [N])))
    Y = np.vstack(res)
    return Y