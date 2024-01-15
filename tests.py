'''
Run tests
'''

def test_convert_genotypes_to_str():
    """
    Testing function that converts ndarray genotype to string
        - Comparing manually-determined solution to function output for a simple example
    """
    l = [[0,1], [1,1]]
    g = convert_genotypes_to_str(l)
    
    assert_that(g[0]).is_equal_to('0/1').is_true()
    assert_that(g[1]).is_equal_to('1/1').is_true()

def test_return_genotype_counts():
    """
    Testing function that returns genotype counts
        - Comparing manually-determined solution to function output for a simple example
    """
    
    l = [[[0,1], [1,1], [0,0]], [[0,0], [0,0], [1,1]]]
    g = [convert_genotypes_to_str(i) for i in l]
    df = pd.DataFrame(g)
    cts = return_genotype_counts(df)
    n_samples = df.shape[1]
    n_genotypes = np.sum(cts, axis = 1)
    assert_that(np.unique(n_genotypes==n_samples)[0]).is_true()

def test_compute_MAFs():