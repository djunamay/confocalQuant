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
    
from confocalQuant.segmentation load_3D, int_to_float, get_anisotropy, do_inference, sigmoid, impose_segmentations
from confocalQuant.image import save_mean_proj
from confocalQuant.qc import get_metadata, return_non_unique_indices, get_day_and_time, return_channel_moments_per_im
from confocalQuant.datahandling import get_meta_vectors, return_results, load_im_from_memmap, load_im_from_memmap_ravel
from confocalQuant.stats import compute_per_cell_stats, compute_nested_anova
from confocalQuant.plotting import plot_boxplot_by_treatment, plot_treatments, plot_lines
#process_image (main)

# segmentation

def test_load_3D():

def test_int_to_float():
    
def test_get_anisotropy():
    
def test_do_inference():
    
def test_sigmoid():
    
def test_impose_segmentations():
    
# image 

def test_save_mean_proj():
    
# qc 

def test_get_metadata():
    
def test_return_non_unique_indices():
    
def test_get_day_and_time():

def test_return_channel_moments_per_im():

# datahandling

def test_get_meta_vectors():

def test_return_results():

def test_load_im_from_memmap():
    
def test_load_im_from_memmap_ravel():
    
# stats

def test_compute_per_cell_stats():

def test_compute_nested_anova():

# plotting 

def test_plot_boxplot_by_treatment():
    
def test_plot_treatments():

def test_plot_lines():