'''
Run tests
'''

from assertpy import assert_that
from aicsimageio import AICSImage
import numpy as np

from confocalQuant.segmentation import load_3D, int_to_float, get_anisotropy, do_inference, sigmoid, impose_segmentations
from confocalQuant.image import save_mean_proj
from confocalQuant.qc import get_metadata, return_non_unique_indices, get_day_and_time, return_channel_moments_per_im
from confocalQuant.data_handling import get_meta_vectors, return_results, load_im_from_memmap, load_im_from_memmap_ravel
from confocalQuant.stats import compute_per_cell_stats, compute_nested_anova
from confocalQuant.plotting import plot_boxplot_by_treatment, plot_treatments, plot_lines
#process_image (main)

# segmentation

def test_load_3D():
    img = AICSImage('./tests/test.czi')

    # check datatype
    out = load_3D(img, [0,1,2,3])
    assert_that(out.dtype==img.dtype).is_true()

    # check loading in default order
    assert_that(np.array_equal(img.get_image_data("ZXY", C=0), out[:,:,:,0])).is_true()
    assert_that(np.array_equal(img.get_image_data("ZXY", C=2), out[:,:,:,2])).is_true()

    # check loading in other order
    out = load_3D(img, [0,3,2,1])
    assert_that(np.array_equal(img.get_image_data("ZXY", C=3), out[:,:,:,1])).is_true()
    assert_that(np.array_equal(img.get_image_data("ZXY", C=1), out[:,:,:,3])).is_true()



# def test_do_inference():
#### need to have a couple of sanity checks; masks should be stitched etc

    
# def test_impose_segmentations():
    
# # image 

# def test_save_mean_proj():
    
# # qc 

# def test_get_metadata():
    
# def test_return_non_unique_indices():
    
# def test_get_day_and_time():

# def test_return_channel_moments_per_im():

# # datahandling

# def test_get_meta_vectors():

# def test_return_results():

# def test_load_im_from_memmap():
#### check that it's the same as loading the original image
    
# def test_load_im_from_memmap_ravel():
    
# # stats

# def test_compute_per_cell_stats():

# def test_compute_nested_anova():

# # plotting 

# def test_plot_boxplot_by_treatment():
    
# def test_plot_treatments():

# def test_plot_lines():



######## nothing to test
# def test_int_to_float():
# def test_get_anisotropy():
# def test_sigmoid():