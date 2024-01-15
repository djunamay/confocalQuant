import numpy as np
import pandas as pd
from os import path
from typing import List, Tuple
from aicsimageio import AICSImage
from skimage.segmentation import find_boundaries
from matplotlib.patches import Rectangle
import czifile
import xml.etree.ElementTree as ET

def get_id_data(ID, zi_per_job, Nzi, mat, masks):
    """
    Extract data and masks corresponding to a specific ID.

    Parameters:
    - ID: int
        The identifier for the specific data subset.
    - zi_per_job: int
        The number of z-planes per job.
    - Nzi: List[Tuple[int]]
        List containing information about the number of z-planes for each ID.
    - mat: np.ndarray
        The original data matrix.
    - masks: np.ndarray
        The masks corresponding to the data.

    Returns:
    - Tuple[np.ndarray, np.ndarray]
        A tuple containing the extracted data matrix and masks.

    This function calculates the start and end indices based on the provided ID, zi_per_job, and Nzi,
    then extracts the corresponding subset of data and masks and returns them as a tuple.
    """
    start = ID*zi_per_job
    end = start + Nzi[ID][0]
    mat_sele = mat[start:end]
    mask_sele = masks[start:end]
    return mat_sele.copy(), mask_sele.copy()


def return_results(path_to_sbatch_file, prefix):
    """
    Retrieve and return results from the output files specified in the Slurm sbatch script.

    Parameters:
    - path_to_sbatch_file (str): Path to the Slurm sbatch script file.
    - prefix (str): Prefix to be added to the folder path for result files.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]: A tuple containing the following:
        - all_mat (np.ndarray): 4D array representing the 'mat.npy' file data.
        - all_masks (np.ndarray): 3D array representing the 'masks.npy' file data.
        - Nzi_per_job (np.ndarray): 1D array representing the 'Nzi_per_job.npy' file data.
        - cells_per_job (int): Number of cells per job.
        - zi_per_job (int): Value of zi_per_job parameter.
        - randID_per_job (np.ndarray): 1D array representing the 'randomID_per_job.npy' file data.
    """
    # get data and params
    params = extract_sbatch_parameters(path_to_sbatch_file)
    folder = params['--folder'][0][1:-1]
    NZi = int(params['--NZi'][0])
    xi_per_job = int(params['--xi_per_job'][0])
    yi_per_job = int(params['--yi_per_job'][0])
    cells_per_job = int(params['--cells_per_job'][0])
    Ncells = int(params['--Ncells'][0])
    Njobs = int(params['--Njobs'][0])
    channels = [int(x) for x in params['--channels']]
    mode = 'r'
    zi_per_job = int(params['--zi_per_job'][0])

    all_mat = np.lib.format.open_memmap(path.join(prefix + folder, 'mat.npy'), shape=(NZi, xi_per_job, yi_per_job, len(channels)), dtype=float, mode=mode)
    all_masks = np.lib.format.open_memmap(path.join(prefix + folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    Nzi_per_job = np.lib.format.open_memmap(path.join(prefix + folder, 'Nzi_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    randID_per_job = np.lib.format.open_memmap(path.join(prefix + folder, 'randomID_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    
    return all_mat, all_masks, Nzi_per_job, cells_per_job, zi_per_job, randID_per_job

def return_non_unique_indices(df):
    """
    Identify and print non-unique indices in the columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the data.

    Returns:
    - temp: pandas DataFrame
        A DataFrame with unique values for each column.

    This function iterates through the columns of the input DataFrame and prints the column names
    where non-unique indices are found. It returns a DataFrame with unique values for each column.

    Example:
    return_non_unique_indices(my_dataframe)
    """
    res = []
    names = []
    for col in df.columns:
        try:
            r = df[col].unique()
        except TypeError:
            r = np.unique([str(x) for x in df[col]])
        res.append(r)
        names.append(col)
    temp = pd.DataFrame(res)
    temp.index = names
    non_unique_indices = temp.index[np.argwhere(np.array([np.sum([x!=None for x in temp.iloc[y]]) for y in range(temp.shape[0])])>1).reshape(-1)]
    print('\n'.join(non_unique_indices))
    return temp

def print_metadata(path_to_czi):
    """
    Print metadata information from a CZI file.

    Parameters:
    - path_to_czi: The file path to the CZI file.

    This function reads the metadata from the CZI file specified by the file path,
    prints the metadata information, and returns nothing.

    Note: This function relies on the czifile.CziFile and AICSImage classes.

    Example:
    print_metadata("/path/to/your/file.czi")
    """
    with czifile.CziFile(path_to_czi) as czi:
            # Read the metadata from the CZI file
            metadata = czi.metadata()
            img = AICSImage(path_to_czi)
            root = ET.fromstring(metadata)
    print(metadata)
        
def extract_sbatch_parameters(file_path):
    """
    Extract parameters from a Slurm sbatch script file and return them as a dictionary.

    Parameters:
    - file_path (str): Path to the sbatch script file.

    Returns:
    - dict: A dictionary containing the extracted parameters, where keys
            are parameter names and values are either single values or lists
            of values if parameters are specified in list form.
    """
    parameters = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith("#"):
                continue

            # Extract key-value pairs or parameters in list form
            parts = line.split()
            if len(parts) >= 2:
                key, *values = parts
                if '\\' in values:
                    # Handle parameters in the form of "--key value1 value2 \"
                    values = values[:values.index('\\')]
                parameters[key] = values
            elif len(parts) == 1:
                # Handle parameters in list form
                parameters.setdefault('list_parameters', []).extend(parts)

    return parameters

def compute_avs(data, filename, treatment, line, value):
    """
    Compute average values based on grouping by filename, treatment, and line.

    Parameters:
    - data: The DataFrame containing the data.
    - filename: The column representing filenames in the DataFrame.
    - treatment: The column representing treatment conditions in the DataFrame.
    - line: The column representing line conditions in the DataFrame.
    - value: The column representing the values to compute averages.

    Returns:
    - mean_per_filename: Average values grouped by filename.
    - mean_per_condition: Average values grouped by treatment and line.
    """
    mean_per_filename = data.groupby(filename)[value].mean()
    mean_per_condition = data.groupby([treatment, line])[value].mean()
    return mean_per_filename, mean_per_condition