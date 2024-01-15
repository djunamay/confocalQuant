import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
from aicsimageio import AICSImage
from skimage.segmentation import find_boundaries
import czifile

def return_channel_moments_per_im(files, path_to_parent, nchannels, max_val):
    """
    Calculate mean, standard deviation, and percentage of values per channel in each image.

    Parameters:
    - files (list): List of CZI file names.
    - path_to_parent (str): Path to the parent directory.
    - nchannels (int): Number of channels in the images.
    - max_val (int): Maximum pixel value indicating clipping.

    Returns:
    - tuple: Three arrays containing mean, standard deviation, and percentage of clipped values per channel.
    """
    
    out_means = np.empty((len(files), nchannels))
    out_stds = np.empty((len(files), nchannels))
    out_percent_clipped = np.empty((len(files), nchannels))
    
    for x in tqdm(range(len(files))):
        czi_file_path = path_to_parent+files[x]
        img = AICSImage(czi_file_path)
        temp1 = []
        temp2 = []
        for i in range(nchannels):
            d = img.data[:,i,:,:,:]
            out_means[x][i] = np.mean(d)
            out_stds[x][i] = np.std(d)
            out_percent_clipped[x][i] = (np.sum(d==max_val)/np.prod(d.shape))*100
        
    return out_means, out_stds, out_percent_clipped

def get_day_and_time(df):
    """
    Extract day and time information from the 'AcquisitionDateAndTime' column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'AcquisitionDateAndTime' column.

    Returns:
    - tuple: Two arrays containing day and time information.
    """
    time = []
    for i in df['AcquisitionDateAndTime']:
        x = i[0].split('T')[1].split('.')[0].split(':')
        time.append(int("".join([str(item) for item in x])))
    day = [int(x[0].split('-')[2].split('T')[0]) for x in df['AcquisitionDateAndTime']]
    return np.array(day), np.array(time)


def get_metadata(czi_file_path):
    """
    Extract metadata information from a CZI file.

    Parameters:
    - czi_file_path (str): Path to the CZI file.

    Returns:
    - dict: Dictionary containing various metadata parameters.
    """
    # Open the CZI file
    with czifile.CziFile(czi_file_path) as czi:
        # Read the metadata from the CZI file
        metadata = czi.metadata()
        img = AICSImage(czi_file_path)
        root = ET.fromstring(metadata)

        RI = root.find(".//RefractiveIndex").text
        PinholeSizeAiry = root.find(".//PinholeSizeAiry").text    
        SizeX = root.find(".//SizeX").text
        SizeY = root.find(".//SizeY").text
        SizeZ = root.find(".//SizeZ").text
        SizeC = root.find(".//SizeC").text

        fluor = [x.text for x in root.findall('.//Fluor')]
        exwave = [x.text for x in root.findall('.//ExcitationWavelength')]
        emwave = [x.text for x in root.findall('.//EmissionWavelength')]
        #pinhole_diam = [x.text for x in root.findall('.//Position')]
        
        parameter_id = "MTBLSMPinholeDiameter"
        pinhole_diam = [x.text for x in root.findall(f'.//ParameterCollection[@Id="{parameter_id}"]/Position')]


        NA = [x.text for x in root.findall('.//NumericalAperture')]
        bits = [x.text for x in root.findall('.//BitsPerPixel')]
        time = [x.text for x in root.findall('.//AcquisitionDateAndTime')]
        intensity = [x.text for x in root.findall('.//Intensity')]

        DigitalGain = [x.text for x in root.findall('.//DigitalGain')]
        DigitalOffset = [x.text for x in root.findall('.//DigitalOffset')]

        TotalMagnification = [x.text for x in root.findall('.//TotalMagnification')]
        TotalAperture = [x.text for x in root.findall('.//TotalAperture')]

        LaserEnableTime = [x.text for x in root.findall('.//LaserEnableTime')]
        #ZStackSliceIndex = [x.text for x in root.findall('.//LineStep')]

        dictionary2 = dict({'RI': RI,
              'PinholeSizeAiry': PinholeSizeAiry,
              'SizeX': SizeX, 
              'SizeY': SizeY,
              'SizeZ': SizeZ,
              'SizeC': SizeC,
              'Fluor': fluor,
              'ExcitationWavelength': exwave,
              'EmissionWavelength': emwave,
              'MTBLSMPinholeDiameter': pinhole_diam,
              'NumericalAperture': NA,
              'BitsPerPixel': bits,
              'AcquisitionDateAndTime': time,
              'Intensity': intensity,
              'DigitalGain': DigitalGain,
              'DigitalOffset': DigitalOffset,
              'TotalMagnification': TotalMagnification,
              'TotalAperture': TotalAperture,
              'LaserEnableTime': LaserEnableTime,
              'pixelsize': img.physical_pixel_sizes})
        
        return dictionary2

#####
    
def return_non_unique_indices(df):
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
    with czifile.CziFile(path_to_czi) as czi:
            # Read the metadata from the CZI file
            metadata = czi.metadata()
            img = AICSImage(path_to_czi)
            root = ET.fromstring(metadata)
    print(metadata)
    



def plot_by(condition, x, y, xlab, ylab):
    for i in np.unique(condition):
        index = condition==i
        plt.scatter(x[index], y[index], label=i)
        plt.legend()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

def plot_hist(path, channel, nbins, scale_log, alpha, color, density):
    img = AICSImage(path)
    plt.hist(img.data[0,channel,:,:,:].ravel(),nbins, alpha=alpha, color=color, density=density)
    if scale_log:
        plt.yscale('log')
    None
    

def impose_segmentation_all(ID, zi_per_job, Nzi, mat, masks, val):
    start = ID*zi_per_job
    end = start + Nzi[ID][0]
    mat_sele = mat[start:end]
    mask_sele = masks[start:end]
    
    o = [find_boundaries(mask_sele[i], mode = 'outer', background = 0) for i in range(mask_sele.shape[0])]
    M = np.stack(o, axis=0)

    superimposed_data = mat_sele.copy()
    for i in range(mat_sele.shape[0]):
        masked = np.where(M[i])
        
        for c in range(mat_sele.shape[-1]):

            superimposed_data[i,:,:,c][masked] = val
        
    return superimposed_data