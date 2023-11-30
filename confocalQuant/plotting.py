import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from scipy.stats import ttest_ind

def extract_sbatch_parameters(file_path):
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

def plot_effect(data, condition, control, treatment, x_name):
    plt.figure(figsize=(10,10))
    index = (data[condition]==control) | (data[condition]==treatment)

    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.5, .85)})

    # assigning a graph to each ax
    sns.boxplot(data=data[index], x=x_name, y=condition,orient="h", ax=ax_box, hue=condition, width=0.5)
    sns.kdeplot(data=data[index], x=x_name, ax=ax_hist, hue=condition)
    #plt.axvline(x=np.median(data[name][data['condition']=='vehicle']), color='red', linestyle='--', label='Vertical Line')
    #plt.axvline(x=np.median(data[name][data['condition']==condition]), color='red', linestyle='--', label='Vertical Line')

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    #plt.show()
    
def plot_by_well(data, name, treatment):
    data['MeanValue'] = data.groupby('wellname')[name].transform('mean')

    temp = data[['condition', 'MeanValue']].drop_duplicates()
    temp = temp[(temp['condition']=='vehicle') | (temp['condition']==treatment)]

    ax = sns.boxplot(x='condition', y='MeanValue', data=temp)

    # Add p-value annotation
    grouped_data = temp.groupby('condition')['MeanValue']
    group_a = grouped_data.get_group('vehicle')
    group_b = grouped_data.get_group(treatment)

    t_stat, p_value = ttest_ind(group_a, group_b)

    # Annotate the plot with the p-value
    x_pos = 0.5  # Adjust the x-position as needed
    y_pos = max(ax.get_ylim())  # Place annotation at the top of the plot
    plt.text(x_pos, y_pos, f'P-Value: {p_value:.3f}', ha='center', va='center', color='red')

def plot_violin(data, name, colname, baseline, treatment):
    # Create a violinplot
    #plt.figure(figsize=(8, 6))
    data2 = data[(data[colname]==baseline) | (data[colname]==treatment)]
    ax = sns.violinplot(x='condition', y=name, data=data2)

    # Add p-value annotation
    grouped_data = data2.groupby(colname)[name]
    group_a = grouped_data.get_group(baseline)
    group_b = grouped_data.get_group(treatment)

    t_stat, p_value = ttest_ind(group_a, group_b)

    # Annotate the plot with the p-value
    x_pos = 0.5  # Adjust the x-position as needed
    y_pos = max(ax.get_ylim())  # Place annotation at the top of the plot
    plt.text(x_pos, y_pos, f'P-Value: {p_value:.3f}', ha='center', va='center', color='red')
    

def return_results(path_to_sbatch_file):
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

    all_mat = np.lib.format.open_memmap(path.join('../.'+folder, 'mat.npy'), shape=(NZi, xi_per_job, yi_per_job, len(channels)), dtype=float, mode=mode)
    all_masks = np.lib.format.open_memmap(path.join('../.'+folder, 'masks.npy'), shape=(NZi, xi_per_job, yi_per_job), dtype='uint16', mode=mode)
    all_Y = np.lib.format.open_memmap(path.join('../.'+folder, 'Y_filtered.npy'), shape=(Ncells, len(channels)+2), dtype=float, mode=mode)
    Ncells_per_job = np.lib.format.open_memmap(path.join('../.'+folder, 'Ncells_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    Nzi_per_job = np.lib.format.open_memmap(path.join('../.'+folder, 'Nzi_per_job.npy'), shape=(Njobs,1), dtype=int, mode=mode)
    
    return all_mat, all_masks, all_Y, Ncells_per_job, Nzi_per_job, cells_per_job, zi_per_job


def concatenate_Y(files, all_Y, cells_per_job, Ncells_per_job, nuclear_col_idx, soma_col_idx, nuclear_percentile, soma_percentile, colnames):
    res = []
    for ID in range(len(files)):
        start = ID*cells_per_job
        end = start + Ncells_per_job[ID][0]
        temp = all_Y[start:end]
        temp = temp[(temp[:,nuclear_col_idx]>np.percentile(temp[:,nuclear_col_idx], nuclear_percentile)) & (temp[:,soma_col_idx]>np.percentile(temp[:,soma_col_idx], soma_percentile))]

        res.append(temp)
    data = pd.DataFrame(np.vstack(res))
    
    data.columns = colnames
    data['filename'] = [files[int(x)].split('.')[0] for x in data['ID']]
    data['wellname'] = [x.split('-')[0] for x in data['filename']]

    return data


def add_metadata(data, path_to_meta):
    df = pd.read_csv(path_to_meta)
    dictionary1 = dict(zip(df['filename'], df['treatment']))
    dictionary2 = dict(zip(df['filename'], df['line']))

    data['treatment'] = [dictionary1[x] for x in data['filename']]
    data['line'] = [dictionary2[x] for x in data['filename']]
    
def exclude_files(exclude, files):
    exclude = set(np.argwhere([x.split('.')[0] in exclude for x in files]).reshape(-1))
    IDS = np.argwhere([x not in exclude for x in range(len(files))]).reshape(-1)
    return IDS

def modify_keep_files(keep_files, ids_to_remove):
    idx = np.argwhere([x in ids_to_remove for x in keep_files])
    keep_files = keep_files.pop(idx)
    return keep_files