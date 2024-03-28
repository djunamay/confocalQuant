import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from aicsimageio import AICSImage
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

from .stats import compute_nested_anova

def plot_boxplot_by_treatment(value, line, df, colors, colors2):
    
    avg_lipidspot = df.groupby(['line', 'treatment', 'well'])[value].mean().reset_index(name='av')

    d = avg_lipidspot[avg_lipidspot['line']==line]

    ax = sns.boxplot(data = d, x = 'treatment', showfliers=False, y = 'av', dodge = True, order = ['vehicle', 'CDP-choline', 'rosiglitazone'], palette = colors, width=.5, boxprops=dict(alpha=1), medianprops=dict(color='black', alpha=1), whiskerprops=dict(color='black', alpha=1), capprops=dict(color = 'black', alpha=1))
    sns.stripplot(data=d, x='treatment', y='av', palette = colors2, dodge=True, jitter=True, alpha=1,  order = ['vehicle', 'CDP-choline', 'rosiglitazone'])

    pairs = [(("vehicle"), ("CDP-choline")), (("vehicle"), ("rosiglitazone"))]  # Define pairs to compare
    annotator = Annotator(ax, pairs, data=d, x='treatment', y='av', order = ['vehicle', 'CDP-choline', 'rosiglitazone'])
    annotator.configure(test='t-test_ind', text_format='full', loc='inside', verbose=2, show_test_name=False)
    
    annotator.apply_and_annotate()
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.ylabel(value)
    plt.xlabel('')
    plt.title(line)

    
def plot_hist(path, channel, nbins, scale_log, alpha, color, density):
    """
    Plot histogram of pixel intensities from a specified channel in an AICSImage.

    Parameters:
    - path (str): Path to the AICSImage file.
    - channel (int): Index of the channel to extract pixel intensities.
    - nbins (int): Number of bins in the histogram.
    - scale_log (bool): Whether to scale the histogram in log scale.
    - alpha (float): Transparency of the histogram bars.
    - color (str): Color of the histogram bars.
    - density (bool): Whether to normalize the histogram to form a probability density.

    Returns:
    - None
    """
    img = AICSImage(path)
    T = img.data[0,channel,:,:,:].ravel()
    T[T==0] = 1
    plt.hist(np.log(int_to_float(T)),nbins, alpha=alpha, color=color, density=density)
    None
  
def plot_treatments(df, x, line, colors, lognormal, binwidth,lab_height, grp='well', size=(20,7)):
    """
    Plot treatments comparison for a specific line in the dataframe.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x (str): Variable to be plotted.
    - line (str): Specific line for which treatments are compared.
    - colors (list): List of colors for treatments.
    - lognormal (bool): Whether to use log scale for the variable.
    - binwidth (float): Width of bins in the histogram.
    - lab_height (float): Height of the p-value text annotation.
    - grp (str): Grouping variable for boxplot. Default is 'well'.
    - size (tuple): Size of the plot figure. Default is (20, 7).

    Returns:
    - None
    """
    sns.set(rc={'figure.figsize':size})
    sns.set_style("whitegrid")
    df_temp = df[df['line']==line]

    if lognormal:
        df_temp[x] = np.log(df_temp[x])
    
    f, ax = plt.subplots(3,len(np.unique(df['treatment']))-1, sharex=True, gridspec_kw={"height_ratios": (1,.25,.5)})

    for i,t in enumerate(np.unique(df['treatment'])[:-1]):
        index = (df_temp['treatment']=='vehicle') | (df_temp['treatment']==t)
        
        # compute pvalue 
        text = compute_nested_anova(df_temp[index], x, 'treatment', grp)
        sns.histplot(data=df_temp[index], x=x, ax=ax[0,i], hue='treatment', binwidth=binwidth, element="step", common_norm=False, stat='density', palette = colors, order=('vehicle', t))
        
        ax[0,i].text(x=np.mean(df_temp[index][x]), y=lab_height, s=text, fontsize=12, color='black')
        
        sns.boxplot(data=df_temp[index], x=x, y='treatment',orient="h", ax=ax[1,i], width=.5, dodge=True, palette = colors)
        sns.boxplot(data=df_temp[index], x=x, y=grp,orient="h", ax=ax[2,i], width=.8, dodge=False, hue = 'treatment', palette = colors)
        ax[2,i].get_legend().remove()
        ax[1,i].axes.get_yaxis().set_visible(False)


def plot_lines(df, x, treatment, colors, lognormal, binwidth, grp='well'):
    """
    Plot lines comparison for a specific treatment in the dataframe.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x (str): Variable to be plotted.
    - treatment (str): Specific treatment for which lines are compared.
    - colors (list): List of colors for lines.
    - lognormal (bool): Whether to use log scale for the variable.
    - binwidth (float): Width of bins in the histogram.
    - grp (str): Grouping variable for boxplot. Default is 'well'.

    Returns:
    - None
    """
    sns.set(rc={'figure.figsize':(6,12)})
    sns.set_style("whitegrid")
    df_temp = df[df['treatment']==treatment]

    if lognormal:
        df_temp[x] = np.log(df_temp[x])
        
    f, ax = plt.subplots(6, sharex=True, gridspec_kw={"height_ratios": (1,.5,1,1,.5,1)})

    index = df_temp['line']!='G2'
    
    # compute pvalue 
    text0 = compute_nested_anova(df_temp[index], x, 'line', grp)
    sns.histplot(data=df_temp[index], x=x, ax=ax[0], hue='line', binwidth=binwidth, element="step", common_norm=False, stat='density', palette = colors)
    ax[0].text(x=np.mean(df_temp[index][x])+2, y=.9, s=text0, fontsize=12, color='black')
    sns.boxplot(data=df_temp[index], x=x, y='line',orient="h", ax=ax[1], width=.5, dodge=True, palette = colors)
    sns.boxplot(data=df_temp[index], x=x, y=grp,orient="h", ax=ax[2], width=.8, dodge=False, hue = 'line', palette = colors)

    index = df_temp['line']!='Y622'
    
    # compute pvalue 
    text1 = compute_nested_anova(df_temp[index], x, 'line', grp)
    sns.histplot(data=df_temp[index], x=x, ax=ax[3], hue='line', binwidth=binwidth, element="step", common_norm=False, stat='density', palette = colors)
    ax[3].text(x=np.mean(df_temp[index][x])+2, y=.9, s=text1, fontsize=12, color='black')
    sns.boxplot(data=df_temp[index], x=x, y='line',orient="h", ax=ax[4], width=.5, dodge=True, palette = colors)
    sns.boxplot(data=df_temp[index], x=x, y=grp,orient="h", ax=ax[5], width=.8, dodge=False, hue = 'line', palette = colors)
    
    for i in range(6):
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
    

def plot_scatter(condition, x, y, xlab, ylab):
    """
    Scatter plot grouped by a condition.

    Parameters:
    - condition (np.ndarray): Array containing the grouping condition.
    - x (np.ndarray): Values for the x-axis.
    - y (np.ndarray): Values for the y-axis.
    - xlab (str): Label for the x-axis.
    - ylab (str): Label for the y-axis.

    Returns:
    - None
    """
    for i in np.unique(condition):
        index = condition==i
        plt.scatter(x[index], y[index], label=i)
        plt.legend()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


