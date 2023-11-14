import ipywidgets as widgets

# # define widget buttons
# buttons = widgets.ToggleButtons(
#     options=[0,1,2, 'All'],
#     description='Speed:',
#     disabled=False,
#     button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
# )

buttons = widgets.SelectMultiple(
    options=[0, 1, 2],
    value=[0],
    #rows=10,
    description='Channels',
    disabled=False
)

int_range = widgets.FloatSlider(
    value=99,
    min=85,
    max=100.0,
    step=0.1,
    description='Normalization:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

int_range_v = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Zi',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
