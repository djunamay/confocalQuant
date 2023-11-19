import ipywidgets as widgets

# # define widget buttons
# buttons = widgets.ToggleButtons(
#     options=[0,1,2, 'All'],
#     description='Speed:',
#     disabled=False,
#     button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
# )

text = widgets.HTML(value='')

dropdown_soma = widgets.Dropdown(
    options=[1],
    value=1,
    description='Soma:',
    disabled=False,
)

dropdown_nuc = widgets.Dropdown(
    options=[1],
    value=1,
    description='Nucleus:',
    disabled=False,
)

buttons = widgets.SelectMultiple(
    options=[0, 1, 2, 3],
    value=[0],
    #rows=10,
    description='Show:',
    disabled=False
)

buttons2 = widgets.SelectMultiple(
    options=[0, 1, 2, 3],
    value=[0],
    #rows=10,
    description='Adjust:',
    disabled=False
)

upper_range = widgets.FloatSlider(
    value=99,
    min=0,
    max=100.0,
    step=0.1,
    description='Upper:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

lower_range = widgets.FloatSlider(
    value=1,
    min=0,
    max=100.0,
    step=0.1,
    description='Lower:',
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

int_range_seg = widgets.IntSlider(
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

