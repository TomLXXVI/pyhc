from copy import deepcopy
from bqplot import *
import IPython.display
from ipywidgets import HBox, VBox, Layout, Text, HTML


ASPECT_RATIO = 16.0 / 9.0  # width / height
FIG_HEIGHT = 750  # px
FIG_WIDTH = FIG_HEIGHT * ASPECT_RATIO
COLORS = ['skyblue', 'red', 'gold', 'olivedrab',
          'darkorange', 'darkcyan', 'coral', 'limegreen',
          'mediumpurple', 'sandybrown']
PARAM_DICT = {
        'figure': {
            'title': 'New Figure',
            'height': f'{FIG_HEIGHT}px',
            'width': f'{FIG_WIDTH}px'
        },
        'x-axis': {
            'title': 'x-axis',
            'tick-format': '0.0f',
            'min': None,
            'max': None,
            'tick-values': None,
        },
        'y-axis': {
            'title': 'y-axis',
            'tick-format': '0.0f',
            'min': None,
            'max': None,
            'tick-values': None,
        },
    }


class _Graph:
    def __init__(self, param_dict=None):
        self.legend = None
        self.toolbar = None
        self.widget = None

        self._update_param_dict(param_dict)

        self.x_scale = LinearScale(
            min=self.param_dict['x-axis']['min'],
            max=self.param_dict['x-axis']['max']
        )
        self.y_scale = LinearScale(
            min=self.param_dict['y-axis']['min'],
            max=self.param_dict['y-axis']['max']
        )
        self.x_ax = Axis(
            label=self.param_dict['x-axis']['title'],
            scale=self.x_scale,
            tick_format=self.param_dict['x-axis']['tick-format'],
            tick_values=self.param_dict['x-axis']['tick-values'],
        )
        self.y_ax = Axis(
            label=self.param_dict['y-axis']['title'],
            scale=self.y_scale,
            orientation='vertical',
            tick_format=self.param_dict['y-axis']['tick-format'],
            tick_values=self.param_dict['y-axis']['tick-values'],
        )
        self.fig = Figure(
            title=self.param_dict['figure']['title'],
            axes=[self.x_ax, self.y_ax],
            layout=Layout(
                width=self.param_dict['figure']['width'],
                height=self.param_dict['figure']['height']
            )
        )

    def _update_param_dict(self, param_dict):
        self.param_dict = deepcopy(PARAM_DICT)
        if param_dict is not None:
            for k1, d1 in param_dict.items():
                if k1 in self.param_dict.keys():
                    d2 = self.param_dict[k1]
                    d2.update(d1)

    def add_legend(self):
        pass

    def add_toolbar(self):
        self.toolbar = Toolbar(figure=self.fig)

    def draw(self):
        widget_items = [self.fig]
        if self.toolbar and self.legend:
            widget_items.append(HBox([self.toolbar, self.legend], layout=Layout(justify_content='space-around')))
        elif self.toolbar:
            widget_items.append(self.toolbar)
        elif self.legend:
            widget_items.append(self.legend)
        self.widget = VBox(widget_items)

    def display(self):
        IPython.display.display(self.widget)

    def get_widget(self):
        return self.widget


class LineGraph(_Graph):
    def __init__(self, x_data, y_data, labels, param_dict):
        super().__init__(param_dict)
        self.labels = []
        colors = []
        for i, label in enumerate(labels):
            color = COLORS[i % len(COLORS)]
            self.labels.append((label, color))
            colors.append(color)
        self.lines = Lines(
                x=x_data,
                y=y_data,
                scales={'x': self.x_scale, 'y': self.y_scale},
                colors=colors
        )
        self.fig.marks = [self.lines]

    def add_legend(self):
        items = ['<ul style="list-style-type: none">']
        for label in self.labels:
            items.append(f'<li style="background: {label[1]}; display: inline; padding: 5px"><b>{label[0]}</b></li>')
        items.append("</ul>")
        html_str = ''.join(items)
        self.legend = HTML(value=html_str)

    def update_y_data(self, y_data):
        self.lines.y = y_data


class GraphMatrix:
    def __init__(self, *graphs, dimensions=None):
        self.graphs = graphs
        self.grid = None
        if dimensions:
            self.num_rows = dimensions[0]
            self.num_cols = dimensions[1]
        else:
            self.num_rows = len(self.graphs)
            self.num_cols = 1
        self.create()

    def create(self):
        i = 0
        matrix = []
        for r in range(self.num_rows):
            row = []
            for c in range(self.num_cols):
                row.append(self.graphs[i].widget)
                i += 1
            matrix.append(HBox(row, layout=Layout(justify_content='space-between')))
        self.grid = VBox(matrix, layout=Layout(align_content='space-between'))

    def display(self):
        IPython.display.display(self.grid)

    def get_widget(self):
        return self.grid
