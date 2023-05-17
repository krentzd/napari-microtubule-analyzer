import napari
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker
import numpy as np

import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, QPushButton

from .degree_of_radiality import compute_degree_of_radiality

# Show image stack in napari viewer and dynamically adjust plot for changing number of slices
# Also update viewer to show corresponding DoR overlay for changing number of slices

# 1) Plot DoR for 1 image
# 2) Plot for various populations and show significance

# Use a thread_worker

# Change flow of function to enable DoR prediction on single image

class RadialityPlotter(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        graph_container = QWidget()

        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground(None)

        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self._graphics_widget)

        self._labels = QWidget()
        self._labels.setLayout(QVBoxLayout())
        self._labels.layout().setSpacing(0)

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)

        num_slices_container = QWidget()
        num_slices_container.setLayout(QHBoxLayout())

        label = QLabel('Number of slices')
        num_slices_container.layout().addWidget(label)
        self._sp_num_slices = QSpinBox()
        self._sp_num_slices.setMinimum(1)
        self._sp_num_slices.setMaximum(10)
        self._sp_num_slices.setValue(3)
        num_slices_container.layout().addWidget(self._sp_num_slices)
        num_slices_container.layout().setSpacing(0)
        self.layout().addWidget(num_slices_container)

        btn_compute = QPushButton("Compute")
        btn_compute.clicked.connect(self._compute)
        self.layout().addWidget(btn_compute)

        self.graph = self._graphics_widget.addPlot()

    def _compute(self):
        self._reset_plot()
        selected_images = self.selected_image_layers()
        for images, color in zip(selected_images, ['red', 'green', 'yellow', 'blue']):
            self.plot_radiality(images.data, color, images.name)

    def plot_radiality(self, images, color='red', name='line1', show_mean=True):
        num_of_slices = self._sp_num_slices.value()
        degree_of_radiality, dor_images = compute_degree_of_radiality(images, num_of_slices)
        # self._add_results(dor_images)
        self.graph.addLegend()
        if show_mean:
            mean_dor = np.mean(degree_of_radiality, axis=0)
            std_dor = np.std(degree_of_radiality, axis=0)

            x = np.array([i for i in range(1, num_of_slices + 1)])
            y = mean_dor
            top = np.array([val + error for val, error in zip(np.nditer(mean_dor), np.nditer(std_dor))])
            bottom = np.array([val - error for val, error in zip(np.nditer(mean_dor), np.nditer(std_dor))])

            error = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, beam=0.1, pen=color)
            self.graph.addItem(error)
            self.graph.plot(x,y, pen=color, name=name)

    def selected_image_layers(self):
        return [layer for layer in self._viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    def _reset_plot(self):
        if not hasattr(self, "graph"):
            self.graph = self._graphics_widget.addPlot()
        else:
            self.graph.clear()

    def _add_results(self, images):
        self._viewer.add_image(images)
