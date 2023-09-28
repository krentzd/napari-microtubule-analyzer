import napari
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker
import numpy as np
import os
import csv
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QComboBox, QAbstractItemView, QCheckBox
from .degree_of_radiality import compute_degree_of_radiality


class RadialityPlotter(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        graph_container = QWidget()

        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground(None)

        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self._graphics_widget)
        self.graph = self._graphics_widget.addPlot()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(graph_container)

        self.results_table = QTableWidget(self)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout().addWidget(self.results_table)

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

        image_input_container = QWidget()
        image_input_container.setLayout(QHBoxLayout())
        image_input_label = QLabel('Image stack')
        image_input_container.layout().addWidget(image_input_label)
        self._image_layers = QComboBox(self)
        image_input_container.layout().addWidget(self._image_layers)
        image_input_container.layout().setSpacing(0)
        self.layout().addWidget(image_input_container)

        ### If 'No labels' selected, then show dropdown of segmentation algorithms from skimage##
        label_input_container = QWidget()
        label_input_container.setLayout(QHBoxLayout())
        label_input_label = QLabel('Cell segmentation stack')
        label_input_container.layout().addWidget(label_input_label)
        self._label_layers = QComboBox(self)
        label_input_container.layout().addWidget(self._label_layers)
        label_input_container.layout().setSpacing(0)
        self.layout().addWidget(label_input_container)

        self._viewer.layers.events.inserted.connect(self._update_combo_boxes)
        # Need to clean up list of combobox
        self._viewer.layers.events.removed.connect(self._update_combo_boxes)

        btn_compute = QPushButton("Compute")
        btn_compute.clicked.connect(self._compute)
        self.layout().addWidget(btn_compute)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self._save_table)
        self.layout().addWidget(save_button)

        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self._reset_widgets)
        self.layout().addWidget(clear_button)

        self._reset_table()
        self. _update_combo_boxes()
        self._analysis_counter = 0

        self.colour_list = ['red', 'green', 'blue', 'yellow']

    def _update_combo_boxes(self):
        for layer_name in [self._image_layers.itemText(i) for i in range(self._image_layers.count())]:
            layer_name_index = self._image_layers.findText(layer_name)
            self._image_layers.removeItem(layer_name_index)

        if 'No labels' not in [self._label_layers.itemText(i) for i in range(self._label_layers.count())]:
            self._label_layers.addItem('No labels')
        for layer in [l for l in self._viewer.layers if isinstance(l, napari.layers.Image)]:
            if layer.name not in [self._image_layers.itemText(i) for i in range(self._image_layers.count())]:
                self._image_layers.addItem(layer.name)
        for layer in [l for l in self._viewer.layers if isinstance(l, napari.layers.Labels)]:
            if layer.name not in [self._label_layers.itemText(i) for i in range(self._label_layers.count())]:
                self._label_layers.addItem(layer.name)

    def _reset_widgets(self):
        self._analysis_counter = 0
        self._reset_plot()
        self._reset_table()

    def _save_table(self):
        path, ok = QFileDialog.getSaveFileName(
            self, 'Save CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if ok:
            columns = range(self.results_table.columnCount())
            header = [self.results_table.horizontalHeaderItem(column).text()
                      for column in columns]
            with open(path, 'w') as csvfile:
                writer = csv.writer(
                    csvfile, dialect='excel', lineterminator='\n')
                writer.writerow(header)
                for row in range(self.results_table.rowCount()):
                    row_to_write = []
                    for col in columns:
                        item = self.results_table.item(row, col)
                        if item != None:
                            row_to_write.append(item.text())
                        else:
                            row_to_write.append('')
                    writer.writerow(row_to_write)

    def _compute(self):
        selected_image = self.selected_image_layer()
        selected_label = self.selected_label_layer()
        # for images, color in zip(selected_images, ['red', 'green']): #Use cmap to choose colours
        self.plot_radiality(selected_image.data, selected_label.data if selected_label else selected_label, self.colour_list[self._analysis_counter], selected_image.name) #Add color_counter to add new colour for new analysis
        self._analysis_counter += 1
        
    def plot_radiality(self, images, labels, color='red', name='line1', show_mean=True):
        num_of_slices = self._sp_num_slices.value()
        degree_of_radiality, dor_images, cell_labels = compute_degree_of_radiality(images, labels, num_of_slices)

        if labels == None:
            self._viewer.add_labels(cell_labels, name=f'Cell Segmentation {name}')

        self._update_table(degree_of_radiality=degree_of_radiality, folder_name=name)

        self.graph.addLegend()
        if show_mean:
            mean_dor = np.mean(degree_of_radiality, axis=0)
            std_dor = np.std(degree_of_radiality, axis=0)

            x = np.array([i for i in range(1, num_of_slices + 1)])
            y = mean_dor
            top = np.array([val + error for (val, error) in zip(np.nditer(mean_dor), np.nditer(std_dor))])
            bottom = np.array([val - error for (val, error) in zip(np.nditer(mean_dor), np.nditer(std_dor))])

            error = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, beam=0.1, pen=color)
            self.graph.addItem(error)
            self.graph.plot(x,y, pen=color, name=name)

    def _reset_table(self):
        self.results_table.setColumnCount(self._sp_num_slices.value() + 1)
        self.results_table.setRowCount(0)
        self.results_table.setHorizontalHeaderLabels(['Folder name'] + [f'Section {i}' for i in range(1, self._sp_num_slices.value() + 1)])

    def _update_table(self, **kwargs):
        self.results_table.setColumnCount(self._sp_num_slices.value() + 1)
        self.results_table.setHorizontalHeaderLabels(['Folder name'] + [f'Section {i}' for i in range(1, self._sp_num_slices.value() + 1)])
        # self.results_table.setRowCount(kwargs['degree_of_radiality'].shape[0])
        # if 'degree_of_radiality' in kwargs:
        for row_val in kwargs['degree_of_radiality']:
            self.results_table.insertRow(self.results_table.rowCount())
            for col, col_val in enumerate(row_val):
                self.results_table.setItem(self.results_table.rowCount()-1, 0, QTableWidgetItem(kwargs['folder_name']))
                self.results_table.setItem(self.results_table.rowCount()-1, col + 1, QTableWidgetItem(str(col_val)))

    def selected_image_layer(self):
        return self._viewer.layers[self._image_layers.currentText()] # Need to force images to be grayscale in case other readers used!
        #[layer for layer in self._viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    def selected_label_layer(self):
        if self._label_layers.currentText() == 'No labels':
            return None
        else:
            return self._viewer.layers[self._label_layers.currentText()]

    def _reset_plot(self):
        if not hasattr(self, "graph"):
            self.graph = self._graphics_widget.addPlot()
        else:
            self.graph.clear()
