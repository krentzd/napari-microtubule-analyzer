import napari
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker
import numpy as np
import os
import csv
import pyqtgraph as pg
from qtpy.QtWidgets import (QWidget, QTabWidget, QHBoxLayout, QVBoxLayout, QLabel,
                            QSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
                            QFileDialog, QComboBox, QAbstractItemView, QCheckBox,
                            QHeaderView, QRadioButton)
from qtpy.QtGui import QPixmap, QFont
from qtpy.QtCore import Qt, QSize
from .degree_of_radiality import compute_degree_of_radiality
from .skewness import compute_skewness
from napari.qt.threading import thread_worker
from napari.settings import get_settings
from .utils import abspath

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class RadialityPlotter(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        graph_container = QWidget()

        results_tab_widget = QTabWidget()


        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground(None)

        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self._graphics_widget)
        self.graph = self._graphics_widget.addPlot()
        self.setLayout(QVBoxLayout())
        # self.layout().addWidget(graph_container)

        self.results_table = QTableWidget(self)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.results_table.cellClicked.connect(self._switch_view_to_selected_row)

        # self.layout().addWidget(self.results_table)

        results_tab_widget.addTab(graph_container, "Graph")
        results_tab_widget.addTab(self.results_table, "Table")
        self.layout().addWidget(results_tab_widget)

        self.vector_field_choice_container = QWidget()
        self.vector_field_choice_container.setLayout(QVBoxLayout())

        header_container = QWidget()
        header_container.setLayout(QHBoxLayout())
        for header_name in ['Pattern', 'Numerator', 'Denominator']:
            header_label = QLabel()
            header_label.setText(header_name)
            header_label.setFont(QFont('Arial', 10, weight=QFont.Bold))
            header_container.layout().addWidget(header_label)

        self.btn_container = QWidget()
        self.btn_container.setLayout(QHBoxLayout())
        self.button_name_list = ['Radial', 'Horizontal', 'Vertical', 'Tangential', 'Random']

        self.icon_img_placeholder = QWidget()
        self.icon_img_placeholder.setLayout(QVBoxLayout())
        self.icon_size_inner = QSize(40, 40)
        self.icon_size_outer = QSize(45, 45)
        self._update_vec_field_icons()

        self._viewer.events.theme.connect(self._update_vec_field_icons)

        numerator_buttons = []
        for button_name in self.button_name_list:
            numerator_buttons.append(QRadioButton(button_name))
        numerator_btn_container = QWidget()
        numerator_btn_container.setLayout(QVBoxLayout())
        for button in numerator_buttons:
            button.toggled.connect(lambda __, button=button: self._numerator_button_state_setter(button))
            numerator_btn_container.layout().addWidget(button)

        denominator_buttons = []
        for button_name in self.button_name_list:
            denominator_buttons.append(QRadioButton(button_name))
        denominator_btn_container = QWidget()
        denominator_btn_container.setLayout(QVBoxLayout())
        for button in denominator_buttons:
            button.toggled.connect(lambda __, button=button: self._denominator_button_state_setter(button))
            denominator_btn_container.layout().addWidget(button)

        self.btn_container.layout().addWidget(self.icon_img_placeholder)
        self.btn_container.layout().addWidget(numerator_btn_container)
        self.btn_container.layout().addWidget(denominator_btn_container)

        self.vector_field_choice_container.layout().addWidget(header_container)
        self.vector_field_choice_container.layout().addWidget(self.btn_container)

        self.layout().addWidget(self.vector_field_choice_container)
        self.vector_field_choice_container.hide()

        vec_field_checkbox = QCheckBox('Choose vector field patterns', self)
        vec_field_checkbox.setChecked(False)
        vec_field_checkbox.stateChanged.connect(self._on_vec_field_checkbox_changed)
        self.layout().addWidget(vec_field_checkbox)

        num_slices_container = QWidget()
        num_slices_container.setLayout(QHBoxLayout())
        label = QLabel('Number of cell subsections')
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
        image_input_label = QLabel('Cell image stack')
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
        self._viewer.layers.events.removed.connect(self._update_combo_boxes)

        metric_container = QWidget()
        metric_container.setLayout(QHBoxLayout())
        metric_label = QLabel('Metric')
        metric_container.layout().addWidget(metric_label)
        self._metric = QComboBox(self)
        for metric_item in ['Degree of Radiality', 'Skewness']:
            self._metric.addItem(metric_item)
        metric_container.layout().addWidget(self._metric)
        metric_container.layout().setSpacing(0)
        self.layout().addWidget(metric_container)

        window_size_container = QWidget()
        window_size_container.setLayout(QHBoxLayout())
        window_size_label = QLabel('Window size')
        window_size_container.layout().addWidget(window_size_label)
        self._window_size = QSpinBox(self)
        self._window_size.setMinimum(1)
        self._window_size.setMaximum(100)
        self._window_size.setValue(1)
        self._window_size.setSingleStep(1)
        window_size_container.layout().addWidget(self._window_size)
        window_size_container.layout().setSpacing(0)
        self.layout().addWidget(window_size_container)

        btn_compute = QPushButton("Compute")
        btn_compute.clicked.connect(self._compute)
        self.layout().addWidget(btn_compute)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self._save_table)
        self.layout().addWidget(save_button)

        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self._reset_widgets)
        self.layout().addWidget(clear_button)

        self._max_slice_counter = 0

        self._update_table()
        self. _update_combo_boxes()
        self._analysis_counter = 0

        colormap = pg.colormap.getFromMatplotlib('tab10')
        self.colors_list = colormap.getColors()

        numerator_buttons[0].setChecked(True)
        denominator_buttons[3].setChecked(True)

        self._numerator_button_state = 'Radial'
        self._denominator_button_state = 'Tangential'

    def _update_vec_field_icons(self, *args):
        if len(args) > 0:
            self.icon_img_placeholder.layout().removeWidget(self.icon_img_container)
            self.icon_img_container.close()

        self.icon_img_container = QWidget()
        self.icon_img_container.setLayout(QVBoxLayout())

        theme_name = get_settings().appearance.theme
        if theme_name == 'system':
            theme_name = 'light'
        for button_name in self.button_name_list:
            im_path = abspath(__file__, f'icon_images/{button_name}_{theme_name}.png')
            icon_img = QPixmap(im_path)
            icon_img = icon_img.scaled(self.icon_size_inner, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            icon_label = QLabel()
            icon_label.setPixmap(icon_img)
            icon_label.setFixedSize(self.icon_size_outer.width(), self.icon_size_outer.height())
            self.icon_img_container.layout().addWidget(icon_label)

        self.icon_img_placeholder.layout().addWidget(self.icon_img_container)

        if len(args) > 0:
            self.icon_img_placeholder.update()

    def _on_vec_field_checkbox_changed(self, value):
        state = Qt.CheckState(value)
        if state == Qt.CheckState.Checked:
            self.vector_field_choice_container.show()
        elif state == Qt.CheckState.Unchecked:
            self.vector_field_choice_container.hide()

    def _numerator_button_state_setter(self, btn):
        self._numerator_button_state = btn.text()

    def _denominator_button_state_setter(self, btn):
        self._denominator_button_state = btn.text()

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
        self._max_slice_counter = 0
        self._reset_plot()
        self._update_table(reset_table=True)

    def _update_max_slice_counter(self, val):
        if val > self._max_slice_counter:
            self._max_slice_counter = val
        else:
            pass

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
        self.plot_metrics(images=selected_image,
                          labels=selected_label.data if selected_label else selected_label,
                          color=self.colors_list[self._analysis_counter], #loop over to first index when max colours reached (use modulo)
                          name=selected_image.name)

        self._analysis_counter += 1

    def plot_metrics(self, images, labels, color, name='line1', show_mean=True):
        num_of_slices = self._sp_num_slices.value()
        self._update_max_slice_counter(num_of_slices)
        window_size = self._window_size.value()

        #Check if file paths included in stack
        if 'file_paths' not in images.metadata.keys():
            images.metadata['file_paths'] = [f'Image {i}' for i in range(images.data.shape[0])]

        if self._metric.currentText() == 'Degree of Radiality':
            num_vec_name = self._numerator_button_state
            den_vec_name = self._denominator_button_state

            results, dor_images, cell_labels, im_paths = compute_degree_of_radiality(images,
                                                                                     labels,
                                                                                     num_of_slices,
                                                                                     num_vec_name,
                                                                                     den_vec_name,
                                                                                     window_size)

            self._viewer.add_image(dor_images[0], name=f'Numerator Directionality {name}', colormap='magma')
            self._viewer.add_image(dor_images[1], name=f'Denominator Directionality {name}', colormap='magma')

        elif self._metric.currentText() == 'Skewness':
            results, cell_labels, im_paths = compute_skewness(images,
                                                              labels,
                                                              num_of_slices)

        if labels == None:
            self._viewer.add_labels(cell_labels, name=f'Cell Segmentation {name}')

        self._update_table(results=results, folder_name=name, im_paths=im_paths, metric=self._metric.currentText())

        self.graph.addLegend()
        if show_mean:
            mean_dor = np.mean(results, axis=0)
            std_dor = np.std(results, axis=0)

            x = np.array([i for i in range(1, num_of_slices + 1)])
            y = mean_dor
            error = pg.ErrorBarItem(x=x, y=y, height=std_dor, beam=0.1, pen=color)
            self.graph.addItem(error)
            self.graph.plot(x,y, pen=color, name=name)

    def _update_table(self, **kwargs):
        self.results_table.setColumnCount(self._max_slice_counter + 3)
        if 'reset_table' in kwargs:
            if kwargs['reset_table'] == True:
                self.results_table.setRowCount(0)

        if 'results' in kwargs:
            for row, row_val in enumerate(kwargs['results']):
                self.results_table.insertRow(self.results_table.rowCount())
                self.results_table.setItem(self.results_table.rowCount()-1, 0, QTableWidgetItem(kwargs['folder_name']))
                self.results_table.setItem(self.results_table.rowCount()-1, 1, QTableWidgetItem(kwargs['im_paths'][row]))
                self.results_table.setItem(self.results_table.rowCount()-1, 2, QTableWidgetItem(kwargs['metric']))
                for col, col_val in enumerate(row_val):
                    self.results_table.setItem(self.results_table.rowCount()-1, col + 3, QTableWidgetItem(str(col_val)))

        for col in range(1, self._max_slice_counter + 1):
            self.results_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
            self.results_table.horizontalHeader().setSectionsClickable(True)

        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setHorizontalHeaderLabels(['Folder name', 'Image name', 'Metric'] + [f'Section {i}' for i in range(1, self._max_slice_counter + 1)])

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

    # If still in viewer, switch viewer to specified image
    def _switch_view_to_selected_row(self, row, column):
        # Verify chosen image still in layers
        selected_layer = self.results_table.item(row, 0).text()
        available_layers = [self._image_layers.itemText(i) for i in range(self._image_layers.count())]
        if selected_layer in available_layers:
            # Find index of specified image
            selected_image_path = self.results_table.item(row, 1).text()
            dim_position = self._viewer.layers[selected_layer].metadata['file_paths'].index(selected_image_path)
            self._viewer.dims.current_step = (dim_position,) + self._viewer.dims.current_step[1:]
