import napari
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, feature, exposure
from skimage.morphology import convex_hull_image, binary_dilation
from skimage import morphology
from skimage.segmentation import flood_fill, flood
from skimage.draw import disk
import math
from tqdm import tqdm
import tifffile
import glob
from scipy import stats
import os
from pyefd import elliptic_fourier_descriptors, plot_efd,reconstruct_contour, calculate_dc_coefficients
import matplotlib.colors as mcolors
import random
import matplotlib
import matplotlib.gridspec as gridspec
from pathlib import Path

from .make_coherence import *
from .make_image_gradients import *
from .make_orientation import *
from .make_structure_tensor_2d import *
from .make_vxvy import *

# From: https://github.com/volume-em/empanada-napari/blob/main/empanada_napari/utils.py
def abspath(root, relpath):
    root = Path(root)
    if root.is_dir():
        path = root/relpath
    else:
        path = root.parent/relpath
    return str(path.absolute())

def tuple_mean(t1, t2, w=(0.5, 0.5)):
    t_a = w[0] * t1[0] + w[1] * t2[0]
    t_b = w[0] * t1[1] + w[1] * t2[1]
    return (t_a, t_b)

def efd_mean(efd_coeffs, efd_locs, w=(0.5, 0.5)):
    efd_coeffs_mean = w[0] * efd_coeffs[0] + w[1] * efd_coeffs[1]
    efd_locs_mean = tuple_mean(efd_locs[0], efd_locs[1], w=w)
    return efd_coeffs_mean, efd_locs_mean

def apply_efd_transform(in_img, num_contours=100, num_points=100):
    img = in_img[0]
    thresh = filters.threshold_otsu(img)
    img_thresh = img > thresh
    chull = convex_hull_image(img_thresh)
    contours = measure.find_contours(chull, 0.8)
    centroid = (np.median(np.where(in_img[1] == np.max(in_img[1]))[1]), np.median(np.where(in_img[1] == np.max(in_img[1]))[0]))

    efd_coeffs = elliptic_fourier_descriptors(np.squeeze(contours), order=10)
    efd_locus = calculate_dc_coefficients(np.squeeze(contours))

    rr, cc = disk(centroid, 5)
    circ_img = np.zeros(img.shape)
    circ_img[cc, rr] = 1
    centroid_contours = measure.find_contours(circ_img, 0.8)

    efd_coeffs_cen = elliptic_fourier_descriptors(np.squeeze(centroid_contours), order=10)
    efd_locus_cen = calculate_dc_coefficients(np.squeeze(centroid_contours))

    efd_recons = []
    for w in [k/num_contours for k in range(0,num_contours)]:
        efd_coeffs_temp, efd_loc_temp = efd_mean((efd_coeffs, efd_coeffs_cen), (efd_locus, efd_locus_cen), w=(1 - w, w))
        efd_recon = reconstruct_contour(efd_coeffs_temp, locus=efd_loc_temp, num_points=num_points)
        efd_recons.append(efd_recon)

    trans_img = np.zeros((num_points, num_contours))

    for angle_coord in range(num_points):
        for r_coord, efd_recon in enumerate(reversed(efd_recons)):
            trans_img[angle_coord, r_coord] = img[int(efd_recon[angle_coord,1]), int(efd_recon[angle_coord,0])]

    return trans_img

def get_efd(img, chull, centroid, num_contours=100, num_points=100):

    contours = measure.find_contours(chull, 0.8)

    efd_coeffs = elliptic_fourier_descriptors(np.squeeze(contours), order=10)
    efd_locus = calculate_dc_coefficients(np.squeeze(contours))

    rr, cc = disk(centroid, 1)
    circ_img = np.zeros(img.shape)
    circ_img[cc, rr] = 1
    centroid_contours = measure.find_contours(circ_img, 0.8)

    efd_coeffs_cen = elliptic_fourier_descriptors(np.squeeze(centroid_contours), order=10)
    efd_locus_cen = calculate_dc_coefficients(np.squeeze(centroid_contours))
    efd_recon_cen = reconstruct_contour(efd_coeffs_cen, locus=efd_locus_cen, num_points=num_points)

    efd_recons = []
    for w in [k/num_contours for k in range(0,num_contours)]:
        efd_coeffs_temp, efd_loc_temp = efd_mean((efd_coeffs, efd_coeffs_cen), (efd_locus, efd_locus_cen), w=(1 - w, w))
        efd_recon = reconstruct_contour(efd_coeffs_temp, locus=efd_loc_temp, num_points=num_points)
        efd_recons.append(efd_recon)
    efd_recons.append(efd_recon_cen)
    return efd_recons

def get_area_mask(img_shape, efd_recon_outer, efd_recon_inner, centroid):
    outer_ring = np.zeros(img_shape)
    outer_ring[efd_recon_outer[:,0].astype(int), efd_recon_outer[:,1].astype(int)] = 1
    outer_mask = flood_fill(outer_ring, (int(centroid[1]), int(centroid[0])), new_value=1, connectivity=0)

    inner_ring = np.zeros(img_shape)
    inner_ring[efd_recon_inner[:,0].astype(int), efd_recon_inner[:,1].astype(int)] = 1
    inner_mask = flood_fill(inner_ring, (int(centroid[1]), int(centroid[0])), new_value=1, connectivity=0)

    return outer_mask - inner_mask

def vector_lengths(u,v):
    return np.array([[np.sqrt(u_ij**2 + v_ij**2) for u_ij, v_ij in zip(u_i,v_i)] for u_i, v_i in zip(u,v)])

def make_radial_vectors(centroid, chull, x_range, y_range):
    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                               np.arange(y_range[0], y_range[1]),
                               indexing = 'ij')

    u = xmesh - centroid[1]
    v = ymesh - centroid[0]

    l = vector_lengths(u,v)

    vx, vy = u / l, v / l

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def make_horizontal_vectors(centroid, chull, x_range, y_range):
    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                               np.arange(y_range[0], y_range[1]),
                               indexing = 'ij')

    u = xmesh * 0
    v = ymesh / ymesh

    l = vector_lengths(u,v)

    vx, vy = u / l, v / l

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def make_vertical_vectors(centroid, chull, x_range, y_range):
    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                               np.arange(y_range[0], y_range[1]),
                               indexing = 'ij')

    u = xmesh / xmesh
    v = ymesh * 0

    l = vector_lengths(u,v)

    vx, vy = u / l, v / l

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def make_random_vectors(centroid, chull, x_range, y_range):
    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                               np.arange(y_range[0], y_range[1]),
                               indexing = 'ij')

    u = random.uniform(-1,1) * xmesh / xmesh
    v = random.uniform(-1,1) * ymesh / ymesh

    l = vector_lengths(u,v)

    vx, vy = u / l, v / l

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def make_tangential_vectors(centroid, chull, x_range, y_range):
    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                               np.arange(y_range[0], y_range[1]),
                               indexing = 'ij')

    u = ymesh - centroid[0]
    v = -xmesh + centroid[1]

    l = vector_lengths(u,v)

    vx, vy = u / l, v / l

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def get_vector_field(img, chull, x_range, y_range, window_size):
    img = img[x_range[0]:x_range[1], y_range[0]:y_range[1]]

    image_filter_sigma = 0.1
    local_sigma = window_size

    threshold_value = max(int(0.5 * np.median(img)), 2)

    filtered_image = filters.gaussian(img, sigma = image_filter_sigma, mode = 'nearest', preserve_range = True)

    image_gradient_x, image_gradient_y = make_image_gradients(filtered_image)

    Structure_Tensor, EigenValues, EigenVectors, Jxx, Jxy, Jyy = make_structure_tensor_2d(image_gradient_x,
                                                                                          image_gradient_y,
                                                                                          local_sigma)

    vx, vy = make_vx_vy(filtered_image, EigenVectors, threshold_value)

    chull = chull[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    vx[~chull] = np.nan
    vy[~chull] = np.nan

    return vx, vy

def return_vec_field(name):
    if name == 'Radial':
        return make_radial_vectors
    elif name == 'Horizontal':
        return make_horizontal_vectors
    elif name == 'Vertical':
        return make_vertical_vectors
    elif name == 'Tangential':
        return make_tangential_vectors
    elif name == 'Random':
        return make_random_vectors

def get_mean_abs_dot_product(x, y, img, vecs, chull, x_range, y_range, numerator_vec_name, denominator_vec_name):
    vx_i, vy_i = vecs

    vx_i, vy_i = vx_i * (img / 255), vy_i * (img / 255)

    numerator_vec_field = return_vec_field(numerator_vec_name)
    denominator_vec_field = return_vec_field(denominator_vec_name)

    vx_n, vy_n = numerator_vec_field((x, y), chull, x_range, y_range)
    vx_d, vy_d = denominator_vec_field((x, y), chull, x_range, y_range)

    dp_map_n = np.abs(np.multiply(vx_i, vx_n) + np.multiply(vy_i, vy_n))
    dp_map_d = np.abs(np.multiply(vx_i, vx_d) + np.multiply(vy_i, vy_d))

    dp_map = (dp_map_n[~np.isnan(dp_map_n)].sum() / dp_map_d[~np.isnan(dp_map_d)].sum())

    return dp_map, np.asarray(dp_map_n), np.asarray(dp_map_d)

def is_touching_img_border(mask):
    chull = morphology.convex_hull_image(mask)
    return np.any(chull[:,-1]) or np.any(chull[:,0]) or np.any(chull[0]) or np.any(chull[-1])

def segment_and_separate_cells(img):
    # Otsu threshold and are threshold
    img_8bit = exposure.rescale_intensity(img, out_range=np.uint8)
    thresh = filters.threshold_otsu(img_8bit)
    img_thresh = img_8bit > thresh
    # img_thresh = morphology.binary_dilation(img, morphology.disk(2))

    min_cell_size = (img.shape[0] / 10) ** 2

    img_thresh = morphology.area_opening(img_thresh, area_threshold=min_cell_size)

    # Identify individual cells with connected components
    img_labels = morphology.label(img_thresh)

    valid_cell_crops = []
    place_holder_array = np.zeros_like(img_labels)
    for l in range(1, img_labels.max() + 1):
        # Discard cells that are touching image borders
        if not is_touching_img_border(img_labels == l):
            chull = morphology.convex_hull_image(img_labels == l)
            place_holder_array = place_holder_array + chull * l
            valid_cell_crops.append(chull * img)

    return valid_cell_crops, place_holder_array

def segment_and_separate_cells_with_centrosome(img_in):
    # Otsu threshold and are threshold
    img = img_in[0]
    img_8bit = exposure.rescale_intensity(img, out_range=np.uint8)
    thresh = filters.threshold_otsu(img_8bit)
    img_thresh = img_8bit > thresh
    # img_thresh = morphology.binary_dilation(img, morphology.disk(2))

    min_cell_size = (img.shape[0] / 10) ** 2

    img_thresh = morphology.area_opening(img_thresh, area_threshold=min_cell_size)

    # Identify individual cells with connected components
    img_labels = morphology.label(img_thresh)

    valid_cell_crops = []
    valid_cell_crops_centrosome = []
    place_holder_array = np.zeros_like(img_labels)
    for l in range(1, img_labels.max() + 1):
        # Discard cells that are touching image borders
        if not is_touching_img_border(img_labels == l):
            chull = morphology.convex_hull_image(img_labels == l)
            place_holder_array = place_holder_array + chull * l
            valid_cell_crops.append(chull * img)
            valid_cell_crops_centrosome.append(chull * img_in[1])

    return valid_cell_crops, valid_cell_crops_centrosome, place_holder_array

def segment_and_separate_cells_from_dir(im_stack):
    cell_crops = []
    chull_list = []
    for im in tqdm(im_stack):
        # im = np.asarray(im)
        valid_cell_crops, img_labels = segment_and_separate_cells(im)
        cell_crops = cell_crops + valid_cell_crops
        chull_list.append(img_labels)

    return cell_crops, chull_list

def apply_cell_segmentation(img, label):
    # Binarize to ensure that connected components can be identified
    img_thresh = label > 0
    # Identify individual cells with connected components
    img_labels = morphology.label(img_thresh)

    valid_cell_crops = []
    place_holder_array = np.zeros_like(img_labels)
    for l in range(1, img_labels.max() + 1):
        # Discard cells that are touching image borders
        if not is_touching_img_border(img_labels == l):
            chull = morphology.convex_hull_image(img_labels == l)
            place_holder_array = place_holder_array + chull * l
            valid_cell_crops.append(chull * img)

    return valid_cell_crops, place_holder_array

def apply_cell_segmentation_from_dir(im_stack, label_stack):
    cell_crops = []
    chull_list = []
    for im, label in zip(tqdm(im_stack, label_stack)):
        # im = np.asarray(im)
        valid_cell_crops, img_labels = apply_cell_segmentation(im, label)
        cell_crops = cell_crops + valid_cell_crops
        chull_list.append(img_labels)

    return cell_crops, chull_list

def get_skewness(img, label):
    properties = measure.regionprops(label.astype(int), img)

    centroid = properties[0].centroid
    center_of_mass = properties[0].weighted_centroid

    fitted_rad = math.sqrt(np.sum(label > 0) / math.pi)

    centroid_to_com_dist = math.sqrt((center_of_mass[0] - centroid[0])**2 + (center_of_mass[1] - centroid[1])**2)

    return centroid_to_com_dist / fitted_rad, centroid, center_of_mass
