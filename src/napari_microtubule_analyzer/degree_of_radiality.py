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

from .make_coherence import *
from .make_image_gradients import *
from .make_orientation import *
from .make_structure_tensor_2d import *
from .make_vxvy import *

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
#     centroid = (ndimage.center_of_mass(chull)[1], ndimage.center_of_mass(chull)[0])

#     centroid = (np.median(np.where(in_img[1] == np.max(in_img[1]))[1]), np.median(np.where(in_img[1] == np.max(in_img[1]))[0]))


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

def get_vector_field(img, chull, x_range, y_range):
    img = img[x_range[0]:x_range[1], y_range[0]:y_range[1]]

    image_filter_sigma = 0.1
    local_sigma = 1

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

def plot_vecs(im, vecs_1, vecs_2, vecs_3, x_range, y_range):

    print(x_range, y_range)

    fig, ax = plt.subplots(1, 4, figsize = (60, 15), sharex = True, sharey = True)
    spacing = 15 # Spacing between plotting the orientation vectors
    scale = 60

    xmesh, ymesh = np.meshgrid(np.arange(x_range[0], x_range[1]),
                                         np.arange(y_range[0], y_range[1]),
                                        indexing = 'ij')


    vx, vy = vecs_1

    ax[0].quiver(ymesh[spacing//2::spacing, spacing//2::spacing],
                 xmesh[spacing//2::spacing, spacing//2::spacing],
                 vy[spacing//2::spacing, spacing//2::spacing],
                 vx[spacing//2::spacing, spacing//2::spacing],
                 scale = scale, headlength = 0, headaxislength = 0,
                 pivot = 'middle', color = 'k', angles = 'xy')

    ax[0].set_title('Image', pad = 20, fontsize = 40)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    vx, vy = vecs_2

    ax[1].quiver(ymesh[spacing//2::spacing, spacing//2::spacing],
                 xmesh[spacing//2::spacing, spacing//2::spacing],
                 vy[spacing//2::spacing, spacing//2::spacing],
                 vx[spacing//2::spacing, spacing//2::spacing],
                 scale = scale, headlength = 0, headaxislength = 0,
                 pivot = 'middle', color = 'k', angles = 'xy')

    ax[1].set_title('Radial', pad = 20, fontsize = 40)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    vx, vy = vecs_3

    ax[2].quiver(ymesh[spacing//2::spacing, spacing//2::spacing],
                 xmesh[spacing//2::spacing, spacing//2::spacing],
                 vy[spacing//2::spacing, spacing//2::spacing],
                 vx[spacing//2::spacing, spacing//2::spacing],
                 scale = scale, headlength = 0, headaxislength = 0,
                 pivot = 'middle', color = 'k', angles = 'xy')

    ax[2].set_title('Tangential', pad = 20, fontsize = 40)
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    im1 = ax[3].imshow(im, vmin = 0, vmax = 1, cmap = 'RdYlBu_r')

    ax[3].set_title('Dot product', pad = 20, fontsize = 40)
    ax[3].set_xticks([])
    ax[3].set_yticks([])


    plt.show()


def get_mean_abs_dot_product(x, y, img, vecs, chull, x_range, y_range):
    vx_i, vy_i = vecs

    vx_i, vy_i = vx_i * (img / 255), vy_i * (img / 255)

    vx_r, vy_r = make_radial_vectors((x, y), chull, x_range, y_range)
    vx_t, vy_t = make_tangential_vectors((x, y), chull, x_range, y_range)

    dp_map_r = np.abs(np.multiply(vx_i, vx_r) + np.multiply(vy_i, vy_r))
    dp_map_t = np.abs(np.multiply(vx_i, vx_t) + np.multiply(vy_i, vy_t))

    dp_map = (dp_map_r[~np.isnan(dp_map_r)].sum() / dp_map_t[~np.isnan(dp_map_t)].sum())

    return dp_map, np.asarray(dp_map_r)

def is_touching_img_border(mask):
    chull = morphology.convex_hull_image(mask)
    return np.any(chull[:,-1]) or np.any(chull[:,0]) or np.any(chull[0]) or np.any(chull[-1])

def segment_and_separate_cells(img):
    # Otsu threshold and are threshold
    thresh = filters.threshold_otsu(img)
    img_thresh = img > thresh
#     img_thresh = morphology.binary_opening(img, morphology.disk(1))

    min_cell_size = (img.shape[0] / 10) ** 2

    img_thresh = morphology.area_opening(img_thresh, area_threshold=min_cell_size)

    # Identify individual cells with connected components
    img_labels = morphology.label(img_thresh)

    valid_cell_crops = []
    for l in range(1, img_labels.max() + 1):
        # Discard cells that are touching image borders
        if not is_touching_img_border(img_labels == l):
            chull = morphology.convex_hull_image(img_labels == l)
            valid_cell_crops.append(chull * img)

    return valid_cell_crops

def segment_and_separate_cells_from_dir(im_stack):
    cell_crops = []
    for im in tqdm(im_stack):
        valid_cell_crops = segment_and_separate_cells(im)
        cell_crops = cell_crops + valid_cell_crops

    return cell_crops

def compute_degree_of_radiality(im_stack: Image, num_of_slices: int):
    dor_list = []
    img_crops = segment_and_separate_cells_from_dir(im_stack)

    stripwise_radial_im_list = []
    for img in tqdm(img_crops):
        thresh = filters.threshold_otsu(img)
        img_thresh = img > thresh
        img_thresh = morphology.area_opening(img_thresh, area_threshold=150)
        chull = convex_hull_image(img_thresh)
        contours = measure.find_contours(chull, 0.8)[0]
        contours[:,0].max()
        x_min = math.floor(contours[:,0].min())
        x_max = math.ceil(contours[:,0].max())
        y_min = math.floor(contours[:,1].min())
        y_max = math.ceil(contours[:,1].max())

        centre_of_mass = ndimage.center_of_mass(chull )
        centre_of_mass = (centre_of_mass[1], centre_of_mass[0])

        efd_recons = get_efd(img, chull, centre_of_mass, num_contours=num_of_slices, num_points=5000)
        efd_recons.reverse()
        vecs = get_vector_field(img, chull, (x_min, x_max), (y_min, y_max))

        stripwise_dor = []
        stripwise_radial_im = np.zeros_like(img)
        img_strip_sum = np.zeros_like(img)

        radial_im_full = np.zeros_like(img)
        for i in range(1,len(efd_recons)):
            efd_recon_inner = efd_recons[i-1]
            efd_recon_outer = efd_recons[i]
            img_mask = get_area_mask(img.shape, efd_recon_outer, efd_recon_inner, centre_of_mass).astype('bool')

            img_strip = img * img_mask
            img_strip_sum += img_strip

            Z, radial_im = get_mean_abs_dot_product(centre_of_mass[0], centre_of_mass[1], img[x_min:x_max, y_min:y_max], vecs=vecs, chull=img_mask, x_range=(x_min, x_max), y_range=(y_min, y_max))
            radial_im[np.isnan(radial_im)] = 0

            radial_im_full[x_min:x_max, y_min:y_max] = radial_im * 255

            stripwise_dor.append(Z)
            stripwise_radial_im += radial_im_full

        stripwise_radial_im_list.append(stripwise_radial_im)
        mse = np.mean(img_strip_sum - img * chull) ** 2

        if (mse < 5) and not (True in [np.isnan(i) for i in stripwise_dor]):
            dor_list.append(stripwise_dor)
        else:
            print('NA')

    # Return DoR as (im, slice, width, height)
    return np.stack(dor_list), np.squeeze(np.stack(stripwise_radial_im_list))
