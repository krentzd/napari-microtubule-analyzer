import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
import math
from tqdm import tqdm

from .utils import (segment_and_separate_cells_with_centrosome,
                    segment_and_separate_cells, apply_cell_segmentation,
                    get_mean_abs_dot_product, get_area_mask, get_efd,
                    get_vector_field)

def compute_degree_of_radiality(im_stack, label_stack, num_of_slices, numerator_vec_name, denominator_vec_name, window_size):
    dor_list = []
    im_path_list = []
    stripwise_radial_im_list = []
    stripwise_tangential_im_list = []
    strip_mask_list = []
    vecs_list = []
    with_centrosome = False

    if 'file_paths' in im_stack.metadata.keys():
        im_paths = im_stack.metadata['file_paths']
    else:
        im_paths = ['' for i in range(im_stack.data.shape[0])]

    if label_stack is None:
        if im_stack.data.shape[1] == 2:
            with_centrosome = True
        im_iterator = tqdm(im_stack.data)
    else:
        im_iterator = zip(tqdm(im_stack.data, label_stack))

    for img_index, im_object in enumerate(im_iterator):
        cell_counter = 0
        if label_stack is None:
            im = im_object
            if with_centrosome:
                cell_crops, cell_centro, img_labels = segment_and_separate_cells_with_centrosome(im)
            else:
                cell_crops, img_labels = segment_and_separate_cells(im)
        else:
            im, label = im_object
            cell_crops, img_labels = apply_cell_segmentation(im, label)

        strip_mask = np.zeros_like(im)
        stripwise_radial_im_sum = np.zeros_like(im)
        stripwise_tangential_im_sum = np.zeros_like(im)

        if with_centrosome:
            vecs_per_img = np.zeros((im.shape[1], im.shape[2], 3), dtype='float64')
        else:
            vecs_per_img = np.zeros((im.shape[0], im.shape[1], 3), dtype='float64')

        if with_centrosome:
            img_iterator = zip(cell_crops, cell_centro)
        else:
            img_iterator = cell_crops

        for img_instance in img_iterator:
            if with_centrosome:
                img, img_centrosome = img_instance
            else:
                img = img_instance
            cell_counter += 1
            thresh = filters.threshold_otsu(img)
            img_thresh = img > thresh
            img_thresh = morphology.area_opening(img_thresh, area_threshold=150)
            chull = morphology.convex_hull_image(img_thresh)
            strip_mask_temp = np.zeros_like(img)

            try:
                contours = measure.find_contours(chull, 0.8)[0]
                contours[:,0].max()
                x_min = math.floor(contours[:,0].min())
                x_max = math.ceil(contours[:,0].max())
                y_min = math.floor(contours[:,1].min())
                y_max = math.ceil(contours[:,1].max())

                if with_centrosome:
                    centre_of_mass = (np.median(np.where(img_centrosome == np.max(img_centrosome))[1]), np.median(np.where(img_centrosome == np.max(img_centrosome))[0]))
                else:
                    centre_of_mass = ndimage.center_of_mass(chull)
                    centre_of_mass = (centre_of_mass[1], centre_of_mass[0])

                efd_recons = get_efd(img,
                                     chull,
                                     centre_of_mass,
                                     num_contours=num_of_slices,
                                     num_points=5000)
                efd_recons.reverse()
                vecs = get_vector_field(img, chull, (x_min, x_max), (y_min, y_max), window_size)
                # Need to specify x and y position, as well as slice position with 3rd dimension
                vecs_per_img[x_min:x_max, y_min:y_max, 1] += vecs[1]
                vecs_per_img[x_min:x_max, y_min:y_max, 2] += vecs[0]
                vecs_per_img[x_min:x_max, y_min:y_max, 0] += np.zeros((x_max - x_min, y_max - y_min))

                stripwise_dor = []
                stripwise_radial_im = np.zeros_like(img)
                stripwise_tangential_im = np.zeros_like(img)
                img_strip_sum = np.zeros_like(img)

                radial_im_full = np.zeros_like(img)
                tangential_im_full = np.zeros_like(img)

                for i in range(1,len(efd_recons)):
                    efd_recon_inner = efd_recons[i-1]
                    efd_recon_outer = efd_recons[i]
                    img_mask = get_area_mask(img.shape,
                                             efd_recon_outer,
                                             efd_recon_inner,
                                             centre_of_mass).astype('bool')

                    img_strip = img * img_mask

                    Z, radial_im, tangential_im = get_mean_abs_dot_product(centre_of_mass[0],
                                                                           centre_of_mass[1],
                                                                           img[x_min:x_max, y_min:y_max],
                                                                           vecs=vecs,
                                                                           chull=img_mask,
                                                                           x_range=(x_min, x_max),
                                                                           y_range=(y_min, y_max),
                                                                           numerator_vec_name=numerator_vec_name,
                                                                           denominator_vec_name=denominator_vec_name)
                    radial_im[np.isnan(radial_im)] = 0
                    tangential_im[np.isnan(tangential_im)] = 0

                    radial_im_full[x_min:x_max, y_min:y_max] = radial_im * 255
                    tangential_im_full[x_min:x_max, y_min:y_max] = tangential_im * 255

                    stripwise_dor.append(Z)
                    stripwise_radial_im += radial_im_full
                    stripwise_tangential_im += tangential_im_full

                    strip_mask_temp += img_mask.astype('uint8') * i



                dor_list.append(stripwise_dor)
                im_path_list.append(im_paths[img_index])

            except IndexError:
                print('No segmentation contours found for image. Ensure correct image segmentation')

            strip_mask += strip_mask_temp
            stripwise_radial_im_sum += stripwise_radial_im
            stripwise_tangential_im_sum += stripwise_tangential_im

        stripwise_radial_im_list.append(stripwise_radial_im_sum)
        stripwise_tangential_im_list.append(stripwise_tangential_im_sum)
        strip_mask_list.append(strip_mask)

    return np.stack(dor_list), (np.squeeze(np.stack(stripwise_radial_im_list)), np.squeeze(np.stack(stripwise_tangential_im_list))), np.stack(strip_mask_list).astype('uint8'), im_path_list
