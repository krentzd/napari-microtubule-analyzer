import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
import math
from tqdm import tqdm

from .utils import (segment_and_separate_cells_with_centrosome,
                    segment_and_separate_cells, apply_cell_segmentation,
                    get_efd, get_area_mask, get_skewness)

def compute_skewness(im_stack, label_stack, num_of_slices):
    skw_list = []
    im_path_list = []
    strip_mask_list = []
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

        if with_centrosome:
            img_iterator = zip(cell_crops, cell_centro)
        else:
            img_iterator = cell_crops

        for img in img_iterator:
            if with_centrosome:
                img, img_centrosome = img
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
                stripwise_skw = []

                for i in range(1,len(efd_recons)):
                    efd_recon_inner = efd_recons[i-1]
                    efd_recon_outer = efd_recons[i]
                    img_mask = get_area_mask(img.shape,
                                             efd_recon_outer,
                                             efd_recon_inner,
                                             centre_of_mass).astype('bool')

                    img_strip = img * img_mask

                    skw, centroid, com = get_skewness(img, img_mask)

                    stripwise_skw.append(skw)


                    strip_mask_temp += img_mask.astype('uint8') * i

                skw_list.append(stripwise_skw)
                im_path_list.append(im_paths[img_index])
                # else:
                    # print('NA')
            except IndexError:
                print('No segmentation contours found for image. Ensure correct image segmentation')

            strip_mask += strip_mask_temp

        strip_mask_list.append(strip_mask)

    return np.stack(skw_list), np.stack(strip_mask_list).astype('uint8'), im_path_list
