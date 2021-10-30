# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:51:05 2020

@author: Kimberley Timmins


Evaluation for segmentation at ADAM challenge at MICCAI 2020
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import scipy.spatial

import evaluation_detection as eval_det
from utils.project_utils import read_dict_csv


def main(output_dir, annotation_dir, location_dir, metadata_file, threshold):

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        raise Exception('output folder not found at %s' % output_dir_path.absolute())
    annotation_dir_path = Path(annotation_dir)
    if not annotation_dir_path.exists():
        raise Exception('annotation folder not found at %s' % annotation_dir_path.absolute())
    location_dir_path = Path(location_dir)
    if not location_dir_path.exists():
        detection = False
        # raise Exception('location folder not found at %s' % location_dir_path.absolute())
    else:
        detection = True
    dataset_metadata_path = Path(metadata_file)
    if not dataset_metadata_path.exists():
        raise Exception('dataset metadata not found at %s' % dataset_metadata_path.absolute())

    metadata = read_dict_csv(dataset_metadata_path)
    metadata = [exam for exam in metadata if exam['subset'] == 'eval']

    dsc_list = []
    #h95_list = []
    vs_list = []

    sensitivity_list = []
    fps_list = []

    for exam_i, exam in enumerate(metadata, 1):
        filename = exam['aneurysm_seg_file'].split('/')[-1]
        filename_location = filename.split('.')[0] + '.txt'

        print(f"exam {exam_i}: {filename}")
        output_path = output_dir_path / filename
        if not output_path.exists():
            raise Exception('output file not found at %s' % output_path.absolute())
        annotation_path = annotation_dir_path / filename
        if not annotation_path.exists():
            raise Exception('annotation file not found at %s' % annotation_path.absolute())

        annotation_image, output_image = get_images(annotation_path.absolute().as_posix(),
                                                    output_path.absolute().as_posix(), threshold)

        dsc = get_dsc(annotation_image, output_image)
        #h95 = get_hausdorff(annotation_image, output_image)
        vs = get_vs(annotation_image, output_image)

        dsc_list.append(dsc)
        #h95_list.append(h95)
        vs_list.append(vs)

        if detection:
            annotation_locations = eval_det.get_locations(location_dir_path / filename_location)
            output_locations = get_center_of_mass(output_image)

            sensitivity, fps = eval_det.get_detection_metrics(annotation_locations, output_locations, annotation_image)

            sensitivity_list.append(sensitivity)
            fps_list.append(fps)

            print('%d exam %s : DSC=%.3f VS=%.3f sensitivity=%.3f FPs/exam=%d'
                  % (exam_i, filename.split('.')[0], dsc, vs, sensitivity, fps))
            #print('%d exam %s : DSC=%.3f HD=%.3f VS=%.3f sensitivity=%.3f FPs/exam=%d'
            #      % (exam_i, filename.split('.')[0], dsc, h95, vs, sensitivity, fps))
        else:
            print('%d exam %s : DSC=%.3f VS=%.3f' % (exam_i, filename.split('.')[0], dsc, vs))
            # print('%d exam %s : DSC=%.3f HD=%.3f VS=%.3f' % (exam_i, filename.split('.')[0], dsc, h95,  vs))

    avg_dsc = sum(dsc_list) / len(dsc_list)
    #avg_h95 = sum(h95_list) / len(h95_list)
    avg_vs = sum(vs_list) / len(vs_list)
    avg_sensitivity = sum(sensitivity_list) / len(sensitivity_list)
    avg_fps = sum(fps_list) / len(fps_list)

    print('---OVERALL STATISTICS---')
    print('Dice: %.3f (higher is better, min=0, max=1)' % avg_dsc)
    #print('HD: %.3f mm (lower is better, min=0, max=+inf)' % avg_h95)
    print('VS: %.3f (higher is better, min=0, max=1)' % avg_vs)

    if detection:
        print('Sensitivity: %.3f (higher is better, min=0, max=1)' % avg_sensitivity)
        print('False Positive Count: %.1f (lower is better, min=0, max=+inf)' % avg_fps)


def get_images(test_filename, result_filename, threshold=0.5):
    """Return the test and result images, thresholded and treated aneurysms removed."""
    test_image = sitk.ReadImage(test_filename, imageIO='NiftiImageIO')
    result_image = sitk.ReadImage(result_filename, imageIO='NiftiImageIO')
    
    assert test_image.GetSize() == result_image.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)
    
    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treated_image = test_image != 2  # treated aneurysms == 2
    masked_result_image = sitk.Mask(result_image, treated_image)
    masked_test_image = sitk.Mask(test_image, treated_image)
    
    # Return two binary masks
    return masked_test_image > threshold, masked_result_image > threshold


def get_dsc(test_image, result_image):
    """Compute the Dice Similarity Coefficient."""
    test_array = sitk.GetArrayFromImage(test_image).flatten()
    result_array = sitk.GetArrayFromImage(result_image).flatten()
    
    test_sum = np.sum(test_array)
    result_sum = np.sum(result_array)
    
    if test_sum == 0 and result_sum == 0:
        # Perfect result in case of no aneurysm
        return np.nan
    elif test_sum == 0 and not result_sum == 0:
        # Some segmentations, while there is no aneurysm
        return 0
    else:
        # There is an aneurysm, return similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(test_array, result_array)


def get_hausdorff(test_image, result_image):
    """Compute the Hausdorff distance."""

    result_statistics = sitk.StatisticsImageFilter()
    result_statistics.Execute(result_image)
  
    if result_statistics.GetSum() == 0:
        hd = np.nan
        return hd

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
    e_test_image = sitk.BinaryErode(test_image, (1, 1, 1))
    e_result_image = sitk.BinaryErode(result_image, (1, 1, 1))

    h_test_image = sitk.Subtract(test_image, e_test_image)
    h_result_image = sitk.Subtract(result_image, e_result_image)

    h_test_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_test_image))).tolist()
    h_result_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_result_image))).tolist()

    test_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_test_indices]
    result_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_result_indices]
    
    def get_distances_from_a_to_b(a, b):
        kd_tree = scipy.spatial.KDTree(a, leafsize=100)
        return kd_tree.query(b, k=1, eps=0, p=2)[0]

    d_test_to_result = get_distances_from_a_to_b(test_coordinates, result_coordinates)
    d_result_to_test = get_distances_from_a_to_b(result_coordinates, test_coordinates)

    hd = max(np.percentile(d_test_to_result, 95), np.percentile(d_result_to_test, 95))
    
    return hd


def get_vs(test_image, result_image):
    """Volumetric Similarity.
    
    VS = 1 - abs(A-B)/(A+B)
    
    A = ground truth
    B = predicted     
    """
    
    test_statistics = sitk.StatisticsImageFilter()
    result_statistics = sitk.StatisticsImageFilter()
    
    test_statistics.Execute(test_image)
    result_statistics.Execute(result_image)
    
    numerator = abs(test_statistics.GetSum() - result_statistics.GetSum())
    denominator = test_statistics.GetSum() + result_statistics.GetSum()
    
    if denominator > 0:
        vs = 1 - float(numerator) / denominator
    else:
        vs = np.nan
            
    return vs


def get_center_of_mass(result_image):
    """Based on result segmentation, find coordinate of centre of mass of predicted aneurysms."""
    result_array = sitk.GetArrayFromImage(result_image)
    if np.sum(result_array) == 0:
        # no detections
        return np.ndarray((0, 3))

    structure = ndimage.generate_binary_structure(rank=result_array.ndim, connectivity=result_array.ndim)
   
    label_array = ndimage.label(result_array, structure)[0]
    index = np.unique(label_array)[1:]

    # Get locations in x, y, z order.
    locations = np.fliplr(ndimage.measurements.center_of_mass(result_array, label_array, index))
    return locations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAM Aneurysm Segmentation Challenge Evaluation")

    parser.add_argument('-o', '--output_dir', default='medical_data/output',
                        help='Directory with segmentation masks produced by an algorithm')
    parser.add_argument('-a', '--annotation_dir', default='medical_data/ane_seg',
                        help='Directory with annotated segmentation masks')
    parser.add_argument('-l', '--location_dir', default='medical_data/location',
                        help='Directory with the location of the aneurysms')
    parser.add_argument('-m', '--metadata_file', default='medical_data/aneurysm_seg.csv',
                        help='File containing dataset metadata')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Threshold to convert the probabilities of a segmentation mask into classes')

    args = parser.parse_args()

    main(args.output_dir, args.annotation_dir, args.location_dir, args.metadata_file, args.threshold)
