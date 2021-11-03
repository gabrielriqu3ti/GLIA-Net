# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:53:21 2020

@author: Kimberley Timmins

Evaluation of detection task at ADAM challenge MICCAI 2020
"""

import argparse
from pathlib import Path

import warnings

import numpy as np
import SimpleITK as sitk

from utils.project_utils import read_dict_csv


def main(output_dir, annotation_dir, location_dir, metadata_file, threshold):

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        raise Exception('output folder not found at %s' % output_dir_path.absolute())
    annotation_dir_path = Path(annotation_dir)
    if not output_dir_path.exists():
        raise Exception('annotation folder not found at %s' % annotation_dir_path.absolute())
    location_dir_path = Path(location_dir)
    if not location_dir_path.exists():
        raise Exception('location folder not found at %s' % location_dir_path.absolute())
    dataset_metadata_path = Path(metadata_file)
    if not dataset_metadata_path.exists():
        raise Exception('dataset metadata not found at %s' % dataset_metadata_path.absolute())

    metadata = read_dict_csv(dataset_metadata_path)
    metadata = [exam for exam in metadata if exam['subset'] == 'eval']

    sensitivity_list = []
    fps_list = []

    for exam_i, exam in enumerate(metadata, 1):
        filename = exam['aneurysm_seg_file'].split('/')[-1]
        filename_location = filename.split('.')[0] + '.txt'

        output_path = output_dir_path / filename
        if not output_path.exists():
            raise Exception('output file not found at %s' % output_path.absolute())
        annotation_path = annotation_dir_path / filename
        if not annotation_path.exists():
            raise Exception('annotation mask file not found at %s' % annotation_path.absolute())
        location_path = location_dir_path / filename_location
        if not location_path.exists():
            raise Exception('location file not found at %s' % location_path.absolute())

        annotation_locations = get_locations(location_path)
        output_locations = get_result(output_path)
        annotation_image = sitk.ReadImage(annotation_path)
    
        sensitivity, fps = get_detection_metrics(annotation_locations, output_locations, annotation_image)

        sensitivity_list.append(sensitivity)
        fps_list.append(fps)

        print('%d exam %s : sensitivity=%.3f FPs/exam=%d' % (exam_i, filename, sensitivity, fps))

    avg_sensitivity = sum(sensitivity_list) / len(sensitivity_list)
    avg_fps = sum(fps_list) / len(fps_list)

    print('---OVERALL STATISTICS---')
    print('Sensitivity: %.3f (higher is better, min=0, max=1)' % avg_sensitivity)
    print('False Positive Count: %d (lower is better, min=0, max=+inf)' % avg_fps)
    

def get_locations(test_filename):
    """Return the locations and radius of actual aneurysms as a NumPy array"""

    # Read comma-separated coordinates from a text file.
    with warnings.catch_warnings():
        # Suppress empty file warning from genfromtxt.
        warnings.filterwarnings("ignore", message=".*Empty input file.*")

        # atleast_2d() makes sure that test_locations is a 2D array, even if there is just a single location.
        # genfromtxt() raises a ValueError if the number of columns is inconsistent.
        test_locations = np.atleast_2d(np.genfromtxt(test_filename, delimiter=',', encoding='utf-8-sig'))

    # Reshape an empty result into a 0x4 array.
    if test_locations.size == 0:
        test_locations = test_locations.reshape(0, 4)

    # DEBUG: verify that the inner dimension size is 4.
    assert test_locations.shape[1] == 4
    
    return test_locations


def get_result(result_filename):
    """Read Result file and extract coordinates as a NumPy array"""

    # Read comma-separated coordinates from a text file.
    with warnings.catch_warnings():
        # Suppress empty file warning from genfromtxt.
        warnings.filterwarnings("ignore", message=".*Empty input file.*")

        # atleast_2d() makes sure that test_locations is a 2D array, even if there is just a single location.
        # genfromtxt() raises a ValueError if the number of columns is inconsistent.
        result_locations = np.atleast_2d(np.genfromtxt(result_filename, delimiter=',', encoding='utf-8-sig'))

    # Reshape an empty result into a 0x3 array.
    if result_locations.size == 0:
        result_locations = result_locations.reshape(0, 3)

    # DEBUG: verify that the inner dimension size is 3.
    assert result_locations.shape[1] == 3
        
    return result_locations


def get_treated_locations(test_image):
    """Return an array with a list of locations of treated aneurysms(based on aneurysms.nii.gz)"""
    treated_image = test_image > 1.5
    treated_array = sitk.GetArrayFromImage(treated_image)
    
    if np.sum(treated_array) == 0:
        # no treated aneurysms
        return np.array([])
    
    # flip so (x,y,z)
    treated_coords = np.flip(np.nonzero(treated_array))
    
    return np.array(list(zip(*treated_coords)))


def get_detection_metrics(test_locations, result_locations, test_image):
    """Calculate sensitivity and false positive count for each image.

    The distance between every result-location and test-locations must be less
    than the radius."""

    test_radii = test_locations[:, -1]

    # Transform the voxel coordinates into physical coordinates. TransformContinuousIndexToPhysicalPoint handles
    # sub-voxel (i.e. floating point) indices.
    test_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord[:3]) for coord in test_locations.astype(float)])
    pred_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord) for coord in result_locations.astype(float)])
    treated_locations = get_treated_locations(test_image)
    treated_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord.astype(float)) for coord in treated_locations.astype(float)])
    
    # Reshape empty arrays into 0x3 arrays.
    if test_coords.size == 0:
        test_coords = test_coords.reshape(0, 3)
    if pred_coords.size == 0:
        pred_coords = pred_coords.reshape(0, 3)
    
    # True positives lie within radius  of true aneurysm. Only count one true positive per aneurysm.
    true_positives = 0
    for location, radius in zip(test_coords, test_radii):
        detected = False
        for detection in pred_coords:
            distance = np.linalg.norm(detection - location)
            if distance <= radius:
                detected = True
        if detected:
            true_positives += 1
    
    false_positives = 0
    for detection in pred_coords:
        found = False
        if detection in treated_coords:
            continue
        for location, radius in zip(test_coords, test_radii):
            distance = np.linalg.norm(location - detection)
            if distance <= radius:
                found = True 
        if not found:
            false_positives += 1
            
    if len(test_locations) == 0:
        sensitivity = np.nan
    else:
        sensitivity = true_positives / len(test_locations)
      
    return sensitivity, false_positives

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAM Aneurysm Detection Challenge Evaluation")

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
