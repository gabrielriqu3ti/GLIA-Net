# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

import evaluation_detection as eval_det
import evaluation_segmentation as eval_seg

from utils.project_utils import read_dict_csv


def get_thresholds(threshold_sampling: str, nb_threshold: int) -> np.ndarray:
    if threshold_sampling == 'logspace':
        thresholds = ((np.logspace(0, 1, nb_threshold + 2) - 1) / 9)[1: -1]
    elif threshold_sampling == 'logspace_pro':
        thresholds = ((np.logspace(0, 1, nb_threshold + 2, base=100) - 1) / 99)[1: -1]
    elif threshold_sampling == 'linspace':
        thresholds = np.linspace(0.0, 1.0, nb_threshold + 2)[1: -1]
    elif threshold_sampling == 'uline':
        thresholds = (((np.logspace(0, 1, nb_threshold // 2 + 2,
                                    base=10000000000) - 1) / 9999999999)[1: -1]) / 2
        if nb_threshold % 2 == 1:
            thresholds = np.append(thresholds, 0.5)
        for i in range(nb_threshold // 2 - 1, -1, -1):
            thresholds = np.append(thresholds, 1.0 - thresholds[i])
    else:
        raise ValueError("ERROR: unrecognized threshold sampling. Valid values are 'logspace', 'logspace_pro', "
                         "'linspace' and 'uline'")

    return thresholds.astype(np.float32)


def plot_performance(opt_metric: str, threshold_np: np.ndarray, dscs: List[float], hd95s: List[float], vss: List[float],
                     sensitivities: List[float], fpss: List[float], enable_detection: bool) -> None:

    # Exception handling
    if len(threshold_np) != len(dscs):
        raise ValueError(f"Error: 'dscs' must have the same length as 'threshold_np'. However {len(dscs)} != {len(threshold_np)}")
    if len(threshold_np) != len(hd95s):
        raise ValueError(f"Error: 'hd95s' must have the same length as 'threshold_np'. However {len(hd95s)} != {len(threshold_np)}")
    if len(threshold_np) != len(vss):
        raise ValueError(f"Error: 'vss' must have the same length as 'threshold_np'. However {len(vss)} != {len(threshold_np)}")
    if len(threshold_np) != len(sensitivities):
        raise ValueError(f"Error: 'sensivities' must have the same length as 'threshold_np'. However {len(sensitivities)} != {len(threshold_np)}")
    if len(threshold_np) != len(fpss):
        raise ValueError(f"Error: 'fpss' must have the same length as 'threshold_np'. However {len(fpss)} != {len(threshold_np)}")

    print('---BEST STATISTICS---')
    print(f'Optimization of the metric {opt_metric}')

    if opt_metric == 'DSC':
        best_threshold = np.argmax(np.array(dscs))
    elif opt_metric == 'HD95':
        best_threshold = np.argmax(np.array(hd95s))
    elif opt_metric == 'VS':
        best_threshold = np.argmax(np.array(vss))
    elif opt_metric == 'Sensitivity':
        best_threshold = np.argmax(np.array(sensitivities))
    elif opt_metric == '#FP/case':
        best_threshold = np.argmax(np.array(fpss))
    else:
        raise ValueError(f'Unsupported optimization metric {opt_metric}')

    print(f'Best threshold: {threshold_np[best_threshold]:.3f}')

    print('Dice: %.3f (higher is better, min=0, max=1)' % dscs[best_threshold])
    print('HD: %.3f mm (lower is better, min=0, max=+inf)' % hd95s[best_threshold])
    print('VS: %.3f (higher is better, min=0, max=1)' % vss[best_threshold])

    if enable_detection:
        print('Sensitivity: %.3f (higher is better, min=0, max=1)' % sensitivities[best_threshold])
        print('False Positive Count: %.1f (lower is better, min=0, max=+inf)' % fpss[best_threshold])

    dsc_np = np.array(dscs, dtype=threshold_np.dtype)
    hd95_np = np.array(hd95s, dtype=threshold_np.dtype)
    vs_np = np.array(vss, dtype=threshold_np.dtype)
    sensitivity_np = np.array(sensitivities, dtype=threshold_np.dtype)
    fps_np = np.array(fpss, dtype=threshold_np.dtype)

    # Plotting
    fig_voxel, ax_voxel_01 = plt.subplots()
    color_01 = 'tab:blue'
    labels_01 = [i / 10 for i in range(11)]
    ax_voxel_01.set_xlabel('Limiar')
    ax_voxel_01.plot(threshold_np, dsc_np, 'b*-', label='DSC')
    ax_voxel_01.plot(threshold_np, vs_np, 'b.--', label='VS')
    ax_voxel_01.tick_params(axis='y', labelcolor=color_01)
    ax_voxel_01.set_ylabel('DSC, VS', color=color_01)
    ax_voxel_01.set_ylim(bottom=0, top=1)
    ax_voxel_01.set_yticks(labels_01)
    ax_voxel_01.set_yticklabels(labels_01)
    ax_voxel_01.set_xlim(left=0, right=1)
    ax_voxel_01.set_xticks(labels_01)
    ax_voxel_01.set_xticklabels(labels_01)
    ax_voxel_01.set_title('Métricas por voxel')
    ax_voxel_01.grid()

    ax_voxel_mm = ax_voxel_01.twinx()
    color_mm = 'tab:red'
    ax_voxel_mm.plot(threshold_np, hd95_np, 'r.-', label='HD95')
    ax_voxel_mm.set_ylabel('HD95 [mm]', color=color_mm)
    ax_voxel_mm.set_xlim(left=0, right=1)
    ax_voxel_mm.tick_params(axis='y', labelcolor=color_mm)
    ax_voxel_mm.grid()

    fig_voxel.legend()
    fig_voxel.tight_layout()
    fig_voxel.savefig('metrics_per_voxel.png', dpi=300)

    if enable_detection:
        fig_target, ax_target_01 = plt.subplots()
        ax_target_01.set_xlabel('Limiar')
        ax_target_01.plot(threshold_np, sensitivity_np, 'b.-', label='Sensibilidade')
        ax_target_01.tick_params(axis='y', labelcolor=color_01)
        ax_target_01.set_ylabel('Sensibilidade', color=color_01)
        ax_target_01.set_ylim(bottom=0, top=1)
        ax_target_01.set_yticks(labels_01)
        ax_target_01.set_yticklabels(labels_01)
        ax_target_01.set_xlim(left=0, right=1)
        ax_target_01.set_xticks(labels_01)
        ax_target_01.set_xticklabels(labels_01)
        ax_target_01.set_title('Métricas por alvo')
        ax_target_01.grid()

        ax_target_ = ax_target_01.twinx()
        color_ = 'tab:red'
        ax_target_.plot(threshold_np, fps_np, 'r.-', label='#FP/caso')
        ax_target_.set_ylabel('FPs/caso', color=color_)
        ax_target_.set_xlim(left=0, right=1)
        ax_target_.tick_params(axis='y', labelcolor=color_)
        ax_target_.grid()

        fig_target.legend()
        fig_target.tight_layout()
        fig_target.savefig('metrics_per_target.png', dpi=300)

        plt.show()


def get_images(test_filename, result_filename, threshold=0.5):
    """Return the test and result images, thresholded and treated aneurysms removed."""
    test_image = sitk.ReadImage(test_filename, imageIO='NiftiImageIO')
    result_image = sitk.ReadImage(result_filename, imageIO='NiftiImageIO')

    assert test_image.GetSize() == result_image.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)

    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treated_image = test_image != 2  # treated aneurysms == 2
    masked_result_image = result_image
    masked_test_image = sitk.Mask(test_image, treated_image)

    # Return two binary masks
    return masked_test_image > threshold, masked_result_image > threshold


def main(output_dir, annotation_dir, location_dir, metadata_file, threshold_sampling, nb_threshold, opt_metric):
    print(f'Number of thresholds = {nb_threshold}')
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

    available_opt_metrics = ['DSC', 'HD95', 'VS', 'Sensitivity', '#FP/case']
    if opt_metric not in available_opt_metrics:
        raise ValueError(f"Invalid metric {opt_metric}. Available options are {available_opt_metrics}")

    metadata = read_dict_csv(dataset_metadata_path.absolute().as_posix())
    metadata = [exam for exam in metadata if exam['subset'] == 'eval']

    thresholds = get_thresholds(threshold_sampling, nb_threshold)
    avg_dsc_list = []
    avg_h95_list = []
    avg_vs_list = []

    avg_sensitivity_list = []
    avg_fps_list = []

    for i in range(nb_threshold):
        print(f'iter {i}')
        dsc_list = []
        h95_list = []
        vs_list = []

        sensitivity_list = []
        fps_list = []

        for exam_i, exam in enumerate(metadata, 1):
            filename = exam['aneurysm_seg_file'].split('/')[-1]
            filename_location = filename.split('.')[0] + '.txt'
            filename_prob = filename.split('.')[0] + '_prob.nii.gz'
            # print(f'    exam {filename_prob}')

            output_path = output_dir_path / filename_prob
            if not output_path.exists():
                raise Exception('output file not found at %s' % output_path.absolute())
            annotation_path = annotation_dir_path / filename
            if not annotation_path.exists():
                raise Exception('annotation file not found at %s' % annotation_path.absolute())

            annotation_image, output_image = get_images(annotation_path.absolute().as_posix(),
                                                        output_path.absolute().as_posix(), thresholds[i])

            dsc = eval_seg.get_dsc(annotation_image, output_image)
            h95 = eval_seg.get_hausdorff(annotation_image, output_image)
            vs = eval_seg.get_vs(annotation_image, output_image)

            if not np.isnan(dsc):
                dsc_list.append(dsc)
            if not np.isnan(h95):
                h95_list.append(h95)
            if not np.isnan(vs):
                vs_list.append(vs)

            if detection:
                annotation_locations = eval_det.get_locations(location_dir_path / filename_location)
                output_locations = eval_seg.get_center_of_mass(output_image)

                sensitivity, fps = eval_det.get_detection_metrics(annotation_locations, output_locations,
                                                                  annotation_image)

                if not np.isnan(sensitivity):
                    sensitivity_list.append(sensitivity)
                fps_list.append(fps)

        if dsc_list:
            avg_dsc = sum(dsc_list) / len(dsc_list)
        else:
            avg_dsc = np.nan
        if h95_list:
            avg_h95 = sum(h95_list) / len(h95_list)
        else:
            avg_h95 = np.nan
        if vs_list:
            avg_vs = sum(vs_list) / len(vs_list)
        else:
            avg_vs = np.nan
        if sensitivity_list:
            avg_sensitivity = sum(sensitivity_list) / len(sensitivity_list)
        else:
            avg_sensitivity = np.nan
        avg_fps = sum(fps_list) / len(fps_list)

        avg_dsc_list.append(avg_dsc)
        avg_h95_list.append(avg_h95)
        avg_vs_list.append(avg_vs)
        avg_sensitivity_list.append(avg_sensitivity)
        avg_fps_list.append(avg_fps)

        if i % np.ceil(nb_threshold / 10) == 0:
            print(f'---OVERALL STATISTICS {100 * i / nb_threshold:2.1f} % threshold={thresholds[i]:1.3f} ---')
            print('Dice: %.3f (higher is better, min=0, max=1)' % avg_dsc)
            print('HD: %.3f mm (lower is better, min=0, max=+inf)' % avg_h95)
            print('VS: %.3f (higher is better, min=0, max=1)' % avg_vs)

            if detection:
                print('Sensitivity: %.3f (higher is better, min=0, max=1)' % avg_sensitivity)
                print('False Positive Count: %.1f (lower is better, min=0, max=+inf)' % avg_fps)

    plot_performance(opt_metric, thresholds, avg_dsc_list, avg_h95_list, avg_vs_list, avg_sensitivity_list,
                     avg_fps_list, detection)


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
    parser.add_argument('--threshold_sampling', type=str, default='linspace',
                        help="Threshold sampling to convert the probabilities of a segmentation mask into classes.  "
                             "Available options are 'logspace', 'logspace_pro', 'linspace' and 'uline'")
    parser.add_argument('--nb_threshold', type=int, default=100,
                        help="Number of thresholds to sample.")
    parser.add_argument('--optimization_metric', type=str, default='DSC',
                        help="Metric to optimize. Available options are 'DSC', 'HD95', 'Sensitivity' and '#FP/case'")

    args = parser.parse_args()

    main(args.output_dir, args.annotation_dir, args.location_dir, args.metadata_file,
         args.threshold_sampling, args.nb_threshold, args.optimization_metric)
