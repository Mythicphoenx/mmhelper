"""
Contains functions that allow the comparison of the code detection
with a manually labelled image
"""
import os
import numpy as np
import skimage.io as skio
from skimage.color import rgb2gray
from skimage.measure import regionprops

def find_best_match(cost_matrix):
    """
    Takes a cost matrix and determines which is the best matches

    Parameters
    ------
    cost_matrix : matrix (N x M)
        Where N is the initial labels and M is the new labels and the
        cost of matching them fills the matrix

    Returns
    ------
    manual_labels : list
        List of labels in the correct order
    detected_labels : list
        List of matching labels for the lowest cost
    """
    detected_labels = []
    manual_labels = []
    for i, j in enumerate(cost_matrix):
        lowest_cost_index = np.where(j == j.min())[0]
        if len(lowest_cost_index) > 1:
            continue
        manual_labels.append(i)
        detected_labels.append(int(lowest_cost_index))
    return manual_labels, detected_labels

def match_labels(manual_im, detected_im):
    """
    Takes two images and matches the bacteria labels based on how well
    overlapped they are

    Parameters
    ------
    manual_im : ndarray (2D) of dtype int
        A labelled image showing the manually labelled bacteria
    detected_im : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria

    Returns
    ------
    man_ids : list
        List of labels in the correct order
    det_ids : list
        List of matching labels for the lowest cost
    man_id_dict : Dictionary
        linking the output from man_ids to the actual label
    det_id_dict : dictionary
        linking the output from det_ids to the actual detected label
    """
    if len(regionprops(detected_im)) == 0 or len(regionprops(manual_im)) == 0:
        return [], [], {}, {}
    axis_w = np.trim_zeros(np.unique(manual_im))
    axis_h = np.trim_zeros(np.unique(detected_im))
    cost_matrix = np.zeros(((len(axis_w), len(axis_h))))
    man_id_dict = {}
    det_id_dict = {}
    for i, lbl1 in enumerate(regionprops(manual_im)):
        for j, lbl2 in enumerate(regionprops(detected_im)):
            overlap_size = np.count_nonzero(
                manual_im[detected_im == lbl2.label] == lbl1.label)
            combined_area = (lbl1.area - overlap_size) + lbl2.area
            jac_score = overlap_size / combined_area
            cost_matrix[i, j] = 1 - jac_score
            det_id_dict[j] = lbl2.label
        man_id_dict[i] = lbl1.label
    man_ids, det_ids = find_best_match(cost_matrix)
    return man_ids, det_ids, man_id_dict, det_id_dict


def determine_precision_recall(
        manual_im, detected_im, man_ids, det_ids, man_id_dict, det_id_dict):
    """
    Takes a set of data from a manually labelled and a code detected image
    and determines the precision and recall scores

    Parameters
    ------
    manual_im : ndarray (2D) of dtype int
        A labelled image showing the manually labelled bacteria
    detected_im : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    man_ids : list
        List of labels in the correct order
    det_ids : list
        List of matching labels for the lowest cost
    man_id_dict : Dictionary
        linking the output from man_ids to the actual label
    det_id_dict : dictionary
        linking the output from det_ids to the actual detected label


    Returns
    ------
    precision_scores : list
        A list of scores showing how precise each detection was
        (accuracy of coverage)
    recall_scores : list
        A list of scores showing how well covered the detection was
        (amount of coverage)
    """
    if (len(man_ids) == 0) or (len(det_ids) == 0):
        return [], []
    precision_scores = []
    recall_scores = []
    for i, j in zip(man_ids, det_ids):
        man_label = man_id_dict[i]
        det_label = det_id_dict[j]
        overlap_size = np.count_nonzero(
            manual_im[detected_im == det_label] == man_label)
        manual_label_size = (manual_im == man_label).sum()
        detected_label_size = (detected_im == det_label).sum()
        precision_cover = overlap_size / manual_label_size
        recall_cover = overlap_size / detected_label_size
        precision_scores.append(precision_cover)
        recall_scores.append(recall_cover)
    return precision_scores, recall_scores


def compare_detection(dir_name, label_output,
                      fluo_data=None, bac_or_well=None):
    """
    Compares the output detection from the code to the same image manually labelled

    Parameters
    ------
    dir_name : str (path)
        The directory where the output detection images are stored
    label_output : str (path)
        The directory where the manual comaprison images are stored
    fluo_data : ndarray (2D), optional
        Matching fluorescent data to the images that were analysed
    bac_or_well : str
        Helps to determine if the comparison is for well detection
        or bacteria detection - the string should be at the end of
        the file names

    Returns
    ------
    combined_precision : list
        A list of scores showing how precise each detection was
        (accuracy of coverage) for all images compared
    combined_recall : list
        A list of scores showing how well covered the detection was
        (amount of coverage) for all images compared
    """
    if os.path.isdir(label_output):
        man_image_list = sorted([f for f in os.listdir(label_output)
                                 if f.endswith('%s.png' % (bac_or_well))])

    if os.path.isdir(dir_name):
        det_image_list = sorted(
            [f for f in os.listdir(dir_name)
             if f.startswith('%s_' % (bac_or_well)) and f.endswith('.png')])

    num_frames_run = len(det_image_list)
    man_image_list = man_image_list[:num_frames_run]

    manual_images = rgb2gray(np.array([skio.imread(os.path.join(
        label_output, f)) for f in man_image_list], dtype=float)).astype(int)
    detected_images = np.array([(skio.imread(os.path.join(dir_name, f)))
                                for f in det_image_list])

    combined_precision = []
    combined_recall = []
    for manual_im, detected_im in zip(manual_images, detected_images):
        man_ids, det_ids, man_id_dict, det_id_dict = match_labels(
            manual_im, detected_im)
        precision_scores, recall_scores = determine_precision_recall(
            manual_im, detected_im, man_ids, det_ids, man_id_dict, det_id_dict)
        combined_precision.append(precision_scores)
        combined_recall.append(recall_scores)
    combined_precision = [val for l in combined_precision for val in l]
    combined_recall = [val for l in combined_recall for val in l]
    return combined_precision, combined_recall
