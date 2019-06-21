# FILE        : bacteria_tracking.py
# CREATED     : 14/11/17 13:08:52
# AUTHOR      : A. Smith <as624@exeter.ac.uk>
# DESCRIPTION : Bacteria tracking functions
#
"""Bacteria tracking functions
"""
from functools import reduce
from skimage.measure import regionprops
import numpy as np

def find_changes(in_list, option_list, well, new_well):
    """
    Takes a list

    Parameters
    ------
    in_list : list
        A list of labels from the current well
    option_list : list
        A list of all the possible combinations possible of how the bacteria
        in the previous well could be in the new well
    well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the old well
    new_well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the new well

    Yields
    ------
    option : list
        Containing the potential output combination
    in_options_dict : dictionary
        where the key is one of the input bacteria labels and the values
        is a list of the number of divisions, area change and centroid change
        for that respective bacteria for that potential output combination
    """
    measurements_in = {}
    measurements_out = {}
    for i, region in enumerate(regionprops(well)):
        # find y centr coord and area of old each bac
        measurements_in[i] = [region.centroid[0], region.area]
    for j, region2 in enumerate(regionprops(new_well)):
        # find y centr coord and area of each new bac
        measurements_out[j] = [region2.centroid[0], region2.area]
    for option in option_list:  # each option is a potential combination of bacteria lineage/death
        in_options_dict = {}
        for in_num, in_options in enumerate(in_list):
            out_bac_area = []
            out_bac_centr = []
            # determine the number of divisions/deaths
            num_divs = (option.count(in_options)) - 1
            for lst, opt in enumerate(option):
                if opt == in_options:  # if the values match append the new centr/areas
                    out_bac_area.append(measurements_out[lst][1])
                    out_bac_centr.append(measurements_out[lst][0])
            # need to divide by biggest number (so prob < 1)
            if sum(out_bac_area) < (measurements_in[in_num][1]):
                # find relative change in area compared to original
                area_chan = sum(out_bac_area) / (measurements_in[in_num][1])
            else:
                # find relative change in area compared to original
                area_chan = (measurements_in[in_num][1]) / sum(out_bac_area)
            if len(out_bac_centr) is not 0:
                # find the average new centroid
                centr_chan = abs(((sum(out_bac_centr)) / (len(out_bac_centr)))
                                 - (measurements_in[in_num][0]))
            else:
                centr_chan = 0
            # assign the values to the correct 'in' label
            in_options_dict[in_options] = [num_divs, area_chan, centr_chan]
        # change_dict[option] = in_options_dict #assign the changes to the
        # respective option
        yield option, in_options_dict  # assign the changes to the respective option
        # return change_dict


def find_probs(
        probs,
        prob_div=0.01,
        prob_death=0.5,
        prob_no_change=0.95,
        av_bac_length=18,
    ):
    """
    Takes a dictionary of information for a potential combination
    and returns an overall probability

    Parameters
    ------
    probs : dictionary
        Key is a unique number of an input bacteria and the value is a
        list of the number of divisions, area change and centroid change
        for that respective bacteria
    prob_div : float, optional
        Probability a bacteria divides between consecutive timepoints (default : 0.01)
    prob_death : float, optional
        Probability a bacteria lyses between consecutive timepoints (default : 0.5)
    prob_no_change : float, optional
        Probability there is no change between consecutive timepoints (default : 0.95)
    av_bac_length : float, optional
        The average bacteria length in pixels (default : 18)

    Returns
    ------
    combined_prob : float
        The overall probability for this combination of events
    """
    probslist = []
    for pro in probs:
        # find the potential number of deaths/divisions for each bac
        divs_deaths = probs[pro][0]
        relative_area = probs[pro][1]  # find the relative area change
        # find the number of pixels the centroid has moved by
        change_centr = probs[pro][2]
        if divs_deaths < 0:  # if the bacteria has died:
            prob_divis = prob_death  # probability simply equals that of death
            prob_centr = 1  # the change in centroid is irrelevant so set probability as 1
            prob_area = 1  # the change in area is irrelevant so set probability as 1
        if divs_deaths == 0:  # if the bacteria hasn't died/or divided
            # probability of division simply equals probability of no change
            prob_divis = prob_no_change
            # the area will be equal to the relative area change - may need
            # adjusting
            prob_area = relative_area
            # if there is no change then set prob to 1 (0 will cause div error)
            if change_centr == 0:
                prob_centr = 1
            else:
                # the greater the change the less likely
                prob_centr = 1 / (abs(change_centr))
        if divs_deaths > 0:  # if bacteria have divided:
            # need to make sure we divide by biggest number to keep prob < 1
            if relative_area < divs_deaths:
                # normalise relative area to the number of divisions
                prob_area = relative_area / divs_deaths
            else:
                # normalise relative area to the number of divisions
                prob_area = divs_deaths / relative_area
            # each division becomes more likely - need to think about it
            prob_divis = prob_div**(divs_deaths * divs_deaths)
            # for each division the bacteria centroid is expected to move half
            # the bac length
            prob_centr = 1 / \
                abs(((divs_deaths * (av_bac_length / 2)) - (change_centr)))
        # combine the probabilities for division, area and centroid
        probslist.append(prob_area * prob_divis * prob_centr)
    # multiply the probabilities across all bacteria
    combined_prob = reduce(lambda x, y: x * y, probslist)
    return combined_prob


def label_most_likely(most_likely, new_well, label_dict_string):
    """
    Takes the most likely combination of how the bacteria may have
    divided/died or moved around and re-labels them accordingly

    Parameters
    ------
    most_likely : list
        Containing the most likely output combination
    new_well : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria in the new well
    label_dict_string : dictionary
        Each key is a unique label of a bacteria, each value is
        a string containing its lineage information

    Returns
    ------
    out_well : ndarray (2D) of dtype int
        A labelled image showing the tracked bacteria in the new well
    label_dict_string : dictionary
        Updated dictionary where each key is a unique label of a bacteria,
        each value is a string containing its lineage information
    """
    out_well = np.zeros(new_well.shape, dtype=new_well.dtype)
    if most_likely is None:
        # if there is no likely option return an empty well
        return out_well, label_dict_string
    new_label_string = 0
    smax = 0
    smax = max(label_dict_string, key=int)
    for i, region in enumerate(regionprops(new_well)):
        if most_likely.count(most_likely[i]) == 1:
            out_well[new_well == region.label] = most_likely[i]
        else:
            smax += 1
            out_well[new_well == region.label] = smax
            if i > 0:
                last_label_start = label_dict_string[most_likely[i - 1]]
            else:
                last_label_start = label_dict_string[most_likely[i]]
            new_label_start = label_dict_string[most_likely[i]]
            if new_label_start != last_label_start:
                new_label_string = 0
            new_label_string += 1
            add_string = "_%s" % (new_label_string)
            label_dict_string[smax] = new_label_start + add_string
    return out_well, label_dict_string
