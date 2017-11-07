from skimage.measure import regionprops
import matplotlib.pyplot as plt
import itertools
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

prob_div = 0.01 #we need a way of determining these but will hard code for now
prob_death = 0.5 #we need a way of determining these but will hard code for now
prob_no_change = 0.95 #may need this to be independent of prob_death/prob_div
av_bac_length = 18 #average back length - may want to code this in - actual bacteria nearer 18


    
def find_changes(in_list, option_list, well, new_well):
    measurements_in = {}
    measurements_out = {}
    change_dict = {}
    for i, region in enumerate(regionprops(well)):
        measurements_in[i] = [region.centroid[0], region.area] #find y centr coord and area of old each bac
    for j, region2 in enumerate(regionprops(new_well)):
        measurements_out[j] = [region2.centroid[0], region2.area] #find y centr coord and area of each new bac
    for option in option_list: #each option is a potential combination of bacteria lineage/death
        in_options_dict = {}
        for in_num, in_options in enumerate(in_list):                
            out_bac_area = []
            out_bac_centr = []
            num_divs = (option.count(in_options))-1 #determine the number of divisions/deaths
            for l, op in enumerate(option):
                if op == in_options: #if the values match append the new centr/areas
                    out_bac_area.append(measurements_out[l][1])
                    out_bac_centr.append(measurements_out[l][0])                        
            if sum(out_bac_area) < (measurements_in[in_num][1]): #need to divide by biggest number (so prob < 1)
                area_chan = sum(out_bac_area)/(measurements_in[in_num][1]) #find relative change in area compared to original
            else:
                area_chan = (measurements_in[in_num][1])/sum(out_bac_area) #find relative change in area compared to original
            if len(out_bac_centr) is not 0: 
                centr_chan = abs(((sum(out_bac_centr))/(len(out_bac_centr)))-(measurements_in[in_num][0])) #find the average new centroid
            else:
                centr_chan = 0
            #assign the values to the correct 'in' label
            in_options_dict[in_options] = [num_divs, area_chan, centr_chan] 
        change_dict[option] = in_options_dict #assign the changes to the respective option
    return change_dict
        
def find_probs(change_dict):
    prob_dict = {}
    temp_dict = {}
    if len(change_dict) == 0:
        most_likely = None
        return most_likely 
    for option, probs in change_dict.items():
        probslist = []
        for p in probs:
            divs_deaths = probs[p][0] #find the potential number of deaths/divisions for each bac
            relative_area = probs[p][1] #find the relative area change
            change_centr = probs[p][2]  #find the number of pixels the centroid has moved by
            if divs_deaths<0: #if the bacteria has died:
                prob_divis = prob_death #probability simply equals that of death
                prob_centr = 1 #the change in centroid is irrelevant so set probability as 1
                prob_area = 1 #the change in area is irrelevant so set probability as 1
            if divs_deaths == 0: #if the bacteria hasn't died/or divided
                prob_divis = prob_no_change #probability of division simply equals probability of no change
                prob_area = relative_area #the area will be equal to the relative area change - may need adjusting
                if change_centr == 0: #if there is no change then set prob to 1 (0 will cause div error)
                    prob_centr = 1
                else:
                    prob_centr = 1/(abs(change_centr)) #the greater the change the less likely
            if divs_deaths > 0: #if bacteria have divided:
                if relative_area < divs_deaths: #need to make sure we divide by biggest number to keep prob < 1
                    prob_area = relative_area/divs_deaths #normalise relative area to the number of divisions
                else:
                    prob_area = divs_deaths/relative_area #normalise relative area to the number of divisions
                prob_divis = prob_div**(divs_deaths*divs_deaths) #each division becomes more likely - need to think about it
                #for each division the bacteria centroid is expected to move half the bac length
                prob_centr = 1/abs(((divs_deaths*(av_bac_length/2))-(change_centr)))
            probslist.append(prob_area*prob_divis*prob_centr) #combine the probabilities for division, area and centroid
            temp_dict[p] = prob_area*prob_divis*prob_centr #same as probslist but makes output more readable during development
        prob_dict[option] = reduce(lambda x, y: x*y, probslist) #multiply the probabilities across all bacteria
        most_likely = max(prob_dict, key=prob_dict.get)
    return most_likely
        
def label_most_likely(most_likely, new_well, label_dict_string):
    out_well = np.zeros(new_well.shape, dtype=new_well.dtype)
    if most_likely is None:
        return out_well, label_dict_string #if there is no likely option return an empty well
    new_label_string = 0
    smax = max(label_dict_string, key=int)
    for i, region in enumerate(regionprops(new_well)):
        if most_likely.count(most_likely[i]) == 1:
            out_well[new_well==region.label] = most_likely[i]
        else:        
            smax+=1
            out_well[new_well==region.label] = smax
            if i > 0:
                last_label_start = label_dict_string[most_likely[i-1]]
            else:
                last_label_start = label_dict_string[most_likely[i]]
            new_label_start = label_dict_string[most_likely[i]]
            if new_label_start != last_label_start:
                new_label_string = 0
            new_label_string += 1
            add_string = "_%s" % (new_label_string)
            label_dict_string[smax] = new_label_start+add_string
    return out_well, label_dict_string
        
