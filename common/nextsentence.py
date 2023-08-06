#!/usr/bin/env python3
"""
This module defines nextsentence dataset preparation.
"""
import numpy as np

def class_reducer(sentence_distance):
    """
    class_reducer converts the distance to the desired predictor class.
    """
    return 1 / (sentence_distance + 1)

def choose_second_sentence(nwindow, index):
    """
    choose_second_sentence chooses the second sentence within nwindow assuming the first sentence
    is at index of window.
    """
    # nwindow is the size of the sliding window of sentences to choose from.
    # window[index] is the first sentence in the output pair
    # randomly return the index of the second sentence
    assert nwindow > 1

    if index == 0:
        s2_prob = [0, 0.5] + [0.5 / (nwindow - 2)] * (nwindow - 2)
    elif index == nwindow - 1:
        s2_prob = [0.5 / (nwindow - 2)] * (nwindow - 2) + [0.5, 0]
    else:
        nnp = 0.5 / (nwindow - 3) # non-neighbor prob
        s2_prob = [nnp] * (index - 1) + [0.25, 0, 0.25] + [nnp] * (nwindow - index - 2)

    s2_index = np.random.choice(nwindow, 1 , p=s2_prob)[0]
    return s2_index
