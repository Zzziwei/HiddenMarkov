#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm
import heapq
from collections import defaultdict

from helpers import *
from part2 import get_emission_parameters

def log(x):
    return np.log(x)

def get_transition_parameters_2o(state_sequences):
    """
    Estimate transition paramters from a collection of state sequences
    """
    n_states = max(flatten(state_sequences)) + 1  # State space size (Excluding START and STOP)
    # transition_map[(prev_2,prev_1,current)]
    # -1 denotes the boundaries of a word. (combines START and STOP)
    transition_map = defaultdict(int)
    for states in state_sequences:
        modified_states = (-1, -1) + tuple(states) + (-1,)
        for i in range(len(modified_states) - 2):
            transition_map[modified_states[i:i+3]] += 1
    
    # Normalize
    total_count = sum(transition_map.values())
    normalized_map = {k: v/total_count for k,v in transition_map.items()}
    return normalized_map


def viterbi_2o(observations, transitions, emissions, n_states):

    n_obs = len(observations)
    states = tuple(range(n_states))
    
    # Default values for viterbi-table
    # The first item in the innermost list is the actual log-score.
    # The second is the "winning" state path that created this score.
    # You need the top k instead of the top 1 now 
    V = [[[]
          for state in states]
         for _ in range(n_obs)]
    
    # Base case
    for state in states:
        # Not that many paths we can work with here
        V[0][state] = (
            # This is the score
            log(transitions.get((-1, -1, state), 1e-323)) \
                + log(emissions[state][observations[0]]),
            # This is the list of states that led here.
            # Technically this should be [START]
            tuple()
        )
    
    # Second base case
    for state in states:
        h = []
        
        for prev_state, (prev_score, prev_state_path) in enumerate(V[0]):
            new_score = prev_score \
                + log(transitions.get((-1, prev_state, state), 0)) \
                + log(emissions[state][observations[1]])
            new_state_path = tuple(prev_state_path) + (prev_state,)

            h.append((new_score, new_state_path))
        
        max_score_path = max(h)
        V[1][state] = max_score_path

    # For each observation...
    for t in range(2, n_obs):
        # For each state...
        for state in states:
            # A list that holds the scores
            h = []
            
            # Grab ALL the previous score-path pairs
            for v2_state, (v2_score, v2_state_path) in enumerate(V[t-2]):
                for v1_state, (v1_score, v1_state_path) in enumerate(V[t-1]):
                        # Calculate a new score from previous state to the current one
                        new_score = v1_score + v2_score \
                            + log(transitions.get((v2_state, v1_state, state), 0)) \
                            + log(emissions[state][observations[t]])
                          
                        # Also save the chain of states that generated this score
                        new_state_path = tuple(v2_state_path) + (v1_state, state,)

                        h.append((new_score, new_state_path))

            # At this point, for regular viterbi just take the max item from the heap.
            # But this isn't regular viterbi
            
            max_score_path = max(h)
                        
            # V[t][state] = [max_prob, prev_state_selected]
            V[t][state] = max_score_path

    # We did a lot of the heavy lifting before
    opt = []
    for last_state, (last_score, last_state_path) in enumerate(V[-1]):
        heapq.heappush(opt, [last_score, last_state_path, last_state])
    
    for last_score, state_path, last_state in heapq.nlargest(1, opt):
        return tuple(state_path) + (last_state,)


def predict_and_save(train_file, test_file, output_file):
    features, labels = load_dataset(train_file)
    smoothed_features = smooth(features, 3)  # Input feature smoothing

    # Numerically encode dataset
    feature_ids, feature_map = encode_numeric(smoothed_features)
    label_ids, label_map = encode_numeric(labels)

    # Get HMM model parameters
    emission_matrix = get_emission_parameters(feature_ids, label_ids)
    transition_matrix = get_transition_parameters_2o(label_ids)

    # Load dev dataset, smooth and numerically encode it
    dev_features = load_dataset(test_file, split=False)
    smoothed_dev_features = smooth_dev(dev_features, smoothed_features)
    dev_feature_ids, _ = encode_numeric(
        smoothed_dev_features, token_map=feature_map
    )  # Make sure to reuse token map from the training set

    # Run Viterbi algorithm to get most likely labels
    print("Running Viterbi algorithm...")
    predicted_dev_labels = []
    for feature_id in tqdm(dev_feature_ids):
        pred = viterbi_2o(feature_id, transition_matrix, emission_matrix, len(emission_matrix))
        predicted_dev_labels.append(pred)

    # Write predictions to file
    print("Writing to file...")
    predicted_dev_labels = decode_numeric(predicted_dev_labels, label_map)
    with open(output_file, "w") as outfile:
        for dev_feature_sequence, predicted_dev_label_sequence in zip(
            dev_features, predicted_dev_labels
        ):
            for dev_feature, predicted_dev_label in zip(
                dev_feature_sequence, predicted_dev_label_sequence
            ):
                print(dev_feature, predicted_dev_label, file=outfile)
            print(file=outfile)


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"})
    DATA_ROOT = Path("data")
    DATASETS = ["SG", "CN", "EN", "AL"]
    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        predict_and_save(
            f"data/{dataset}/train",
            f"data/{dataset}/dev.in",
            f"data/{dataset}/dev.p3.out",
        )
