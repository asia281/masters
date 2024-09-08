"""Unit test for excercises."""
import argparse

import numpy as np
import torch

from create_delex_data import belief_state, requested_state
from model.policy import SoftmaxPolicy
from utils.db_pointer import one_hot_vector
from utils.delexicalize import prepare_slot_values_independent


_DOMAIN = "restaurant"

def test_task_a():
    summary_bstate = []
    bstate = {_DOMAIN: 
        {'semi': {
            'food': 'british',
            'pricerange': "don't care",
            'name': 'not mentioned',
            'area': 'centre'}
        }
    }
    summary_bstate = belief_state(bstate, summary_bstate, _DOMAIN)
    expected = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]

    assert len(summary_bstate) == 12
    assert all([v1 == v2 for v1, v2 in zip(summary_bstate, expected)])

    print ("-"*80)
    print("All Sanity Checks Passed for Question 1a")
    print ("-"*80)


def test_task_b():
    num_entities = 7
    vector = np.zeros(6)

    available_entities = one_hot_vector(num_entities, _DOMAIN, vector)
    expected = np.array([0, 0, 0, 1, 0, 0])

    assert len(available_entities) == 6
    assert all([v1 == v2 for v1, v2 in zip(available_entities, expected)])

    print ("-"*80)
    print("All Sanity Checks Passed for Question 1b")
    print ("-"*80)


def test_task_c():
    summary_bstate = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]
    bstate = {_DOMAIN: 
        {        
        'semi': {
            'food': 'british', 'requested': [""],
            'pricerange': "don't care",
            'name': 'not mentioned',
            'area': 'centre'}
        }
    }
    summary_bstate = requested_state(bstate, summary_bstate, _DOMAIN)
    expected = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    assert len(summary_bstate) == 19
    assert all([v1 == v2 for v1, v2 in zip(summary_bstate, expected)])

    print ("-"*80)
    print("All Sanity Checks Passed for Question 1c")
    print ("-"*80)


def test_task_d():
    delexicalized_list = prepare_slot_values_independent()

    assert ('centre', '[value_area]') in delexicalized_list
    assert ('italian', '[value_food]') in delexicalized_list
    print ("-"*80)
    print("All Sanity Checks Passed for Question 1d")
    print ("-"*80)


def test_task_f():
    # python train.py --policy softmax
    policy = SoftmaxPolicy(50, 50, 6, 19)
    encodings = (torch.rand([1, 6, 50]), torch.rand([1, 6, 50]))
    db_tensor = torch.rand([6, 6])
    bs_tensor = torch.rand([6, 19])

    output = policy(encodings, db_tensor, bs_tensor)

    assert len(output) == 2
    assert type(output) == tuple
    assert output[0].shape[0] == 1
    assert output[0].shape[1] == 6
    assert output[0].shape[2] == 50

    print ("-"*80)
    print("All Sanity Checks Passed for Question 1f")
    print ("-"*80)


def main(args):
    if args.task == "a":
        test_task_a()
    elif args.task == "b":
        test_task_b()
    elif args.task == "c":
        test_task_c()
    elif args.task == "d":
        test_task_d()
    elif args.task == "f":
        test_task_f()
    else:
        raise ValueError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2S')
    parser.add_argument(
        '--task', type=str, default="a", choices=['a', 'b', 'c', 'd', 'f'])
    args = parser.parse_args()

    main(args)
