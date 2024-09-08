# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from utils import evaluate_places
import argparse
    

argp = argparse.ArgumentParser()
argp.add_argument('--eval_path',
    help="Path of the data to evaluate model on", default=None)
print(argp) 
args = argp.parse_args()
path = args.eval_path

preds = []
for _ in open(path, 'r'):
    preds.append('London')

total, correct = evaluate_places(path, preds)
print('London baseline accuracy: %.2f' % (correct / total))