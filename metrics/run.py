import argparse
import os
import subprocess
import sys


def _write_ground_truth_to_file(dset, task_dir, ground_truth_path):
    """
    Write the ground-truth natural-language outputs to file. Thereafter, this file
    will be used by the metrics runner in [e2e-metrics](https://github.com/tuetschek/e2e-metrics#usage).

    To complete this task, we require iterating over both the structured inputs, called `x`, as well
    as the ground-truth, called `y`.

    For each distinct entry in the former, we may have multiple outputs in the latter.

    For example, the first 3 rows of `x` might be:

    __start_name__ Alimentum __end_name__ __start_area__ city centre __end_area__ __start_familyFriendly__ no __end_familyFriendly__
    __start_name__ Alimentum __end_name__ __start_area__ city centre __end_area__ __start_familyFriendly__ no __end_familyFriendly__
    __start_name__ Alimentum __end_name__ __start_area__ city centre __end_area__ __start_familyFriendly__ no __end_familyFriendly__

    The (corresponding) first 3 rows of `y` might be:

    There is a place in the city centre , Alimentum , that is not family - friendly . <eos>|||6,8,5 8,9,7 9,10,0 10,11,7 17,18,7 18,19,8
    In the city centre there is a venue name Alimentum , this is not a family - friendly venue . <eos>|||2,4,5 9,10,0 10,11,7 19,20,7 20,21,8
    Alimentum is not a family - friendly place , located in city centre . <eos>|||0,1,0 8,9,7 11,13,5 13,14,7 14,15,8

    One-to-many.

    Finally, we parse the latter then write to disk. The required format is: one natural-language output per line, where blocks
    of multiple outputs corresponding to a single distinct input are separate by a newline character.

    For example: https://github.com/tuetschek/e2e-metrics/blob/master/example-inputs/devel-conc.txt
    """

    # x: the structured input
    # y: the ground-truth natural-language output

    x_fname = f'src_{dset}.txt'
    y_fname = f'{dset}.txt'

    x_path = os.path.join(task_dir, x_fname)
    y_path = os.path.join(task_dir, y_fname)

    # Read x, y into memory
    x = open(x_path, 'r').readlines()
    y = open(y_path, 'r').readlines()
    assert len(x) == len(y), 'Your structured inputs (x) and natural-language outputs (y) do not line up 1 to 1.'

    if os.path.exists(ground_truth_path):
        os.remove(ground_truth_path)
    ground_truth_file = open(ground_truth_path, 'a')

    last_x_row = ''
    for i, (x_row, y_row) in enumerate(zip(x, y)):
        # If we encounter an output corresponding to a novel distinct input, separate this block with a newline.
        if i > 0 and x_row != last_x_row:
            ground_truth_file.write('\n')

        # An example `y_row`: `There is a place in the city centre , Alimentum , that is not family - friendly . <eos>|||6,8,5 8,9,7 9,10,0 10,11,7 17,18,7 18,19,8`
        # This line will extract: `There is a place in the city centre , Alimentum , that is not family - friendly .`
        output = y_row.split('<eos>|||')[0].strip()
        ground_truth_file.write(output + '\n')

        last_x_row = x_row
    ground_truth_file.close()


def _write_predictions_to_file(dset, task_dir, preds_path, gen_output_path):
    """
    One prediction (generated natural-language output) per line.

    For example: https://github.com/tuetschek/e2e-metrics/blob/master/example-inputs/baseline-output.txt
    """
    if os.path.exists(preds_path):
        os.remove(preds_path)

    gen_output = open(gen_output_path, 'r').readlines()
    start_idx = [i for i, line in enumerate(gen_output) if line.startswith('assuming we start on line')][0] + 1

    with open(preds_path, 'a') as f:
        for row in gen_output[start_idx:]:
            output = row.split('|||')[0]
            f.write(output + '\n')


def _run_metrics(ground_truth_path, preds_path, output_path, run_in_subprocess=False):
    """
    Run the metrics suite, as per: [e2e-metrics](https://github.com/tuetschek/e2e-metrics#usage).
    """
    commands = [
        'python',
        'metrics/e2e-metrics/measure_scores.py',
        ground_truth_path,
        preds_path
    ]
    if run_in_subprocess:
        _ = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print(f"\nRun:\n\n{' '.join(commands)} | tee {output_path}")
        print()


def run(task, dset, gen_output_path):
    if task == 'e2e':
        task_dir = 'data/e2e_aligned'
    else:
        raise NotImplementedError('Will/Ethan have not yet implemented logic for wikibio.')

    # Write ground-truth output to disk.
    ground_truth_path = f'metrics/tmp/{task}/measure_scores__ground_truth__{dset}.txt'
    _write_ground_truth_to_file(dset, task_dir, ground_truth_path)

    # Write predictions to disk.
    preds_path = f'metrics/tmp/{task}/measure_scores__predictions__{dset}.txt'
    _write_predictions_to_file(dset, task_dir, preds_path, gen_output_path)

    output_path = os.path.join('metrics', 'results', task, gen_output_path.split('/')[-1])
    _run_metrics(ground_truth_path, preds_path, output_path)


if __name__ == '__main__':
    desc = """
    Example usage:

    python metrics/run.py --task e2e --dset valid --gen-output-path gens/20181129-gen-e2e-300-55-5-src_uniq_valid.txt
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--task', type=str, required=True, help='[e2e|wikibio]')
    parser.add_argument('--dset', type=str, required=True, help='[train|valid|test]')
    parser.add_argument('--gen-output-path', type=str, required=True,
                        help='The relative path to the generated output, e.g. gens/gen-e2e-300-60-src_uniq_valid.txt')
    args = parser.parse_args()
    run(**args.__dict__)
