import argparse
import os
import subprocess
import sys


def _write_ground_truth_to_file(dset, task_dir, ground_truth_path):
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
        # Assuming a many-to-one relationship between outputs and *distinct* inputs,
        # if we encounter an output corresponding to a novel distinct input, separate
        # this block with a newline.
        if i > 0 and x_row != last_x_row:
            ground_truth_file.write('\n')

        # An example `y_row`: `There is a place in the city centre , Alimentum , that is not family - friendly . <eos>|||6,8,5 8,9,7 9,10,0 10,11,7 17,18,7 18,19,8`
        # This line will extract: `There is a place in the city centre , Alimentum , that is not family - friendly .`
        output = y_row.split('<eos>|||')[0].strip()
        ground_truth_file.write(output + '\n')

        last_x_row = x_row
    ground_truth_file.close()


def _write_predictions_to_file(dset, task_dir, preds_path, gen_output_path):
    if os.path.exists(preds_path):
        os.remove(preds_path)

    gen_output = open(gen_output_path, 'r').readlines()
    start_idx = [i for i, line in enumerate(gen_output) if line.startswith('assuming we start on line')][0] + 1

    with open(preds_path, 'a') as f:
        for row in gen_output[start_idx:]:
            output = row.split('|||')[0]
            f.write(output + '\n')


def _run_metrics(ground_truth_path, preds_path, run_in_subprocess=False):
    commands = [
        sys.executable,
        'metrics/e2e-metrics/measure_scores.py',
        ground_truth_path,
        preds_path
    ]
    if run_in_subprocess:
        _ = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print(f"\nRun:\n\n{' '.join(commands)}")
        print()



def run(task, dset, gen_output_path):
    if task == 'e2e':
        task_dir = 'data/e2e_aligned'
    else:
        raise NotImplementedError('Will/Ethan have not yet implemented logic for wikibio.')

    # Write ground-truth output in the format required by [e2e-metrics](https://github.com/tuetschek/e2e-metrics#usage).
    ground_truth_path = f'metrics/tmp/{task}/measure_scores__ground_truth__{dset}.txt'
    _write_ground_truth_to_file(dset, task_dir, ground_truth_path)

    # Write predictions in the format required by [e2e-metrics](https://github.com/tuetschek/e2e-metrics#usage).
    preds_path = f'metrics/tmp/{task}/measure_scores__predictions__{dset}.txt'
    _write_predictions_to_file(dset, task_dir, preds_path, gen_output_path)

    _run_metrics(ground_truth_path, preds_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='[e2e|wikibio]')
    parser.add_argument('--dset', type=str, required=True, help='[train|valid|test]')
    parser.add_argument('--gen-output-path', type=str, required=True,
                        help='The relative path to the generated output, e.g. gens/gen-e2e-300-60-src_uniq_valid.txt')
    args = parser.parse_args()
    run(**args.__dict__)
