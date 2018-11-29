# import csv
import os

task = 'e2e'
task_dir = 'data/e2e_aligned'

# for validation set
dset = 'valid'

x_fname = f'src_{dset}.txt'
y_fname = f'{dset}.txt'

x_path = os.path.join(task_dir, x_fname)
y_path = os.path.join(task_dir, y_fname)
y_eval_path = f'metrics/tmp/{task}/measure_scores__{dset}.txt'

x = open(x_path, 'r').readlines()
y = open(y_path, 'r').readlines()

assert len(x) == len(y), 'Your structured inputs (x) and natural-language outputs (y) do not line up 1 to 1.'

y_eval_file = open(y_eval_path, 'a')

last_x_row = ''
for i, (x_row, y_row) in enumerate(zip(x, y)):
    if i > 0 and x_row != last_x_row:
        y_eval_file.write('\n')

    output = y_row.split('<eos>|||')[0].strip()
    y_eval_file.write(output + '\n')

    last_x_row = x_row
