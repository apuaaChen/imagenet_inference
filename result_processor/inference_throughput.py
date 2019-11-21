import json
import numpy as np
import csv


model_list = ['resnet50', 'mobilenetv2']
batch_size = [256, 128, 64, 32, 16, 8, 4, 2, 1]

logdir = '../json/'


def wash(durations):
    new_durations = []
    while True:
        mean = np.mean(durations)
        std = np.std(durations)
        for d in durations:
            if mean - std * 3. < d < mean + std * 3.:
                new_durations.append(d)
        if len(new_durations) == len(durations):
            break
        else:
            durations = new_durations
            new_durations = []
    return new_durations


header = ['models']
metric = ['']
for bs in batch_size:
    header.append('%d' % bs)
    header.append('')
    metric.append('mean')
    metric.append('std')

rows = [header, metric]


for m in model_list:
    results = [m]
    for bs in batch_size:
        with open(logdir + m + '_%d/result.json' % bs) as json_file:
            log = json.load(json_file)
        durations = log['durations']
        washed_durations = wash(durations)
        throughput = np.divide(float(bs), washed_durations)
        mean = np.mean(throughput)
        std = np.std(throughput)
        results.append('%.4f' % mean)
        results.append('%.4f' % std)

    rows.append(results)

with open('./inference_time.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for r in rows:
        writer.writerow(r)
