import os
import sys
import time
import subprocess
from tqdm import tqdm

seeds = {}

seed = 1
for items in [1, 3, 5, 10]:
    for ty in ['bounded-strongly-corr', 'uncorr-similar-weights', 'uncorr']:
        for cap in range(1, 11):
            seeds[(items, ty, cap)] = seed
            seed += 1

results_dir = '/home/caios/repos/flns-ttp/results/'

results_raw = os.listdir(results_dir)
results_raw = [f for f in results_raw if f.endswith('.flns.log')]

results = {}
for f in results_raw:
    instance = f.split('.')[0]
    if instance not in results:
        results[instance] = []
    results[instance].append(f)

instances_folder = '/home/caios/repos/flns-ttp/instances/'
instances_dir = os.listdir(instances_folder)

execute_list = []

for instance in instances_dir:
    if not instance.endswith('.ttp'):
        continue

    instance = instance[:-4]
    instance_params = instance.split('_')

    tsp = instance_params[0]
    nodes = int(''.join([x for x in instance_params[0] if x.isdigit()]))
    items = int(''.join([x for x in instance_params[1] if x.isdigit()]))
    items = items // (nodes - 1)
    ty = instance_params[2]
    cap = int(instance_params[3])
    base_seed = seeds[(items, ty, cap)]

    found_seeds = []
    for result in results[instance]:
        with open(results_dir + result, 'r') as file:
            seed = 0
            for line in file:
                if line.startswith('seed:'):
                    seed = int(line[6:])
                    break
            found_seeds.append(seed)

    for execution in range(5):
        seed = 120 * execution + base_seed
        if seed not in found_seeds:
            execute_list.append((instance, seed))

num_cpus = int(sys.argv[1])
tasks = []
available_pbars = [tqdm(total=600, position=pos) for pos in range(num_cpus)]

while execute_list:
    while len(tasks) >= num_cpus:
        time.sleep(1)
        for _, pbar in tasks:
            pbar.update(1)

        finished = []
        for task in tasks:
            task[0].poll()
            if task[0].returncode is not None:
                finished.append(task)

        for task in finished:
            task[1].refresh()
            available_pbars.append(task[1])
            tasks.remove(task)

    instance, seed = execute_list.pop()

    pbar = available_pbars.pop()
    pbar.reset()
    desc = f'{seed}/{instance}'
    pbar.set_description(f'{desc:<50}')

    task = subprocess.Popen(['./target/release/flns', '--silent', '--seed', str(seed), '-lll', '-o', 'results',
                             instances_folder + instance + '.ttp'])
    tasks.append((task, pbar))
