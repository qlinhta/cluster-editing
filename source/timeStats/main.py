import csv
import time
import sys
from os import listdir
from os.path import isfile, join

from kernelizedMultiCCInstance import KernelizedMultiCCInstance
from fileHandler import File

if __name__ == '__main__':
    opt = [i for i in range(5, 55)]
    weight = 10
    iterations = 30

    onlyfiles = [f for f in listdir('files/') if isfile(join('files/', f))]
    data = []
    for file in onlyfiles:
        file_data = {}
        file_data['File name'] = file
        start_time = time.time()
        # start processing
        f = File()
        number, edges, edges_exists = f.read_file(file)
        instance = KernelizedMultiCCInstance(number, edges, edges_exists)

        # running the greedy BFS solution
        instance.greedy_bfs_fill(opt, weight, iterations)
        modified = instance.print_sol()
        file_data['Edges modified'] = modified
        time_taken = time.time() - start_time
        file_data['Execution time'] = time_taken
        data.append(file_data)
        print(data)

    print(data)

    field_names = ['File name', 'Edges modified', 'Execution time']
    with open('stats.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)