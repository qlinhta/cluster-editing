'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 29/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

import time
import sys

from SIGTERM import Killer
from KernelizationMultipleCC import KernelizationMultipleCC
from InputDIMACS import InputFiles

if __name__ == '__main__':
    opt = [i for i in range(5, 55)]
    weight = 10
    iterations = 30 #best
    killer = Killer()
    start_time = time.time()
    # start processing
    f = InputFiles()
    number, edges, edges_exists = f.read_file()
    instance = KernelizationMultipleCC(number, edges, edges_exists)

    # running the greedy BFS solution
    instance.greedy_bfs_fill(opt, weight, killer)
    instance.print_sol()

    print("--- Total Time %s seconds ---" % (time.time() - start_time))
