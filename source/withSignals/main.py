import time
import sys

from SIGTERM import Killer
from kernelizedMultiCCInstance import KernelizedMultiCCInstance
from fileHandler import File

if __name__ == '__main__':
    opt = [i for i in range(5, 55)]
    weight = 10
    iterations = 30 #best
    killer = Killer()
    start_time = time.time()
    # start processing
    f = File()
    number, edges, edges_exists = f.read_file()
    instance = KernelizedMultiCCInstance(number, edges, edges_exists)

    # running the greedy BFS solution
    instance.greedy_bfs_fill(opt, weight, killer)
    instance.print_sol()

    print("--- Total Time %s seconds ---" % (time.time() - start_time))
