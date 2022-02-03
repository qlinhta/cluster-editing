import time
import sys

from kernelizedMultiCCInstance import KernelizedMultiCCInstance
from fileHandler import File

if __name__ == '__main__':
    opt = [i for i in range(5, 55)]
    weight = 10
    iterations = 1

    start_time = time.time()
    # start processing
    f = File()
    number, edges = f.read_file()
    instance = KernelizedMultiCCInstance(number, edges)

    # running the greedy BFS solution
    instance.greedy_bfs_fill(opt, weight, iterations)
    sol = instance.get_sol()

    # output to console
    for s in sol:
        sys.stdout.write(s + "\n")

    # writing a solution in a file
    f.write_results(sol)

    print("--- Total Time %s seconds ---" % (time.time() - start_time))
