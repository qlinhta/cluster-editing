'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 29/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

'''IN CASE MY PROGRAM DO NOT GO WELL WITH SIGTERM SIGNAL, PLEASE EXECUTE all_in_one_for_optil.py TO CONSIDER MY RESULTS
'''

'''
This code was developed by Quyen Linh TA
() Section KernelizedMultipleCC: Developed with advices of Dr. Noah Vogel | Sr.Data Scientist | Microsoft AI Lab
() Section Instance: Developed with instructions & advices of George H. Smith & Dr. Dale T. Burney | Microsoft AI Lab
() Reference algorithms and support theory by Pr. Robyn K. Martinez, Cornell University
'''


import time

from KernelizationMultipleCC import KernelizationMultipleCC
from InputDIMACS import InputFiles

import signal


class Killer:
    exit_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        self.exit_now = True

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
    instance.greedy_bfs_fill(opt, weight, killer)
    instance.print_sol()

    print("--- Total Time %s seconds ---" % (time.time() - start_time))
