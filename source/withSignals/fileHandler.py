'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 14/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

import sys


class File:
    def read_file(self):
        data = iter([line.strip() for line in sys.stdin])
        first_item = next(data)
        number_of_vertices = int(first_item.split()[2])
        edges = []
        edges_exists = {}
        for d in data:
            u, v = [int(vertex) for vertex in d.split()]
            u, v = (min(u - 1, v - 1), max(u - 1, v - 1))
            edges.append((u, v))
            edges_exists[(u, v)] = True

        return number_of_vertices, edges, edges_exists
