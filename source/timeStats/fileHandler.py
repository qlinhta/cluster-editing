import sys


class File:
    def read_file(self, file):
        data = iter([line.strip() for line in open('files/' + file)])
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
