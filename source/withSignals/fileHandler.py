import sys

class File:
    def read_file(self):
        try:
            data = [line.strip() for line in sys.stdin]
        except FileNotFoundError:
            raise Exception("File not found Please add correct file path")

        number_of_vertices = int(data[0].split()[2])
        edges = []

        for d in data[1:]:
            u, v = [int(vertex) for vertex in d.split()]
            edges.append((min(u - 1, v - 1), max(u - 1, v - 1)))

        return number_of_vertices, edges

    def write_results(self, solution):
        f = open('Output.txt', 'w')
        for sol in solution:
            f.write(sol + '\n')
        f.close()

