import random
import sys

from instance import Instance
from graph import Graph
from localSearch import bfs_destroy_repair_ls_one_it_order_norand


# Class handle an instance and divide in multiple connected components
# the time complexity of this class is O(m) where m is the number of edges
class KernelizedMultiCCInstance:
    def __init__(self, n, edges, edges_exists):
        self._vertex_to_cc = [None for _ in range(n)]  # hold pair (x,y)
        self._initial_edges = edges
        self._edges_exists = edges_exists
        self._initial_edges = sorted(self._initial_edges)  # sorted edges to get data faster

        # initializing the graph object
        self.g = Graph(n, self._initial_edges)
        self.g.kernelize()
        # Compute cc info
        self._ccs = self.g.connected_components()

        # Sort them by decreasing size
        self._ccs = sorted(self._ccs, key=len, reverse=True)
        self._n_cc = len(self._ccs)

        # adding values in
        for i in range(self._n_cc):
            for j in range(len(self._ccs[i])):
                self._vertex_to_cc[self._ccs[i][j]] = (i, j)

        # build a list of edges and an instance graph for each cc
        self._cc_edges = [[] for _ in range(self._n_cc)]  # list contains the list of pairs
        for (u, v) in self._initial_edges:
            cc_u, id_u = self._vertex_to_cc[u]
            cc_v, id_v = self._vertex_to_cc[v]
            if cc_u == cc_v:
                self._cc_edges[cc_u].append((id_u, id_v))

        self._cc_instances = []
        for i in range(self._n_cc):
            self._cc_instances.append(Instance(len(self._ccs[i]), self._cc_edges[i]))

        self._solutions = []
        self._costs = []
        self._total_cost = 0

        for instance in self._cc_instances:
            self._solutions.append(instance.sol())
            self._costs.append(instance.cost())
            self._total_cost += instance.cost()

    def cost(self):
        return self._total_cost

    def m(self):
        return len(self._initial_edges)

    def greedy_bfs_fill(self, n_destroy_options, default_weight, iterations):
        n_cc = len(self._cc_instances)

        seen = []  # nested list contains the seen boolean information
        for cc in self._cc_instances:
            seen.append([False] * cc.n)

        # to store the order of cc
        orders = [None] * self._n_cc
        for i in range(n_cc):
            cc = self._cc_instances[i]
            orders[i] = [None for _ in range(cc.n)]
            for j in range(cc.n):
                orders[i][j] = j

        interesting = [True] * n_cc
        # first, mark all ccs that we should not look at.
        for i in range(n_cc):
            cc = self._cc_instances[i]
            n = cc.n
            m = cc.m()
            if n <= 3:
                break

            if m == ((n * (n - 1)) / 2):
                interesting[i] = False

        # introduction of randomness
        nd_weights = [[default_weight for _ in range(len(n_destroy_options))]] * self._n_cc
        last_improv = [0] * self._n_cc

        # runs as the number of iterations
        for _ in range(iterations):  # iterate until killed
            for i in range(self._n_cc):
                if not interesting[i]:
                    continue

                cc = self._cc_instances[i]
                n = cc.n

                # Stop iteration when we reach small ccs
                if n <= 3:
                    break

                self._total_cost -= self._costs[i]
                # selecting random number from n_destory_options
                nd_sel = random.choices(range(len(n_destroy_options)), weights=nd_weights[i])[0]

                same_zero_size = cc.m() > 3 * cc.n
                # calling local search
                improv = bfs_destroy_repair_ls_one_it_order_norand(
                    cc, n, n_destroy_options[nd_sel],
                    orders[i], seen[i], self._solutions[i], self._costs[i], same_zero_size)

                # check if improvement then change in weights accordingly
                if improv:
                    nd_weights[i][nd_sel] += 1
                    random.shuffle(nd_weights[i])
                    last_improv[i] = 0
                elif (last_improv[i] + 1) * 3 > cc.n:
                    last_improv[i] = 0
                    cc.reinit_all_zero()

                if self._costs[i] <= 1:
                    interesting[i] = False

        for i in range(self._n_cc):
            self._cc_instances[i].reinit_state(self._solutions[i], self._costs[i])

    def count_sol(self):
        res = 0
        n_cc = len(self._cc_instances)

        for i in range(n_cc):
            n = self._cc_instances[i].n

            cluster_of = self._solutions[i]
            clusters = [range(n)]

            for u in range(n):
                clusters[cluster_of[u]] = self._ccs[i][u]

            for c in clusters:
                for x in c:
                    for y in c:
                        u = self._ccs[i][x]
                        v = self._ccs[i][y]
                        if u < v:
                            if (u, v) in self._initial_edges:
                                res += 1

        for e in self._initial_edges:
            u, v = e
            cc_u, id_u = self._vertex_to_cc[u]
            cc_v, id_v = self._vertex_to_cc[v]
            if cc_u != cc_v or (cc_u == cc_v and self._solutions[cc_u][id_u] != self._solutions[cc_v][id_v]):
                res += 1

        return res

    def print_sol(self):
        modified = 0
        n_cc = len(self._cc_instances)

        # Concatenate clusters of each instance
        for i in range(n_cc):
            n = self._cc_instances[i].n
            cluster_of = self._solutions[i]

            clusters = [[] for _ in range(n)]

            for u in range(n):
                clusters[cluster_of[u]].append(u)

            clusters = [x for x in clusters if x is not None]

            for c in clusters:
                for x in c:
                    for y in c:
                        if x != y:
                            u = self._ccs[i][x]
                            v = self._ccs[i][y]
                            if u < v and self._edges_exists.get((u, v)) is None:
                                modified += 1

        # Compute edges to delete
        for e in self._initial_edges:
            u, v = e
            cc_u, id_u = self._vertex_to_cc[u]
            cc_v, id_v = self._vertex_to_cc[v]
            if cc_u != cc_v or (cc_u == cc_v and self._solutions[cc_u][id_u] != self._solutions[cc_v][id_v]):
                modified += 1

        return modified
