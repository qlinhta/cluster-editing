'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 14/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

import sys
class Union:

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.count = [1 for i in range(n)]

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[a] != root:
            next_ele = self.parent[a]
            self.parent[a] = root
            a = next_ele

        return root

    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.count[a] < self.count[b]:
            a, b = b, a

        self.parent[b] = a
        self.count[a] += self.count[b]
        return True

class File:
    def read_file(self):
        data = iter([line.strip() for line in open('../realInstances/heur113.gr')])
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

    def write_results(self, solution):
        f = open('Output.txt', 'w')
        for sol in solution:
            f.write(sol + '\n')
        f.close()


class Graph:
    def __init__(self, n, edges):
        self._adj = [[] for _ in range(n)]
        self._n = n  # number of vertices
        self._m = 0  # number of edges

        # adding edges
        for e in edges:
            try:
                u, v = e
                self.add_edge(u, v)
            except TypeError:
                pass

        # sorting the edges for faster access
        self.sort_edges()

    def n(self):
        return self._n

    def m(self):
        return self._m

    # add edge between u and v vertex
    def add_edge(self, u, v):
        self._adj[u].append(v)
        self._adj[v].append(u)
        self._m += 1

    # remove edge between u and v vertex
    def remove_edge(self, u, v):
        try:
            self._adj[u].remove(v)
            self._adj[v].remove(u)
            self._m -= 1
        except ValueError:
            pass

    # sort the edges
    def sort_edges(self):
        for vertex in range(self._n):
            self._adj[vertex] = sorted(self._adj[vertex])

    # return the neighbors of u
    def neighbors(self, u):
        return self._adj[u]

    # return the number of neighbors
    def degree(self, u):
        return len(self._adj[u])

    # return check if both vertices are adjacent
    def is_adjacent(self, u, v):
        return 1 if v in self._adj[u] else 0

    # kernelization
    def remove_excess_degree_one(self):
        deleted = False
        delete_decision = [False] * self._n  # maintains the record of items to be deleted
        for vertex_u in range(self._n):
            if self.degree(vertex_u) == 1:
                continue

            # delete all but one neigbor of degree 1 of v
            delete_next = False
            for vertex_v in self._adj[vertex_u]:
                if delete_decision[vertex_v]:
                    continue

                if self.degree(vertex_v) == 1:
                    if delete_next:
                        delete_decision[vertex_v] = True
                        deleted = True
                    else:
                        delete_next = True

        self._m = 0
        for vertex_u in range(self._n):
            if delete_decision[vertex_u]:
                self._adj[vertex_u] = []
            else:
                self._adj[vertex_u] = [neighbor for neighbor in self._adj[vertex_u] if not delete_decision[neighbor]]
            self._m += len(self._adj[vertex_u])

        self._m /= 2
        return deleted

    def disjoint_neighborhoods(self, u, v):
        return self.num_intersecting_neighbors(u, v) == 0

    # ajdacency lists are assumed to be sorted in increasing order
    def num_intersecting_neighbors(self, u, v):
        return len(set(self._adj[u]).intersection(set(self._adj[v])))

    def remove_edge_disjoint_neighbors(self):
        deleted = False
        marked_vertices = [-1] * self._n

        # Mark vertices that have a degree 1 adjacent or two degree 2 that are adjacent, adjacents
        for vertex_u in range(self._n):
            if self.degree(vertex_u) == 1:
                marked_vertices[self._adj[vertex_u][0]] = vertex_u
                continue

            if self.degree(vertex_u) == 2:
                first_neighbor = self._adj[vertex_u][0]
                second_neighbor = self._adj[vertex_u][1]

                if self.degree(first_neighbor) != 2:
                    if self.degree(second_neighbor) == 2:
                        first_neighbor, second_neighbor = second_neighbor, first_neighbor
                    else:
                        continue

                #  vertex_v has degree 2
                first_nested_neighbor = self._adj[first_neighbor][0]
                second_nested_neighbor = self._adj[first_neighbor][1]

                if first_nested_neighbor == second_nested_neighbor:
                    first_nested_neighbor, second_nested_neighbor = second_nested_neighbor, first_nested_neighbor
                #  vertex_x ==vertex_u
                if second_neighbor == second_nested_neighbor:
                    marked_vertices[second_neighbor] = vertex_u

        temp = []

        for vertex_u in range(self._n):
            if marked_vertices[vertex_u] == -1:
                continue

            temp.clear()

            for vertex_v in self._adj[vertex_u]:
                if vertex_v != marked_vertices[vertex_u] and self.disjoint_neighborhoods(vertex_u, vertex_v):
                    vertex_to_deleted = self._lower_bound(self._adj[vertex_v], vertex_u)
                    if vertex_to_deleted:
                        self._adj[vertex_v].remove(vertex_to_deleted)
                        deleted = True
                else:
                    temp.append(vertex_v)

            self._adj[vertex_u] = temp

        self._m = 0
        for vertex_u in range(self._n):
            self._m += self.degree(vertex_u)
        self._m /= 2

        return deleted

    def _lower_bound(self, nums, target):
        if len(nums) > 0:
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] >= target:
                    r = mid - 1
                else:
                    l = mid + 1
            return nums[l - 1]
        return None

    # If 2 degree 2 vertices v,w are
    # adjacent to u,x that are not adjacent,
    # remove two non adjacent edges in this c4.
    def remove_c4(self):
        deleted = False

        for vertex_u in range(self._n):
            for vertex_v in self._adj[vertex_u]:
                if self.degree(vertex_v) != 2:
                    continue

                #  other_neighbor_x is the other neighbor of vertex_v
                other_neighbor_x = self._adj[vertex_v][0]
                if other_neighbor_x == vertex_u:
                    other_neighbor_x = self._adj[vertex_v][1]

                # vertex_u and x must be non-adjacent
                if self.is_adjacent(vertex_u, other_neighbor_x) == 1:
                    continue

                for vertex_w in self._adj[vertex_u]:
                    if self.degree(vertex_w) != 2 or vertex_w == vertex_v:
                        continue

                    other_neighbor_y = self._adj[vertex_w][0]
                    if other_neighbor_y == vertex_u:
                        other_neighbor_y = self._adj[vertex_w][1]

                    if other_neighbor_y != other_neighbor_x:
                        continue

                    self.remove_edge(vertex_u, vertex_v)
                    self.remove_edge(vertex_w, other_neighbor_x)

                    deleted = True
                    break

        self._m = 0
        for vertex_u in range(self._n):
            self._m += self.degree(vertex_u)
        self._m /= 2

        return deleted

    # If 3 degree <= 3 vertices u,v,w form a triangle
    # which is not in any diamond,
    # isolate them.
    def remove_deg3_triangles(self):
        deleted = False
        to_deleted = []

        for vertex_u in range(self._n):

            if self.degree(vertex_u) != 3:
                continue

            to_deleted.clear()

            neighbor_1 = self._adj[vertex_u][0]
            neighbor_2 = self._adj[vertex_u][1]
            neighbor_3 = self._adj[vertex_u][2]

            min_degree = min(self.degree(neighbor_1), self.degree(neighbor_2), self.degree(neighbor_3))
            if min_degree > 3:  # we need at least a degree 2 or 3 in neighbors:
                continue

            nb = self.is_adjacent(neighbor_1, neighbor_2) + \
                 self.is_adjacent(neighbor_2, neighbor_3) + \
                 self.is_adjacent(neighbor_3, neighbor_1)
            if nb == 3:  # K_4
                if self.degree(neighbor_2) == 3:
                    neighbor_1, neighbor_2 = neighbor_2, neighbor_1

                if self.degree(neighbor_3) == 3:
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                # neighbor_1 has degree 3
                if self.degree(neighbor_2) <= 5 and self.degree(neighbor_3) <= 5:
                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_2, vertex_x))

                    for vertex_x in self.neighbors(neighbor_3):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_3, vertex_x))

            elif nb == 2:  # Diamond
                if self.degree(neighbor_1) <= 3 and self.degree(neighbor_2) and self.degree(neighbor_3) <= 3:
                    if self.is_adjacent(neighbor_1, neighbor_2):
                        neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                    if self.is_adjacent(neighbor_1, neighbor_2):
                        neighbor_2, neighbor_3 = neighbor_3, neighbor_2
                    # neighbor_1 and neighbor_2 are not adjacent

                    if self.num_intersecting_neighbors(neighbor_1, neighbor_2) == 2:  # not a diamond
                        for vertex_x in self.neighbors(neighbor_1):
                            if vertex_x != neighbor_3 and vertex_x != vertex_u:
                                to_deleted.append((neighbor_1, vertex_x))

                        for vertex_x in self.neighbors(neighbor_2):
                            if vertex_x != neighbor_3 and vertex_x != vertex_u:
                                to_deleted.append((neighbor_2, vertex_x))

            elif nb == 1:  # Triangle
                if self.is_adjacent(neighbor_1, neighbor_3):
                    neighbor_2, neighbor_3 = neighbor_3, neighbor_2

                if self.is_adjacent(neighbor_2, neighbor_3):
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1
                # neighbor_1 and neighbor_2 are adjacent, the others are not

                if self.degree(neighbor_1) <= 3 and \
                        self.degree(neighbor_2) <= 3 and \
                        self.num_intersecting_neighbors(neighbor_1, neighbor_2) == 1:  # not a diamond

                    to_deleted.append((vertex_u, neighbor_3))

                    for vertex_x in self.neighbors(neighbor_1):
                        if vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_1, vertex_x))

                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_2, vertex_x))

            for x, y in to_deleted:
                self.remove_edge(x, y)
                deleted = True

        return deleted

    # If 3 degree <= 3 vertices u,v,w form a triangle
    # which is not in any diamond,
    # isolate them
    def isolate_small_complete(self, s):
        deleted = False
        out_degrees = []
        outer = []

        for vertex_u in range(self._n):
            if self.degree(vertex_u) != s - 1:
                continue

            a = 0
            for vertex_v in self._adj[vertex_u]:
                for vertex_w in self._adj[vertex_u]:
                    if vertex_v != vertex_w and self.is_adjacent(vertex_v, vertex_w) == 1:
                        a += 1
            # If neighborhood is not a clique, continue
            if a != (s - 1) * (s - 2):
                continue

            out_degrees.clear()
            out_degrees.append(self.degree(vertex_u) - s + 1)

            for vertex_v in self._adj[vertex_u]:
                out_degrees.append(self.degree(vertex_v) - s + 1)

            out_degrees = sorted(out_degrees)

            acc = 0
            acc_degrees = 0

            valid = True
            for i in range(s):
                acc += s - i - 1
                acc_degrees += out_degrees[s - i - 1]
                if acc_degrees > acc:
                    valid = False
                    break

            if not valid:
                continue

            outer.clear()
            for vertex_v in self._adj[vertex_u]:
                for vertex_w in self._adj[vertex_v]:
                    if vertex_w != vertex_u and self.is_adjacent(vertex_w, vertex_u) == 0:
                        outer.append(vertex_w)

            outer = sorted(outer)
            outer_size = len(outer)

            for i in range(outer_size - 1):
                if outer[i] == outer[i + 1]:
                    valid = False

            if not valid:
                continue

            for vertex_v in self._adj[vertex_u]:
                candidates = self._adj[vertex_v]
                for vertex_x in candidates:
                    if vertex_x != vertex_u and self.is_adjacent(vertex_x, vertex_u) == 0:
                        if self.is_adjacent(vertex_x, vertex_v) == 1:
                            deleted = True
                            self.remove_edge(vertex_x, vertex_v)
        return deleted

    def kernelize(self):
        m_removed = self._m
        i = 0
        while True:
            cont = False

            while self.remove_excess_degree_one():
                cont = True

            while self.remove_edge_disjoint_neighbors():
                cont = True

            while self.remove_c4():
                cont = True

            while self.remove_deg3_triangles():
                cont = True

            for s in range(3, 11):
                while self.isolate_small_complete(s):
                    cont = True

            i += 1

            if not cont:
                break

        m_removed -= self._m
        return m_removed

    def connected_components(self):
        uf = Union(self._n)

        for vertex_u in range(self._n):
            for vertex_v in self._adj[vertex_u]:
                uf.merge(vertex_u, vertex_v)

        cc_count = 0
        cc_id_aux = [0 for i in range(self._n)]
        for vertex_u in range(self._n):
            if uf.parent[vertex_u] == vertex_u:
                cc_id_aux[vertex_u] = cc_count
                cc_count += 1
        ccs = [[] for i in range(cc_count)]
        for vertex_u in range(self._n):
            ccs[cc_id_aux[uf.find(vertex_u)]].append(vertex_u)

        return ccs


class Instance:
    def __init__(self, n, edges):
        self._g = Graph(n, edges)
        self._cluster_of = [0] * self._g.n()
        self._cluster_size = [0] * self._g.n()
        self.candidate_clusters_adj = [0] * self._g.n()
        self.n = self._g.n()

        self._cluster_size[0] = self.n
        self._zero_size_cluster = []
        for i in range(self.n):
            self._zero_size_cluster.append(i)

        self._cost = (self.n * (self.n - 1)) / 2 - self._g.m()

    def reinit_all_zero(self):
        self._cluster_size = [0] * len(self._cluster_size)
        self._cluster_of = [0] * len(self._cluster_of)
        self._cluster_size[0] = self.n

        self._zero_size_cluster.clear()
        for i in range(self.n):
            self._zero_size_cluster.append(i)
        self._cost = (self.n * (self.n - 1)) / 2 - self._g.m()

    def reinit_state(self, v, cost):
        for i in range(self.n):
            self._cluster_size[i] = 0

        for i in range(self.n):
            self._cluster_of[i] = v[i]
            self._cluster_size[v[i]] += 1

        self._cost = cost
        self._zero_size_cluster.clear()
        for i in range(self.n):
            if self._cluster_size[i] == 0:
                self._zero_size_cluster.append(i)

    def get_zero_size(self):
        res = self._zero_size_cluster[-1]

        while self._cluster_size[res] != 0 and len(self._zero_size_cluster) != 0:
            self._zero_size_cluster.pop()
            res = self._zero_size_cluster[-1]

        return res

    def _move(self, v, c):
        cv = self._cluster_of[v]
        if cv == c: return
        self._cluster_of[v] = c
        self._cluster_size[cv] -= 1
        self._cluster_size[c] += 1

        if self._cluster_size[cv] == 0:
            self._zero_size_cluster.append(cv)

    def greedy_move(self, v):
        cv = self._cluster_of[v]

        for u in self._g.neighbors(v):
            cu = self._cluster_of[u]
            self.candidate_clusters_adj[cu] += 1

        self_edges = self.candidate_clusters_adj[cv]
        best_cost = 0
        best_cluster = -1
        self_cost = -(self._cluster_size[cv] - 1 - 2 * self_edges)

        for u in self._g.neighbors(v):
            cu = self._cluster_of[u]
            if cu == cv:
                continue

            cost = (self._cluster_size[cu] - 2 * self.candidate_clusters_adj[cu]) + self_cost

            if cost < best_cost:
                best_cost = cost
                best_cluster = cu

        for u in self._g.neighbors(v):
            cu = self._cluster_of[u]
            self.candidate_clusters_adj[cu] = 0

        if self_cost < best_cost and len(self._zero_size_cluster) != 0:
            best_cost = self_cost
            best_cluster = self.get_zero_size()

        if best_cluster == -1:
            return False

        self._move(v, best_cluster)
        self._cost += best_cost

        return True

    def delta_cost(self, v, c):
        cv = self._cluster_of[v]
        if cv == c:
            return 0

        self_edges = 0
        to_edges = 0

        for u in self._g.neighbors(v):
            if self._cluster_of[u] == cv:
                self_edges += 1

            if self._cluster_of[u] == c:
                to_edges += 1

        return (self._cluster_size[c] - 2 * to_edges) - (self._cluster_size[cv] - 1 - 2 * self_edges)

    def move_with_delta(self, v, c, delta):
        self._move(v, c)
        self._cost += delta

    def move_to_zero_size(self, vs):
        for v in vs:
            if self._cluster_size[self._cluster_of[v]] == 1:
                continue
            c = self.get_zero_size()
            self.move_with_delta(v, c, self.delta_cost(v, c))

    def move_to_same_zero_size(self, vs):
        first_alone = self._cluster_size[self._cluster_of[vs[0]]] == 1
        c = first_alone if self._cluster_of[vs[0]] else self.get_zero_size()

        for i in range(first_alone, len(vs)):
            v = vs[i]
            self.move_with_delta(v, c, self.delta_cost(v, c))

    def revert_cluster_of_with_cost(self, vs, old_cluster_of_vs, old_cost):
        for i in range(len(vs)):
            v = vs[i]
            c = old_cluster_of_vs[i]
            self._move(v, c)

        self._cost = old_cost

    def destroy_greedy_repair(self, vs, same_zero_size):
        if same_zero_size:
            self.move_to_same_zero_size(vs)
        else:
            self.move_to_zero_size(vs)

        for v in vs:
            self.greedy_move(v)

    def bfs_fill_vs(self, v, nv, vs, cluster_of_vs, seen):
        q = [v]
        while len(q) != 0 and len(vs) < nv:
            v = q.pop(0)
            if seen[v]:
                continue
            seen[v] = True
            vs.append(v)
            cluster_of_vs.append(self._cluster_of[v])
            q.extend(self._g.neighbors(v))

    def m(self):
        return self._g.m()

    def sol(self):
        return self._cluster_of

    def cost(self):
        return self._cost


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
                                sys.stdout.write(f'{u + 1} {v + 1}\n')

        # Compute edges to delete
        for e in self._initial_edges:
            u, v = e
            cc_u, id_u = self._vertex_to_cc[u]
            cc_v, id_v = self._vertex_to_cc[v]
            if cc_u != cc_v or (cc_u == cc_v and self._solutions[cc_u][id_u] != self._solutions[cc_v][id_v]):
                sys.stdout.write(f'{u + 1} {v + 1}\n')


import random


def bfs_destroy_repair_ls_one_it_order_norand(g, n, ndestroy, order, seen, best_sol, best_cost, same_zero_size):
    vs = []
    old_cluster_of_vs = []

    seen = [False] * len(seen)
    random.shuffle(order)

    res = False

    for i in range(n):
        if seen[order[i]]:
            continue

        vs.clear()
        old_cluster_of_vs.clear()
        g.bfs_fill_vs(order[i], ndestroy, vs, old_cluster_of_vs, seen)

        if len(vs) == 0:
            continue

        random.shuffle(vs)
        for j in range(len(vs)):
            old_cluster_of_vs[j] = g.sol()[vs[j]]

        old_cost = g.cost()
        g.destroy_greedy_repair(vs, same_zero_size)

        if g.cost() < best_cost:
            res = True
            best_cost = g.cost()
            best_sol = g.sol()

        if g.cost() > old_cost:
            g.revert_cluster_of_with_cost(vs, old_cluster_of_vs, old_cost)

    return res


import time

if __name__ == '__main__':
    opt = [i for i in range(5, 55)]
    weight = 10
    iterations = 30

    start_time = time.time()
    # start processing
    f = File()
    number, edges, edges_exists = f.read_file()
    instance = KernelizedMultiCCInstance(number, edges, edges_exists)

    # running the greedy BFS solution
    instance.greedy_bfs_fill(opt, weight, iterations)
    instance.print_sol()
    print("--- Total Time %s seconds ---" % (time.time() - start_time))