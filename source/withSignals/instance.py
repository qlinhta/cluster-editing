'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 14/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

from graph import Graph
import queue


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
