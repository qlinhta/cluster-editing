'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 16/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''


class Union:
    def __init__(self, n): # n is number of vertices
        self.parent = [i for i in range(n)] # parent of each vertex
        self.count = [1 for i in range(n)] # number of vertices in each cluster

    def find(self, a): # find the root of a
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


class Graph:
    def __init__(self, n, edges): # n is number of vertices, edges is list of edges
        self._adj = [[] for _ in range(n)] # adjacency list
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

    def n(self): # return number of vertices
        return self._n

    def m(self): # return number of edges
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

    def degree(self, u):
        return len(self._adj[u])  # return the number of neighbors

    def adjacent(self, u, v):
        return 1 if v in self._adj[u] else 0  # return check if both vertices are adjacent

    # kernelization
    def connected_components(self): # return the number of connected components
        p = Union(self._n) # create a union-find data structure

        for u in range(self._n):
            '''
            For each vertex u, we iterate through all its neighbors v and merge u and v into the same cluster.
            Because we are using Union-Find data structure, we can find the root of u and v in O(1) time.
            '''
            for v in self._adj[u]: # O(d(u))
                p.merge(u, v) # O(1)

        countConnectedComponents = 0 # number of connected components
        connectedComponentsID = [0 for i in range(self._n)] # the id of connected component of each vertex
        for u in range(self._n): # O(n)
            if p.parent[u] == u: # if u is the root of its cluster
                connectedComponentsID[u] = countConnectedComponents # assign the id of connected component for u
                countConnectedComponents += 1 # increase the number of connected components
        ccs = [[] for i in range(countConnectedComponents)] # list of connected components
        for u in range(self._n): # O(n)
            ccs[connectedComponentsID[p.find(u)]].append(u) # add u to the connected component of u

        return ccs # return the list of connected components

    def deleteExcessDegreeOne(self): # delete excess degree one vertices
        deletionCheck = False
        decision = [False] * self._n  # decision[i] = True if vertex i is deleted else False
        for vertex_u in range(self._n):
            if self.degree(vertex_u) == 1: # if vertex_u has degree 1
                continue
            willDelete = False
            for vertex_v in self._adj[vertex_u]: # O(d(u))
                if decision[vertex_v]: # if vertex_v is deleted then continue
                    continue

                if self.degree(vertex_v) == 1: # if vertex_v has degree 1
                    if willDelete:
                        decision[vertex_v] = True
                        deletionCheck = True
                    else:
                        willDelete = True
        self._m = 0
        for vertex_u in range(self._n):
            if decision[vertex_u]:
                self._adj[vertex_u] = []
            else:
                self._adj[vertex_u] = [neighbor for neighbor in self._adj[vertex_u] if not decision[neighbor]]
            self._m += len(self._adj[vertex_u])

        self._m /= 2
        return deletionCheck

    def intersectionNeighbors(self, u, v): # return the number of common neighbors of u and v
        return len(set(self._adj[u]).intersection(set(self._adj[v])))

    def disjointNeighbors(self, u, v): # return the number of disjoint neighbors of u and v
        return self.intersectionNeighbors(u, v) == 0

    def deleteEdgesDisjointNeig(self): # delete edges between disjoint neighbors of u and v
        deletionCheck = False
        markedVertex = [-1] * self._n

        # Mark vertices that have a degree 1 adjacent or two degree 2 that are adjacent, adjacents
        for u in range(self._n):
            if self.degree(u) == 1:
                markedVertex[self._adj[u][0]] = u
                continue

            if self.degree(u) == 2:
                first_neighbor = self._adj[u][0]
                second_neighbor = self._adj[u][1]

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
                    markedVertex[second_neighbor] = u

        temp = []

        for u in range(self._n):
            if markedVertex[u] == -1:
                continue

            temp.clear()

            for v in self._adj[u]:
                if v != markedVertex[u] and self.disjointNeighbors(u, v):
                    vertex_to_deleted = self.lower_bound(self._adj[v], u)
                    if vertex_to_deleted:
                        self._adj[v].remove(vertex_to_deleted)
                        deletionCheck = True
                else:
                    temp.append(v)

            self._adj[u] = temp

        self._m = 0
        for u in range(self._n):
            self._m += self.degree(u)
        self._m /= 2

        return deletionCheck

    def lower_bound(self, nums, target):
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

    def deleteC4(self):
        deletionCheck = False

        for u in range(self._n):
            for v in self._adj[u]:
                if self.degree(v) != 2:
                    continue

                #  other_neighbor_x is the other neighbor of v
                other_neighbor_x = self._adj[v][0]
                if other_neighbor_x == u:
                    other_neighbor_x = self._adj[v][1]

                # u and x must be non-adjacent
                if self.adjacent(u, other_neighbor_x) == 1:
                    continue

                for w in self._adj[u]:
                    if self.degree(w) != 2 or w == v:
                        continue

                    other_neighbor_y = self._adj[w][0]
                    if other_neighbor_y == u:
                        other_neighbor_y = self._adj[w][1]

                    if other_neighbor_y != other_neighbor_x:
                        continue

                    self.remove_edge(u, v)
                    self.remove_edge(w, other_neighbor_x)

                    deletionCheck = True
                    break

        self._m = 0
        for u in range(self._n):
            self._m += self.degree(u)
        self._m /= 2

        return deletionCheck

    def deleteDegThreeTriangles(self):
        deletionCheck = False
        to_deleted = []

        for u in range(self._n):

            if self.degree(u) != 3:
                continue

            to_deleted.clear()

            neighbor_1 = self._adj[u][0]
            neighbor_2 = self._adj[u][1]
            neighbor_3 = self._adj[u][2]

            min_degree = min(self.degree(neighbor_1), self.degree(neighbor_2), self.degree(neighbor_3))
            if min_degree > 3:  # we need at least a degree 2 or 3 in neighbors:
                continue

            nb = self.adjacent(neighbor_1, neighbor_2) + \
                 self.adjacent(neighbor_2, neighbor_3) + \
                 self.adjacent(neighbor_3, neighbor_1)
            if nb == 3:  # K_4
                if self.degree(neighbor_2) == 3:
                    neighbor_1, neighbor_2 = neighbor_2, neighbor_1

                if self.degree(neighbor_3) == 3:
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                # neighbor_1 has degree 3
                if self.degree(neighbor_2) <= 5 and self.degree(neighbor_3) <= 5:
                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != u:
                            to_deleted.append((neighbor_2, vertex_x))

                    for vertex_x in self.neighbors(neighbor_3):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != u:
                            to_deleted.append((neighbor_3, vertex_x))

            elif nb == 2:  # Diamond
                if self.degree(neighbor_1) <= 3 and self.degree(neighbor_2) and self.degree(neighbor_3) <= 3:
                    if self.adjacent(neighbor_1, neighbor_2):
                        neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                    if self.adjacent(neighbor_1, neighbor_2):
                        neighbor_2, neighbor_3 = neighbor_3, neighbor_2
                    # neighbor_1 and neighbor_2 are not adjacent

                    if self.intersectionNeighbors(neighbor_1, neighbor_2) == 2:  # not a diamond
                        for vertex_x in self.neighbors(neighbor_1):
                            if vertex_x != neighbor_3 and vertex_x != u:
                                to_deleted.append((neighbor_1, vertex_x))

                        for vertex_x in self.neighbors(neighbor_2):
                            if vertex_x != neighbor_3 and vertex_x != u:
                                to_deleted.append((neighbor_2, vertex_x))

            elif nb == 1:  # Triangle
                if self.adjacent(neighbor_1, neighbor_3):
                    neighbor_2, neighbor_3 = neighbor_3, neighbor_2

                if self.adjacent(neighbor_2, neighbor_3):
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1
                # neighbor_1 and neighbor_2 are adjacent, the others are not

                if self.degree(neighbor_1) <= 3 and \
                        self.degree(neighbor_2) <= 3 and \
                        self.intersectionNeighbors(neighbor_1, neighbor_2) == 1:  # not a diamond

                    to_deleted.append((u, neighbor_3))

                    for vertex_x in self.neighbors(neighbor_1):
                        if vertex_x != neighbor_2 and vertex_x != u:
                            to_deleted.append((neighbor_1, vertex_x))

                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != u:
                            to_deleted.append((neighbor_2, vertex_x))

            for x, y in to_deleted:
                self.remove_edge(x, y)
                deletionCheck = True

        return deletionCheck

    def smallCompleteIsolation(self, s):
        deletionCheck = False
        out_degrees = []
        outer = []

        for u in range(self._n):
            if self.degree(u) != s - 1:
                continue

            a = 0
            for vertex_v in self._adj[u]:
                for vertex_w in self._adj[u]:
                    if vertex_v != vertex_w and self.adjacent(vertex_v, vertex_w) == 1:
                        a += 1
            # If neighborhood is not a clique, continue
            if a != (s - 1) * (s - 2):
                continue

            out_degrees.clear()
            out_degrees.append(self.degree(u) - s + 1)

            for vertex_v in self._adj[u]:
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
            for vertex_v in self._adj[u]:
                for vertex_w in self._adj[vertex_v]:
                    if vertex_w != u and self.adjacent(vertex_w, u) == 0:
                        outer.append(vertex_w)

            outer = sorted(outer)
            outer_size = len(outer)

            for i in range(outer_size - 1):
                if outer[i] == outer[i + 1]:
                    valid = False

            if not valid:
                continue

            for vertex_v in self._adj[u]:
                candidates = self._adj[vertex_v]
                for vertex_x in candidates:
                    if vertex_x != u and self.adjacent(vertex_x, u) == 0:
                        if self.adjacent(vertex_x, vertex_v) == 1:
                            deletionCheck = True
                            self.remove_edge(vertex_x, vertex_v)
        return deletionCheck

    def kernelize(self):
        m_removed = self._m
        i = 0
        while True:
            cont = False

            while self.deleteExcessDegreeOne():
                cont = True

            while self.deleteEdgesDisjointNeig():
                cont = True

            while self.deleteC4():
                cont = True

            while self.deleteDegThreeTriangles():
                cont = True

            for s in range(3, 11):
                while self.smallCompleteIsolation(s):
                    cont = True

            i += 1

            if not cont:
                break

        m_removed -= self._m
        return m_removed
