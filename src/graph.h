//
// Created by qlinhta on 03/01/2022.
//

#ifndef PROJET_2021_22_QLINHTA_GRAPH_H
#define PROJET_2021_22_QLINHTA_GRAPH_H
#include "bits/stdc++.h"

using namespace std;

class Graph {

public:
    Graph();
    map<int,unordered_set<int>> graph;
    int vertices;
    int edges;
    void printGraph();
    unordered_set<int> operator [](int i); // Opérateur [] - Renvoie le unordered_set lié à une certaine clé
    void edgeAddition(int x, int y);
    void edgeCut(int x, int y);
    void cutAllEdge(unordered_set<int> cutEdgeSet);
    bool edgeExist(int x, int y);
    bool cliqueCheck(std::unordered_set<int>& clique);
    unordered_set<int> findClique(int startingVertex);
    unordered_set<int> findCluster(int startingVertex);
    float findConnection(int startVertex);
    int buildingCluster(unordered_set<int>& clique);
};


#endif //PROJET_2021_22_QLINHTA_GRAPH_H
