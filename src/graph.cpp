//
// Created by qlinhta on 03/01/2022.
//

#include "graph.h"
#include "bits/stdc++.h"
using namespace std;

Graph::Graph() {
    string read;
    getline(cin, line);
    read = read.substr(6);
    int blank = read.find(' ');
    vertices = stoi(read.substr(0, blank));
    edges = stoi(read.substr(blank));

}