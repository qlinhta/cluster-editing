//
// Created by qlinhta on 03/01/2022.
//
#include "bits/stdc++.h"

int main(int argc,  char* argv[]) {
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = term;
    action.sa_flags = 0;
    sigaction(SIGTERM, &action, NULL);

}