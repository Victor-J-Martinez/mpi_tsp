#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> /* Include MPI's header file */

static int min_int(int a, int b) {
    return (a < b) ? a : b;
}

int dfs(int n, int cost[n][n], int vis[], int last, int cnt) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (cnt == n)
        return cost[last][0];

    int minCost = 1000000000;

    for (int city = 1; city < n; city++) {
        if (!vis[city]) {
            vis[city] = 1;
            printf("dfs; RANK %d in city: %d\n", myrank, city);
            printf("cost from last city(%d) to city(%d): %d\n", last, city, cost[last][city]);
            int candidate = cost[last][city] + dfs(n, cost, vis, city, cnt + 1);
            minCost = min_int(minCost, candidate);
            vis[city] = 0;
        }
    }

    return minCost;
}

int parallel_tsp(int n, int cost[n][n], int rank, int nprocs) {
    int vis[n];
    memset(vis, 0, sizeof(vis));
    vis[0] = 1;

    int localMin = 1000000000;

    for (int city = 1; city < n; city++) {
        if (((city - 1) % nprocs) == rank) {
            vis[city] = 1;
            printf("PAR_TSP; RANK %d in city: %d\n", rank, city);

            printf("cost from last city(%d) to city(%d): %d\n", 0, city, cost[0][city]);
            int candidate = cost[0][city] + dfs(n, cost, vis, city, 2);
            localMin = min_int(localMin, candidate);
            vis[city] = 0;
        }
    }

    return localMin;
}

int main(void) {
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int cost[4][4];
    if (rank == 0) {
        int initial_cost[4][4] = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };
        memcpy(cost, initial_cost, sizeof(cost));
    }

    MPI_Bcast(&cost[0][0], 4 * 4, MPI_INT, 0, MPI_COMM_WORLD);

    int localMin = parallel_tsp(4, cost, rank, nprocs);
    int globalMin;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%d\n", globalMin);
    }

    MPI_Finalize();
    return 0;
}
