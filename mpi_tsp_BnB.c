#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <mpi.h> /* Include MPI's header file */

// Return the minimum of two integers
static int min_int(int a, int b)
{
    return (a < b) ? a : b;
}

// Find the smallest number in the cities cost matrix
static int find_min_edge(int n, int cost[n][n])
{
    int minEdge = INT_MAX;
    // iterate through the 2d array/matrix to find the smallest number(that is not 0)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // skip i==j because that is the cost of a city to itself, which is 0 in our examples
            if (i != j && cost[i][j] < minEdge)
                minEdge = cost[i][j];
        }
    }
    //print the minEdge and number of cities
    // printf("Minimum edge cost: %d\n", minEdge);
    // printf("Number of cities: %d\n", n);
    return minEdge;
}

static int branch_and_bound(int n, int cost[n][n], int vis[], int last, int cnt, int curCost, int *best, int minEdge)
{
    // int myrank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Prune this path if current cost + lower bound of remaining path is greater than or equal to the best cost found so far
    int remainingEdges = n - cnt + 1;
    if (curCost + remainingEdges * minEdge >= *best)
        return *best;

    // Base case
    if (cnt == n)
    {
        // calculate the cost of returning to the starting city to complete the tour
        int tourCost = curCost + cost[last][0];
        // update the best cost if this tour is better than the best found so far
        if (tourCost < *best)
            *best = tourCost;
        return tourCost;
    }

    for (int city = 1; city < n; city++)
    {
        // if the city hasn't been visited.
        if (!vis[city])
        {
            // calculate the cost of visiting this city from the last city
            int nextCost = curCost + cost[last][city];
            // calculate the lower bound of the cost to complete the tour if we visit this city next
            int lowerBound = nextCost + (n - cnt) * minEdge;
            // if the lower bound is greater than or equal to the best cost found so far, prune this path
            if (lowerBound >= *best)
                continue;

            vis[city] = 1;
            branch_and_bound(n, cost, vis, city, cnt + 1, nextCost, best, minEdge);
            vis[city] = 0;
        }
    }

    return *best;
}

int parallel_tsp(int n, int cost[n][n], int rank, int nprocs)
{
    // Make an array to keep track of visited cities, initialized to 0 (not visited)
    int vis[n];
    memset(vis, 0, sizeof(vis));
    // Mark the starting city (0) as visited
    vis[0] = 1;

    int localMin = INT_MAX;
    // Find lowest edge cost in cost matrix. Will be used
    int minEdge = find_min_edge(n, cost);

    for (int city = 1; city < n; city++)
    {
        // Use modulo to distribute the cities among the processes/ranks.
        if (((city - 1) % nprocs) == rank)
        {
            // //print which rank is processing which city
            // printf("Rank %d processing city %d\n", rank, city);
            // //print localMin
            // printf("Rank %d localMin before: %d\n", rank, localMin);
            vis[city] = 1;
            int currentCost = cost[0][city];
            int lowerBound = currentCost + (n - 1) * minEdge;
            if (lowerBound < localMin)
                branch_and_bound(n, cost, vis, city, 2, currentCost, &localMin, minEdge);
            vis[city] = 0;
        }
    }

    return localMin;
}

int main(void)
{
    int rank, nprocs;
    double start, end;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 4 City TSP Cost Matrix
    int cost[4][4];
    if (rank == 0)
    {
        int initial_cost[4][4] = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}};
        memcpy(cost, initial_cost, sizeof(cost));
        start = MPI_Wtime();
    }

    // Broadcast the cost matrix to all processes
    MPI_Bcast(&cost[0][0], 4 * 4, MPI_INT, 0, MPI_COMM_WORLD);

    // TSP function returns the minimum cost of the TSP tour
    int localMin = parallel_tsp(4, cost, rank, nprocs);
    // Reduce all local minimums to find the global minimum
    int globalMin;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Only the root process will print the result
    if (rank == 0)
    {
        end = MPI_Wtime();
        printf("\n\n4-city shortest path: %d\n", globalMin);
        printf("4-city time taken: %f seconds\n\n", end - start);
    }

    /* *********************************************************************************************************** */
    // 13 City TSP Cost Matrix
    int cost13[13][13];
    if (rank == 0)
    {
        int initial_cost13[13][13] = {
            // Source: https://developers.google.com/optimization/routing/tsp#c++_2
            /* 0. New York - 1. Los Angeles - 2. Chicago - 3. Minneapolis - 4. Denver - 5. Dallas - 6. Seattle -
            7. Boston - 8. San Francisco - 9. St. Louis - 10. Houston - 11. Phoenix - 12. Salt Lake City */
            {0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972},
            {2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579},
            {713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260},
            {1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987},
            {1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371},
            {1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999},
            {2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701},
            {213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099},
            {2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600},
            {875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162},
            {1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200},
            {2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504},
            {1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0}};

        memcpy(cost13, initial_cost13, sizeof(cost13));
        start = MPI_Wtime();
    }
    MPI_Bcast(&cost13[0][0], 13 * 13, MPI_INT, 0, MPI_COMM_WORLD);

    localMin = parallel_tsp(13, cost13, rank, nprocs);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        end = MPI_Wtime();
        printf("\n\n13-city shortest path: %d\n", globalMin);
        printf("13-city time taken: %f seconds\n\n", end - start);
    }

    /* *********************************************************************************************************** */
    // 17 City TSP Cost Matrix
    int cost17[17][17];
    if (rank == 0)
    {
        int initial_cost17[17][17] = {
            // Source: https://developers.google.com/optimization/routing/vrp#distance_matrix_api
            // Locations are arbitrary and do not correspond to actual cities
            {0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662},
            {548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210},
            {776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754},
            {696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358},
            {582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244},
            {274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708},
            {502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480},
            {194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856},
            {308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514},
            {194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468},
            {536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354},
            {502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844},
            {388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730},
            {354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536},
            {468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194},
            {776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798},
            {662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0}};

        memcpy(cost17, initial_cost17, sizeof(cost17));
        start = MPI_Wtime();
    }
    MPI_Bcast(&cost17[0][0], 17 * 17, MPI_INT, 0, MPI_COMM_WORLD);

    localMin = parallel_tsp(17, cost17, rank, nprocs);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        end = MPI_Wtime();
        printf("\n\n17-city shortest path: %d\n", globalMin);
        printf("17-city time taken: %f seconds\n\n", end - start);
    }

    MPI_Finalize();
    return 0;
}