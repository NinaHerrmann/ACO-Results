#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <malloc.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>
#include "../include/aco_v1_cuda_algorithm.cuh"
#include "Randoms.cpp"
#define CUDA_ERROR_CHECK

//ACO Constants
#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
#define block_size 64
#define IROULETE 32

//Device Variables
__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_block_size;
__device__ int d_GRAPH_SIZE;

using namespace std;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void setup_rand_kernel(curandState * state, unsigned long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state[id]);

      if(id == 0){
	      d_PHERINIT = 0.005;
	      d_EVAPORATION = 0.5;
	      d_ALPHA = 1;
	      d_BETA = 2;
	      d_TAUMAX = 2;
      }
  __syncthreads();
}

__global__ void calculate_distance_kernel(double* dist, double* cities, int n_cities){
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int to = pos % n_cities;
    int from =  pos / n_cities;
    if(pos < (n_cities*n_cities)){
		if(from != to){
				dist[pos] = sqrt(pow(cities[from*2] - cities[to*2],2) + pow(cities[(from*2) + 1] - cities[(to*2) + 1],2));
			 }else{
				dist[pos] = 0.0;
		}
    }
}

__global__ void calculate_iroulette_kernel(double* dist, double* cities, int* iroulette, int n_cities){
	int c_index = blockIdx.x;
    for(int i = 0 ; i< IROULETE ; i++){
    	double distance = 999999.9;
    	double c_dist = 0.0;
    	int city = -1;
		for(int j = 0 ;j<n_cities;j++){
			bool check = true;
			for(int k = 0 ; k < i ; k++){
				if(iroulette[c_index * IROULETE + k] == j){
					check = false;
				}
			}
			if(c_index!=j && check){
				c_dist = dist[(c_index*n_cities) + j];
				if(c_dist < distance){
					distance = c_dist;
					city = j;
				}
			}
		}
		iroulette[c_index * IROULETE + i] = city;
	}
}

__global__ void route_kernel2(int n_cities, int* routes, double* c_phero, double* c_dist, double* d_probabilities,  int* iroulette, curandState* rand_states){
	int ant_index = blockIdx.x * blockDim.x + threadIdx.x;
	int initialCity = static_cast<int>(curand_uniform(&rand_states[ant_index]) * (static_cast<double>((n_cities))));
	int next_city = -1;
	routes[ant_index * n_cities] = initialCity;
	for (int i=0; i < n_cities-1; i++) {
		int cityi = routes[ant_index*n_cities+i];
		int count = 0;
		double sum = 0.0;
		double ETA [IROULETE] = {};
		double TAU [IROULETE] = {};
		double ETAxTAU [IROULETE] = {};
		for (int c = 0; c < IROULETE; c++) {
			next_city =  iroulette[(cityi * IROULETE) + c];
			if (!vizited(ant_index, next_city, routes, n_cities, i)){
				ETA[c] = (double) pow (1 / c_dist[cityi*n_cities+ next_city], d_BETA);
				TAU[c] = (double) pow (c_phero[(cityi*n_cities)+ next_city], d_ALPHA);
				ETAxTAU[c] = ETA[c] * TAU[c];
				sum += ETAxTAU[c];
			}
		}
		for (int c = 0; c < IROULETE; c++) {
			next_city =  iroulette[(cityi * IROULETE) + c];
			if (vizited(ant_index, next_city,routes, n_cities, i)) {
				d_probabilities[ant_index*n_cities+c] = 0.0;
			}else{
				d_probabilities[ant_index*n_cities+c] = ETAxTAU[c] / sum;
				count++;
			}
		}
		if (0 == count) {
			int breaknumber;
			bool is_set = false;
			for(int nc = 0; nc < n_cities; nc++){
				if(!vizited(ant_index, nc, routes, n_cities, i) && !is_set){
						breaknumber = nc;
						is_set = true;
				}
			}
			routes[(ant_index * n_cities) + (i + 1)] = breaknumber;
		}
		else{
			int chosen_city = city(ant_index, n_cities, d_probabilities, rand_states);
			routes[(ant_index * n_cities) + (i + 1)] = iroulette[cityi*IROULETE+chosen_city];
		}
	}
}

__global__ void update_pheromones_kernel(int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* c_phero, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence) {
	int Q = 11340;
	double RO = 0.5;
	for (int k=0; k<NUMBEROFANTS[0]; k++) {
		double rlength = d_length(k, NUMBEROFCITIES[0], ROUTES, DIST);
		routes_distance[k] = rlength;
		for (int r=0; r < NUMBEROFCITIES[0]-1; r++) {
			int cityi = ROUTES[k * NUMBEROFCITIES[0] + r];
			int cityj = ROUTES[k * NUMBEROFCITIES[0] + r + 1];
			DELTAPHEROMONES[cityi* NUMBEROFCITIES[0] + cityj] += Q / rlength;
			DELTAPHEROMONES[cityj* NUMBEROFCITIES[0] + cityi] += Q / rlength;
		}
		if(routes_distance[k] < bestRoute[0]){
			bestRoute[0] = routes_distance[k];
			for (int count=0; count < NUMBEROFCITIES[0]; count++) {
				d_best_sequence[count] = ROUTES[k * NUMBEROFCITIES[0]+count];
			}
		}
	}

	for (int i=0; i<NUMBEROFCITIES[0]; i++) {
		for (int j=0; j<NUMBEROFCITIES[0]; j++) {
			c_phero[i * NUMBEROFCITIES[0] + j] = (1 - RO) * c_phero[i * NUMBEROFCITIES[0] +j] + DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j];
			DELTAPHEROMONES[i * NUMBEROFCITIES[0] +j] = 0.0;
			c_phero[j * NUMBEROFCITIES[0] + i] = (1 - RO) * c_phero[j * NUMBEROFCITIES[0] +i] + DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i];
			DELTAPHEROMONES[j * NUMBEROFCITIES[0] +i] = 0.0;
		}
	}
}

__device__ bool vizited(int antk, int c, int* ROUTES, int NUMBEROFCITIES, int step) {
	bool visited = false;
	for (int l=0; l <= step; l++) {
		if (ROUTES[antk*NUMBEROFCITIES+l] == c) {
			visited = true;
		}
	}
	return visited;
}

__global__ void evaporation_kernel(double* c_phero, int* NUMBEROFCITIES) {
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	double RO = 0.5;
	int city2 = (NUMBEROFCITIES[0]*NUMBEROFCITIES[0]);
	if(x_index < city2){
		if(blockIdx.x !=  threadIdx.x){
			c_phero[x_index] = (1 - RO) * c_phero[x_index];
		}
	}
}

__global__ void pheromone_deposit_kernel(int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* c_phero, double* DELTAPHEROMONES, double* DIST, double* routes_distance, double* bestRoute, int* d_best_sequence) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int city_id = pos % NUMBEROFCITIES[0];
	int ant_k =  pos / NUMBEROFCITIES[0];
	int city2 = NUMBEROFCITIES[0]*NUMBEROFANTS[0];
	double Q = 11340.0;
	if((city_id < (NUMBEROFCITIES[0]-1)) && (pos < city2)){
		double rlength = d_length(ant_k, NUMBEROFCITIES[0], ROUTES, DIST);
		if(city_id == 0){
			routes_distance[ant_k] = rlength;
		}
		__syncthreads();
		int cityi = ROUTES[ant_k * NUMBEROFCITIES[0] + city_id];
		int cityj = ROUTES[ant_k * NUMBEROFCITIES[0] + city_id + 1];
		double delta_pheromone =  Q / rlength;
		atomicAdd(&c_phero[cityi * NUMBEROFCITIES[0] + cityj],  delta_pheromone);
		atomicAdd(&c_phero[cityj * NUMBEROFCITIES[0] + cityi],  delta_pheromone);
	}
}

__global__ void best_ant_kernel(int* NUMBEROFANTS, int* NUMBEROFCITIES, int* ROUTES, double* routes_distance, double* bestRoute, int* d_best_sequence){
	for(int k = 0 ; k<NUMBEROFANTS[0] ; k++){
		if(routes_distance[k] < bestRoute[0]){
			bestRoute[0] = routes_distance[k];
			for (int count=0; count < NUMBEROFCITIES[0]; count++) {
				d_best_sequence[count] = ROUTES[k * NUMBEROFCITIES[0]+count];
			}
		}
	}
}

__device__ double PHI (int cityi, int cityj, int n_cities, double* c_dist, double* c_phero, double sum) {
	double dista = c_dist[(cityi*n_cities)+cityj];
	double ETAij = (double) pow (1 / dista , d_BETA);
	double TAUij = (double) pow (c_phero[(cityi * n_cities) + cityj],   d_ALPHA);
	return (ETAij * TAUij) / sum;
}

__device__ int city (int antK, int n_cities, double* probabilities, curandState* rand_states) {
    double random = curand_uniform(&rand_states[antK]);
	int i = 0;
	double sum = probabilities[antK*n_cities];
	while (sum < random){ // && i < n_cities-1) {
		i++;
		sum += probabilities[antK*n_cities+i];
	}
	return (int) i;
}

__device__ double d_length (int antk, int n_cities, int* routes, double* distances) {
	double sum = 0.0;
	for (int j=0; j<n_cities-1; j++) {
		int cityi = routes[antk*n_cities+j];
		int cityj = routes[antk*n_cities+j+1];
		sum += distances[cityi*n_cities + cityj];
	}
	int cityi = routes[antk*n_cities+n_cities-1];
	int cityj = routes[antk*n_cities];
	sum += distances[cityi*n_cities + cityj];
	return sum;
}

double run_aco(int n_ant, int n_iterations, int problem){ //158 Lines
	int n_cities = 0;
	int n_ants = n_ant;
	int n_blocks = n_ant/block_size;

	switch (problem) {
		case 1:
			n_cities = 38; //Djbouti
			break;
		case 2:
			n_cities = 980; //Luxemburg
			break;
		case 3:
			n_cities = 194; //Catar
			break;
		case 4:
			n_cities = 280;
			break;
		case 5:
			n_cities = 198;
			break;
		case 6:
			n_cities =  1291;
			break;
		case 7:
			n_cities = 318;
			break;
		case 8:
			n_cities = 442;
			break;
		case 9:
			n_cities = 1173;
			break;
		case 10:
			n_cities = 1002;
			break;
		case 11:
			n_cities = 2392;
			break;
		case 12:
			n_cities =  783;
			break;
	}

	int GPU_N;
	const int MAX_GPU_COUNT = 1;
	cudaGetDeviceCount(&GPU_N);
	if (GPU_N > MAX_GPU_COUNT) {
		GPU_N = MAX_GPU_COUNT;
	}
	cudaStream_t stream[GPU_N];

	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);
	}

	//Create Host variables
    double* cities = new double[n_cities*2];
    double* phero = new double[n_cities*n_cities];
    double* dist  = new double[n_cities*n_cities];
	int* best_sequence  = new int[n_cities];
	double bestRoute = 99999999.9;

	//device variables
    double* d_cities;
    double* d_pheromones;
    double* d_delta_phero;
    double* d_distances;
    double* d_routes_distance;
    double* d_bestRoute;
    double* d_probabilities;
    int* d_iroulette;
    int* d_routes;
    int* d_nants;
    int* d_ncities;
    int* d_best_sequence;

    //Init Random Generators
    curandState* d_rand_states_ind;
	cudaMalloc(&d_rand_states_ind, n_ants *  sizeof(curandState));

    //alloc device variables
    cudaMalloc((void**) &d_cities, n_cities*2*sizeof(double));
    cudaMalloc((void**) &d_pheromones, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_delta_phero, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_distances, n_cities*n_cities*sizeof(double));
    cudaMalloc((void**) &d_routes, n_ants*n_cities*sizeof(int));
    cudaMalloc((void**) &d_routes_distance, n_ants*n_cities*sizeof(double));
    cudaMalloc((void**) &d_bestRoute, sizeof(double));
    cudaMalloc((void**) &d_nants, sizeof(int));
    cudaMalloc((void**) &d_ncities, sizeof(int));
    cudaMalloc((void**) &d_best_sequence, n_cities*sizeof(int));
    cudaMalloc((void**) &d_probabilities, n_ants*n_cities*sizeof(double));
    cudaMalloc((void**) &d_iroulette, n_cities*IROULETE*sizeof(int));

    cudaDeviceSynchronize();

    setup_rand_kernel<<<n_blocks, block_size, 0, stream[0]>>>(d_rand_states_ind, time(NULL));
    cudaDeviceSynchronize();

	readMap(cities, phero, dist, n_cities, problem);
	cudaDeviceSynchronize();

	cudaMemcpy(d_bestRoute, &bestRoute, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pheromones, phero, n_cities*n_cities*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nants, &n_ants, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ncities, &n_cities, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cities, cities, n_cities*2*sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	int kernel_city_threads = n_cities;
	int kernel_city_blocks = n_cities;
	int kernel_ant_blocks = n_ants;

	while(kernel_city_threads > 1024){
		if(kernel_city_threads % 2 == 0){
			kernel_city_threads = kernel_city_threads / 2;
		}else{
			kernel_city_threads = (kernel_city_threads / 2) + 1;
		}
		kernel_city_blocks = kernel_city_blocks * 2;
		kernel_ant_blocks = kernel_ant_blocks * 2;
	}

	calculate_distance_kernel<<<kernel_city_blocks, kernel_city_threads>>>(d_distances, d_cities, n_cities); // calculates the distances of each city+
	cudaDeviceSynchronize();

	calculate_iroulette_kernel<<<n_cities, 1>>>(d_distances, d_cities, d_iroulette, n_cities); // calculates the distances of each city+
	cudaDeviceSynchronize();

	cudaMemcpy(dist, d_distances, (n_cities*n_cities)*sizeof(double), cudaMemcpyDeviceToHost);

	int iteration = 0;
	double seconds_kernel4 = 0.0;
	double seconds_kernel3 = 0.0;
	double seconds_kernel5 = 0.0;

	while(iteration < n_iterations){

		route_kernel2<<<n_blocks, block_size, 0, stream[0]>>>(n_cities, d_routes, d_pheromones, d_distances, d_probabilities,d_iroulette, d_rand_states_ind);
		cudaDeviceSynchronize();

		evaporation_kernel<<<kernel_city_blocks, kernel_city_threads>>>(d_pheromones, d_ncities);
		pheromone_deposit_kernel<<<kernel_ant_blocks,kernel_city_threads>>>(d_nants, d_ncities, d_routes, d_pheromones, d_delta_phero, d_distances, d_routes_distance, d_bestRoute, d_best_sequence);
		cudaDeviceSynchronize();

		best_ant_kernel<<<1,1>>>(d_nants, d_ncities, d_routes, d_routes_distance, d_bestRoute, d_best_sequence);
		cudaDeviceSynchronize();

		iteration ++;
	}

	cudaMemcpy(best_sequence, d_best_sequence, n_cities*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bestRoute, d_bestRoute, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

    return bestRoute;
}

