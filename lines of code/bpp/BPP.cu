#config PLATFORM GPU CUDA
#config PROCESSES 1
#config CORES 1
#config GPUS 1
#config MODE debug

// Configs for ACO
const int ants = 32;
const int BETA = 1;
const double EVAPORATION = 0.5;
const double PHERINIT = 0.005;
const int ALPHA = 1;
const int TAUMAX = 2;
const int BLOCK_SIZE = 32;
const int Q = 32;
// Params to change depending on problem
const int itemtypes = 50;
const int itemcount = 59;
const int pheromone_matrix_size = 2500;
const int antssquaredtimes = 12800;
const int antssquaredcount = 15104;

int bin_capacity = 1000;
// 2* itemtype
const int itemssquaredants = 256;

// TODO Implement DeviceArray to use no option.
array<double, pheromone_matrix_size, dist, yes> d_phero;
array<double, pheromone_matrix_size, dist, yes> d_delta_phero;
array<int, ants, dist, yes> d_fitness;
array<double, antssquaredtimes, dist, yes> d_probabilities;

array<double, itemssquaredants, dist, yes> d_eta;
array<double, itemssquaredants, dist, yes> d_tau;
array<int, itemtypes, dist, yes> bpp_items_weight;
array<int, itemtypes, dist, yes> bpp_items_quantity;

array<int, itemssquaredants, dist, yes> copy_bpp_items_weight;
array<int, itemssquaredants, dist, yes> copy_bpp_items_quantity;

array<int, antssquaredcount, dist, yes> d_bins;
array<int, itemcount, dist, yes> d_best_solution;
array<double, ants, dist, yes> d_rand_states_ind;

int copybppitemsweight(int antss, int indexx, int valuee){
	int new_index = indexx % antss;
	return bpp_items_weight[new_index];
}
int copybppitemsquantity(int antss, int indexx, int valuee){
	int new_index = indexx % antss;
	return bpp_items_quantity[new_index];
}

int packing_kernel(int object_weight, int itemtypess, int itemcountt, int BETA2, int bin_capacity2, int iindex, int y){
	
	int ant_index = iindex;

	int object_bin_index = ant_index * itemtypes;
	int bins_used = 0;

	int actual_bin_weight = 0;
	int n_items_in_actual_bin = 0;

	//Used to check if there are still objects that could fit in the actual bin
	int possible_items_to_this_bin = 0;

	//prefix
	int bpp_items_prefix = (int)ant_index*itemtypess;

	//Start first bin -> Get heaviest item available and add to first bin
	int object_index = 0;
	int object_quantity = 0;
	int new_object_weight = 0;

	for(int i = 0 ; i < itemtypes; i++){
		if(bpp_items_weight == object_weight){
			object_index = i;
		}	
	}

	d_bins[ant_index* (int)itemtypes] = object_index;
	copy_bpp_items_quantity[bpp_items_prefix + object_index] = copy_bpp_items_quantity[bpp_items_prefix + object_index] - 1;
	n_items_in_actual_bin = n_items_in_actual_bin + 1;
	actual_bin_weight += object_weight;


	bins_used = bins_used + 1;
	int weight_object_j = 0;
	int object_i = 0;
	int quantity_object_j = 0;
	//Loop to build complete bins
	for (int ii = 0; ii < itemtypes-1; ii++) {
		double eta_tau_sum = 0.0;
		for (int j = 0; j < itemtypess; j++) {
			d_eta[object_bin_index+j] = 0.0;
			d_tau[object_bin_index+j] = 0.0;
			d_probabilities[object_bin_index+j] = 0.0;
			weight_object_j = copy_bpp_items_weight[bpp_items_prefix + j];
			quantity_object_j = copy_bpp_items_quantity[bpp_items_prefix +j];
			if(quantity_object_j > 0){
				// In case the weight can go into bin. sollten wir nicht das schwerste als naechstes nehmen? 
				if (weight_object_j < (bin_capacity-actual_bin_weight)){
					if(actual_bin_weight == 0){
						d_eta[object_bin_index+j] = 1.0;
					}else{
						for(int k = 0 ; k < n_items_in_actual_bin ; k++){
							// Access out of bounds? i > itemtypes
							// Nehme das objekt was ?
							object_i = d_bins[object_bin_index+ii-k];
							d_eta[object_bin_index+j] += d_phero[(object_i*itemtypes) + j];
						}
						d_eta[object_bin_index+j] = (double) (d_eta[object_bin_index+j] / n_items_in_actual_bin);
					}
					d_tau[object_bin_index+j] = (double) mkt::pow(weight_object_j, BETA);
					eta_tau_sum += d_eta[object_bin_index+j] * d_tau[object_bin_index+j];
					possible_items_to_this_bin = possible_items_to_this_bin + 1;
				}
			}
		}
		if(possible_items_to_this_bin > 0){
			for (int j = 0; j < itemtypes; j++) {
				d_probabilities[object_bin_index+j] = (d_eta[object_bin_index+j] * d_tau[object_bin_index+j]) / eta_tau_sum;
				d_eta[object_bin_index+j] = 0.0;
				d_tau[object_bin_index+j] = 0.0;
			}
			eta_tau_sum = 0.0;
			double random = 0.0; // curand_uniform(&rand_states[ant_index]);
			int object_j = 0;
			double sum = d_probabilities[object_bin_index];
			for (int s = 0; s > -1; s++){
				object_j = object_j + 1;
				sum = sum + d_probabilities[object_bin_index+object_j];
				if (sum < random) {
					s = -2;
				}
			}

			//Add selected object to the list
			d_bins[ant_index*(int)itemtypes+ii+1] = object_j;

			//Add weight to actual bin
			weight_object_j = copy_bpp_items_weight[bpp_items_prefix + object_j];
			actual_bin_weight += weight_object_j;

			copy_bpp_items_quantity[bpp_items_prefix + object_j] = copy_bpp_items_quantity[bpp_items_prefix + object_j] - 1;
			n_items_in_actual_bin = n_items_in_actual_bin + 1;
			possible_items_to_this_bin = 0;

		}else{

			possible_items_to_this_bin = 0;
			actual_bin_weight = 0;
			actual_bin_weight = 0;

			//Start first bin -> Get heaviest item available and add to first bin
			object_index = 0;
			object_weight = 0;
			object_quantity = 0;
			new_object_weight = 0;

			for(int k = 0 ; k < itemtypes ; k++){

				object_quantity = copy_bpp_items_quantity[bpp_items_prefix + k];
				new_object_weight = copy_bpp_items_weight[bpp_items_prefix + k];

				if(object_quantity > 0){
					if (new_object_weight > object_weight){
						object_index = k;
						object_weight = new_object_weight;
					}
				}
			}

			copy_bpp_items_quantity[bpp_items_prefix + object_index] = copy_bpp_items_quantity[bpp_items_prefix + object_index] - 1;
			d_bins[ant_index*itemtypes+ii+1] = object_index;
			n_items_in_actual_bin = n_items_in_actual_bin + 1;
			actual_bin_weight += object_weight;

			bins_used = bins_used + 1;
		}
		}
		return bins_used;
}

double evaporation_kernel(int itemtypess, double EVAPORATION2, int iindex, double y) {

	//Evaporation Rate
	double result = 0.0;
	double RO = EVAPORATION2;
	if(iindex % itemtypess != 0) {
		result = (1 - RO) * d_phero[iindex];
	}
	return result;

}

int update_pheromones_kernel(int itemcountt, int bin_capacity2, int iindex, int value){

	int ant_index = iindex;

	double ant_fitness = d_fitness[ant_index] * 1.0;

	double actual_bin_weight = 0.0;
	int actual_bin_object_index = 0;
	int actual_bin_n_objects = 0;

	for(int i = 0 ; i < itemcountt ; i++){

		double object_weight = (double)d_bins[ant_index*itemcountt+i];

		if(actual_bin_weight + object_weight < bin_capacity2){
			actual_bin_n_objects = actual_bin_n_objects + 1;
			actual_bin_weight = actual_bin_weight + object_weight;
		}else{
			//update pheromones between items from actual bin index -> n-objects
			for(int j = 0; j<actual_bin_n_objects; j++){
				for(int k = j+1; k<actual_bin_n_objects; k++){

					int object_i = d_bins[ant_index*itemcountt+actual_bin_object_index+j];
					int object_j = d_bins[ant_index*itemcountt+actual_bin_object_index+k];

					double delta_pheromone =  Q / (d_fitness[ant_index] * 1.0);

					d_phero[object_i * itemcountt + object_j] =  delta_pheromone + d_phero[object_i * itemcountt + object_j];
					d_phero[object_j * itemcountt + object_i] =  delta_pheromone + d_phero[object_j * itemcountt + object_i];
				}
			}

			//Start new bin count
			actual_bin_n_objects = 1;
			actual_bin_weight = object_weight;
			actual_bin_object_index = i;
		}
	}
	return value;
}


main{
	// Set bestroute high so it is fastly replaced by the number for an existing route
	mkt::roi_start();

	int best_fitness = 999999;

    //Execution Time measure
	double mean_times = 0.0;
	int n_iterations = 5;
	//START iterations
	for (int iterate = 0; iterate < n_iterations; iterate++){

		//START clock
		mkt::roi_start();
		copy_bpp_items_quantity.mapIndexInPlace(copybppitemsquantity(ants));
		copy_bpp_items_weight.mapIndexInPlace(copybppitemsweight(ants));
		int maxobject = bpp_items_weight.reduce(max);
		d_fitness.mapIndexInPlace(packing_kernel(maxobject, itemtypes, itemcount, BETA, bin_capacity));

		//Create Solution / Start Packing
		d_fitness.mapIndexInPlace(packing_kernel(maxobject, itemtypes, itemcount, BETA, bin_capacity));
		d_phero.mapIndexInPlace(evaporation_kernel(itemtypes, EVAPORATION));
		d_fitness.mapIndexInPlace(update_pheromones_kernel(itemtypes, itemcount, bin_capacity));
		best_fitness = d_fitness.reduce(min);
		
		mkt::roi_end();
	}
	mkt::roi_end();
}