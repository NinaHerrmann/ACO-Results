#config PLATFORM GPU CUDA
#config PROCESSES 1
#config CORES 24
#config GPUS 1
#config MODE debug

const int ants = 256;
const int ncities = 256;
const int IROULETE = 32;
const double PHERINIT = 0.005;
const double EVAPORATION = 0.5;
const int ALPHA = 1;
const int BETA = 2;
const int TAUMAX = 2;
const int block_size = 64;

array<double,512,dist, yes> cities; // cities * 2
array<double,65536,dist, yes> phero; // cities squared
array<double,65536,dist, yes> phero_new; // cities squared
array<double,65536,dist, yes> distance; // cities squared
array<int,512,dist, yes> best_sequence;
array<double,65536,dist, yes> d_delta_phero; // cities squared
array<double,256,dist, yes> d_routes_distance; //n_ants ? TODO ask change in model
array<double,65536,dist, yes> d_probabilities;//n_ants*n_cities
array<int,8192,dist, yes> d_iroulette; // IROULETTE * cities
array<int,65536,dist, yes> d_routes; //n_ants*n_cities

double writeIndex(int i, double y){
	return (double)mkt::rand(0.0, 10.0);
}

double calculate_distance(int i, double y){
	// Distance to itself is zero
	double returner = 0.0;
	// If it is not the distance to itself:
	
	int j = i / ncities;
	int currentcity = (int) i % ncities;
	if (j != currentcity) {
		// sqrt(pow(cities[j*2] - cities[currentcity*2],2) + pow(cities[(j*2) + 1] - cities[(currentcity*2) + 1],2));
		// Euclidean Distance to other city.
		double difference = cities[j*2] - cities[currentcity*2];
		double nextdifference = cities[(j*2) + 1] - cities[(currentcity*2) + 1];
		double pow2 = 2.0;
		double first = mkt::pow(difference, pow2);
		double second = mkt::pow(nextdifference, pow2);
		
		double xdistance = mkt::sqrt(first, second);
		double ydistance = (cities[(j*2) + 1] - cities[(currentcity*2) + 1])*(cities[(j*2) + 1] - cities[(currentcity*2) + 1]);
		returner = mkt::sqrt(xdistance + ydistance);
	}
	return returner;
}

// Returns the 32 closest cities. 
int calculate_iroulette(int irouletteindex, int value){
	// c_index * IROULETE + i == zurzeitiger index
	// this means 
	int i = irouletteindex - (ncities * IROULETE);
	int c_index = (irouletteindex-i) % IROULETE;
	int returner = 0;
	double citydistance = 999999.9;
	double c_dist = 0.0;
	int city = -1;

	for(int j = 0 ;j<ncities;j++){

		bool check = true;

		for(int k = 0 ; k < i ; k++){
			if(d_iroulette[c_index * IROULETE + k] == j){
				check = false;
			}
		}

		if(c_index!=j && check){
			c_dist = distance[(c_index*ncities) + j];
			if(c_dist < citydistance){
				citydistance = c_dist;
				city = j;
			}
		}
	}
	return city;
}

int route_kernel(int Index, int value){
	int newroute = 0;
	int ant_index = ants/block_size;
	int i = Index - ((ant_index * ncities) - 1);
	
	int initialCity = 0;
	double sum = 0.0;

	int next_city = -1;
	double ETA = 0.0;
	double TAU = 0.0;

	// Initialize startcity.
	d_routes[Index * ncities] = initialCity;
	int cityi = d_routes[Index -1];
	int count = 0;
	
	for (int c = 0; c < IROULETE; c++) {

		next_city =  d_iroulette[(cityi * IROULETE) + c];
		int visited = 0;
		// has city been visited? Vielleicht in datenstruktur abspeichern. 
		for (int l=0; l <= i; l++) {
			if (d_routes[ant_index*ncities+l] == next_city) {
				visited = 1;
			}
		}
		if (cityi != next_city && !visited){
			int indexpath = cityi*ncities+ next_city;
			double firstnumber = 1 / distance[indexpath];
			ETA = (double) mkt::pow(firstnumber, BETA);
			TAU = (double) mkt::pow(phero[indexpath], ALPHA);
			sum += ETA * TAU;
		}	
	}

	for (int c = 0; c < IROULETE; c++) {

		next_city = d_iroulette[(cityi * IROULETE) + c];
		int visited = 0;
		// has city been visited? Vielleicht in Datenstruktur abspeichern. 
		for (int l=0; l <= i; l++) {
			if (d_routes[ant_index*ncities+l] == next_city) {
				visited = 1;
			}
		}
		if (cityi == next_city || visited) {
			d_probabilities[ant_index*ncities+c] = 0.0;
		} else {
			double dista = (double)distance[cityi*ncities+next_city];
			// ALPHA ist 1 BETA 2
			double ETAij = 0.0; //(double) mkt::pow(1 / dista , BETA);
			double TAUij = 0.0; //(double) mkt::pow(phero[(cityi * ncities) + next_city], ALPHA);			
			d_probabilities[ant_index*ncities+c] = (ETAij * TAUij) / sum;
			count = count++;
		}
	}

	// deadlock --- it reaches a place where there are no further connections
	if (0 == count) {
		int breaknumber = 0;
		for(int nc = 0; nc < ncities; nc++){
			int visited = 0;
			// has city been visited? Vielleicht in datenstruktur abspeichern. 
			for (int l=0; l <= i; l++) {
				if (d_routes[ant_index*ncities+l] == nc) {
					visited = 1;
				}
			}
			if(!visited){
				breaknumber = nc;
			}
		}
		newroute = breaknumber;
	} else {
		//city(int antK, int n_cities, double* probabilities, curandState* rand_states) {
		double random = mkt::rand(0.0, (double)ncities);
		int ii = 0;
		double summ = d_probabilities[ant_index*ncities];
	
		//while (summ < random){ // && i < n_cities-1) {
		//	i++;
		//	summ += d_probabilities[ant_index*n_cities+i];
		//}
		// simulating while.
		for(int check = 1; check > 0; check++){
			ii = ii++;
			summ += d_probabilities[ant_index*ncities+ii];
			if (summ >= random){
				check = -2;
			}
		}
		int chosen_city = ii;
		newroute = d_iroulette[cityi*IROULETE+chosen_city];
	}
	// datapoint we are updating routes[(ant_index * n_cities) + (i + 1)]
	return newroute;
}

double update_best_sequence_kernel(int Index, double value) {
	int Q = 11340;
	double RO = 0.5;
	double rlength = 0.0;
	int k = Index;
	double sum = 0.0;
	for (int j=0; j<ncities-1; j++) {

		int cityi_infor = d_routes[k*ncities+j];
		int cityj_infor = d_routes[k*ncities+j+1];

		sum += distance[cityi_infor*ncities + cityj_infor];
	}

	int cityi_old = d_routes[k*ncities+ncities-1];
	int cityj_old = d_routes[k*ncities];

	sum += distance[cityi_old*ncities + cityj_old];
	// d_length(k, n_cities[0], routes, distances)
	rlength = sum;
	//d_routes_distance[k] = rlength;

	for (int r=0; r < ncities-1; r++) {

		int cityi = d_routes[k * ncities + r];
		int cityj = d_routes[k * ncities + r + 1];

		d_delta_phero[cityi* ncities + cityj] += Q / rlength;
		d_delta_phero[cityj* ncities + cityi] += Q / rlength;
	}
	return rlength;
}

double update_pheromones_kernel(int Index, double value) {
	int Q = 11340;
	double RO = 0.5;
	for (int i = 0; i<ants; i++){
		
	}
	// Index ncities * ncities + ncities
	double write = (1 - RO) * phero[Index] + d_delta_phero[Index];
	d_delta_phero[Index] = 0.0;

	return write;
}

main{
	double bestroute = 99999999.9;
	mkt::roi_start();
	mkt::print(cities);
	cities.mapIndexInPlace(writeIndex());
	mkt::print(cities);
	
	distance.mapIndexInPlace(calculate_distance());
	d_iroulette.mapIndexInPlace(calculate_iroulette());
	int iterations = 5;
	for (int i = 0; i < iterations; i++){
		d_routes.mapIndexInPlace(route_kernel());
		d_routes_distance.mapIndexInPlace(update_best_sequence_kernel());
        for (int k=0; k<ants; k++) {
			if(d_routes_distance[k] < bestroute){
				bestroute = d_routes_distance[k];
				for (int count = 0; count < ncities; count++) {
					best_sequence[count] = d_routes[k * ncities + count];
				}
			}	
		}
		phero_new.mapIndexInPlace(update_pheromones_kernel());
	}
	mkt::print(distance);
	mkt::roi_end();
}
