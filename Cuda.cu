#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define block size and grid size (for CUDA parallel computing) for 8192
#define blocks  32     // Number of thread blocks per grid
#define grids   32       // Number of thread grids
#define PI      3.1415926535897932
#define FREE    2          // Degrees of freedom
#define NLCUT   1.0     // Neighbor list cutoff distance (in units of small particle diameter)
#define LISTMAX 32      // Maximum number of neighbors in the neighbor list

/**
 * tpbox - Stores simulation box parameters
 * natom: Total number of particles
 * x, y: Box dimensions
 * xinv, yinv: Reciprocals of box dimensions (used for periodic boundary conditions)
 * phi: Volume fraction
 * vol: Box volume
 * dens: Particle number density
 * ratio: Diameter ratio of large to small particles
 * alpha: Repulsive potential parameter
 * dt: Time step size
 * fd: Self-propulsion force magnitude
 */
typedef struct 
{
    unsigned natom;
    double   x, xinv;
    double   y, yinv;
    double   phi, vol, dens;
    double   ratio;
    double   alpha;
    double   dt;
    double   fd;
} tpbox;

/**
 * tpcontr - Simulation control parameters
 * relaxtime: System relaxation time
 * ndt: Output data time step interval
 * nperiod: Total number of simulation periods
 */
typedef struct 
{
    double  relaxtime;
    int     ndt;
    int     nperiod;
} tpcontr;

/**
 * tpsys - Stores system state
 * pre: Pressure
 * pot: Potential energy
 * forpre: Force sum used for pressure calculation
 * max_dis: Maximum displacement (used to check for neighbor list updates)
 */
typedef struct 
{
    double  pre, pot;
    double  forpre;
	double  max_dis;
} tpsys;

/**
 * tpvec - Stores basic particle information
 * d: Particle diameter
 * x, y: Position coordinates
 * fx, fy: Force components
 */
typedef struct 
{
    double  d; // Particle diameter
    double  x, y; 
    double  fx, fy; 
} tpvec;

/**
 * tpheun - Stores intermediate variables needed for Heun integrator
 * thex: Angle of self-propulsion direction
 * cosx, cosy: Unit vector components of self-propulsion direction
 * x, y: Position in the prediction step
 * fx, fy: Force in the prediction step
 */
typedef struct 
{
	double  thex;
	double  cosx, cosy;
    double  x, y;
    double  fx, fy;
} tpheun;

/**
 * tplist - Stores particle neighbor lists
 * nbsum: Number of neighbors
 * nb: Array of neighbor indices
 * x, y: Particle position when list was created (for checking if list needs update)
 */
typedef struct 
{
    int nbsum;
    int nb[LISTMAX];
    double  x, y;
} tplist; 

// Global variables
int np;                             
tpcontr contr;                          // Control parameters
tpbox box;                              // Box parameters
__constant__ tpbox dbox;                // Box parameters in GPU constant memory
__device__ __managed__ tpsys dsyst;     // System state in unified memory
// Shared memory variables (shared within each thread block)
__shared__ double block_forpre[blocks]; // For pressure calculation reduction
__shared__ double block_pot[blocks];    // For potential energy reduction
__shared__ double block_maxdis[blocks]; // For maximum displacement reduction

// Function declarations
void   	get_inital  (tpvec *con);                  // Initialize particle positions
void 	read_inital (tpvec *con, char *filename);  // Read initial configuration from file
void 	get_self    (tpheun *heun);                // Initialize self-propulsion directions
void 	active 	 	(tpvec *con, tplist *nblist, tpheun *heun);  // Main simulation loop

// CUDA kernel function declarations
__global__ void predic     (tpvec *dcon, tpheun *dheun);           // Prediction step of Heun integrator
__global__ void correc 	   (tpvec *dcon, tpheun *dheun);           // Correction step of Heun integrator
__global__ void update     (tpvec *dcon, tplist *dnblist);         // Update particle positions for neighbor list
__global__ void make_list  (tpvec *dcon, tplist *dnblist);         // Build neighbor list
__global__ void cal_force  (tpvec *dcon, tplist *dnblist);         // Calculate repulsive forces between particles
__global__ void check_list (tpvec *dcon, tplist *dnblist);         // Check if neighbor list needs update
__global__ void heun_force (tpheun *dheun, tpvec *dcon, tplist *dnblist);  // Calculate forces in prediction step

/**
 * atomicAdd - Perform atomic addition operation on double-precision floating point
 * Early CUDA versions don't natively support double-precision atomic operations,
 * this function provides an implementation
 * Uses atomic compare-and-swap (CAS) operation
 */
__device__ double atomicAdd (double* address, double val) 
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS (address_as_ull, assumed,
        __double_as_longlong (val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double (old);
}

/**
 * atomicMax - Perform atomic maximum operation on double-precision floating point
 * Also uses atomic compare-and-swap (CAS) operation
 */
__device__ double atomicMax (double* address, double val) 
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS (address_as_ull, assumed,
            __double_as_longlong (val>__longlong_as_double(assumed) ? val : __longlong_as_double(assumed)) 
        );
    } while (assumed != old);
    return __longlong_as_double (old);
}

/**
 * main - Main function
 * Initialize the system and start the simulation
 */
int main (void) 
{    
    // Set simulation parameters
  	contr.ndt       = 100;        // Output interval steps
    contr.relaxtime = 100.0;      // System relaxation time
    contr.nperiod   = 10000;      // Number of simulation periods
    box.ratio       = 1.4;        // Diameter ratio of large to small particles
    box.alpha       = 2.0;        // Repulsive force exponent
    box.natom       = 4096;       // Total number of particles
    box.phi         = 0.84;       // Volume fraction
    box.fd 	        = 1e-3;       // Self-propulsion force magnitude
    box.dt          = 0.1;        // Time step size
  
    // Allocate memory and initialize particles
	tpvec *con = (tpvec *) malloc(box.natom*sizeof(tpvec));
    memset(con, 0.0, box.natom*sizeof(tpvec));
    get_inital (con); 
    
    // Save initial configuration
    FILE *cio = fopen ("con0.dat", "w+");
    for (int i=0; i<box.natom; i++) {
        fprintf(cio,"%26.16e\t%26.16e\n", con[i].x, con[i].y);
    }
    fclose(cio);

    // Allocate memory for neighbor lists
    tplist *nblist = (tplist *) malloc(box.natom*sizeof(tplist));
	memset (nblist, 0.0, box.natom*sizeof(tplist));
    
    // Allocate memory for Heun integrator and initialize self-propulsion directions
    tpheun *heun = (tpheun *) malloc(box.natom*sizeof(tpheun));
    memset (heun, 0.0, box.natom*sizeof(tpheun));
	get_self (heun);	

    // Save self-propulsion directions
    FILE *ang = fopen ("angle.dat", "w+");
    for (int i=0; i<box.natom; i++) {
        fprintf(ang,"%26.16e\t%26.16e\n", heun[i].cosx, heun[i].cosy);
    }
    fclose(ang);

    // Start main simulation
	active (con, nblist, heun);

    return 0;
}

/**
 * active - Main simulation loop
 * Executes particle dynamics simulation on the GPU
 * Parameters:
 *   con - Particle array
 *   nblist - Neighbor list array
 *   heun - Heun integrator array
 */
void active (tpvec *con, tplist *nblist, tpheun *heun) 
{
    // Set CUDA thread block and grid dimensions
	dim3   dimblock = blocks;
	dim3   dimgrid  = grids;

	int istart = (int)(contr.relaxtime / box.dt); 	
	
	// Allocate GPU memory
	tpvec  *dcon;     // Particle array on GPU
	tplist *dnblist;  // Neighbor list on GPU
	tpheun *dheun;    // Heun integrator on GPU
	
	cudaMalloc( (void**)&dcon,    box.natom*sizeof(tpvec)  );  
	cudaMalloc( (void**)&dnblist, box.natom*sizeof(tplist) ); 
	cudaMalloc( (void**)&dheun,   box.natom*sizeof(tpheun) ); 
	
	// Copy data from CPU to GPU
	cudaMemcpy( dcon,    con,     box.natom*sizeof(tpvec),  cudaMemcpyHostToDevice );
	cudaMemcpy( dnblist, nblist,  box.natom*sizeof(tplist), cudaMemcpyHostToDevice );
	cudaMemcpy( dheun,   heun,    box.natom*sizeof(tpheun), cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( dbox, &box, sizeof(tpbox) );   // Copy box parameters to constant memory
	cudaDeviceSynchronize();  // Synchronize to ensure all copy operations complete
    
    // Initialize neighbor list
	update <<<dimgrid, dimblock>>> (dcon, dnblist);    // Update particle positions for neighbor list
	HANDLE_ERROR (cudaDeviceSynchronize());
	make_list <<<dimgrid, dimblock>>> (dcon, dnblist); // Build neighbor list
	HANDLE_ERROR (cudaDeviceSynchronize()); 

	int step;	
    int steptot = istart + contr.nperiod * contr.ndt;  // Total simulation steps
    
    // Main simulation loop
    for (step=1; step < steptot; step++)
    {
        // Reset force and energy counters
		dsyst.forpre = 0.0;
    	dsyst.pot    = 0.0;
		
		// Calculate repulsive forces between particles
		cal_force <<<dimgrid, dimblock>>> (dcon, dnblist);
		HANDLE_ERROR (cudaDeviceSynchronize());
		
		// Calculate system pressure and potential energy
		dsyst.pre    = dsyst.forpre / (double)(FREE * box.vol);
    	dsyst.pot    = 0.5 * dsyst.pot / (double)(box.natom);

        // Prediction step of Heun integrator
        predic <<<dimgrid, dimblock>>> (dcon, dheun);
    	HANDLE_ERROR (cudaDeviceSynchronize());
		
		// Calculate forces at predicted positions
		heun_force <<<dimgrid, dimblock>>> (dheun, dcon, dnblist);
		HANDLE_ERROR (cudaDeviceSynchronize());
        
        // Correction step of Heun integrator
        correc <<<dimgrid, dimblock>>> (dcon, dheun);
		HANDLE_ERROR (cudaDeviceSynchronize());

        // Check if neighbor list needs to be updated
		dsyst.max_dis = 0.0;        
		check_list <<<dimgrid, dimblock>>> (dcon, dnblist);
		HANDLE_ERROR (cudaDeviceSynchronize());  

        // If maximum displacement exceeds threshold, update neighbor list
  		if (dsyst.max_dis > NLCUT)
		{
	        printf("%d\t%lf\n", 0, dsyst.max_dis);
            update <<<dimgrid, dimblock>>> (dcon, dnblist);
			HANDLE_ERROR (cudaDeviceSynchronize());

			make_list <<<dimgrid, dimblock>>> (dcon, dnblist);
			HANDLE_ERROR (cudaDeviceSynchronize()); 
		}	
        // Output system state at specified intervals
        if (step%contr.ndt == 0){
	        printf ("%d\t,%16.6e\t%16.6e\n", step, dsyst.pre, dsyst.pot);
        }
        // Additional data collection and analysis code can be added here
        if (step>istart && step%contr.ndt==0){             
        }
    }

    // Free GPU memory
	cudaFree (dcon);  
	cudaFree (dnblist); 
	cudaFree (dheun);
    return;
}

/**
 * check_list - Check if neighbor list needs to be updated
 * Calculates displacement of each particle from when the neighbor list was created,
 * finds the maximum displacement
 * Parameters:
 *   dcon - Particle array on GPU
 *   dnblist - Neighbor list array on GPU
 */
__global__ void check_list (tpvec *dcon, tplist *dnblist)
{
    int ni  = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
	if (ni >= dbox.natom) return;  // Boundary check
	int tid = threadIdx.x;  // Thread index within block
		
	// Calculate particle displacement since neighbor list creation
	double rx = dcon[ni].x - dnblist[ni].x;
	double ry = dcon[ni].y - dnblist[ni].y;
    block_maxdis[tid] = 2.0 * sqrt( rx * rx + ry * ry);  // Store in shared memory

    __syncthreads();  // Synchronize all threads within block

    // Parallel reduction to find maximum displacement within block
    int jj = blocks / 2;
    while (jj != 0) {
        if (tid < jj){
            block_maxdis[tid] = fmax( block_maxdis[tid], block_maxdis[tid+jj] );
        }
        __syncthreads();
        jj = jj / 2;
    }
    
    // First thread of block updates global maximum displacement using atomic operation
    if (tid == 0){
        atomicMax( &dsyst.max_dis, block_maxdis[0] );		
	}
    return;
}

/**
 * cal_force - Calculate repulsive forces between particles
 * Optimized using neighbor lists
 * Parameters:
 *   dcon - Particle array on GPU
 *   dnblist - Neighbor list array on GPU
 */
__global__ void cal_force (tpvec *dcon, tplist *dnblist)
{
	int ni  = blockIdx.x * blockDim.x + threadIdx.x;	// Global thread index
	if (ni >= dbox.natom) return;  // Boundary check
	int tid = threadIdx.x;  // Thread index within block

    // Initialize forces to zero
	dcon[ni].fx = 0.0;  
    dcon[ni].fy = 0.0;             
	double di = dcon[ni].d;  // Particle diameter
	double xi = dcon[ni].x;  // Particle x-coordinate
	double yi = dcon[ni].y;  // Particle y-coordinate
	double forpre = 0.0;     // Force accumulator for pressure calculation
	double pot    = 0.0;     // Potential energy accumulator
	int thjmax = dnblist[ni].nbsum;  // Number of neighbors

    // Loop through all neighbors
	for (int kl=0; kl <= thjmax; kl++)
    {   
		int nj = dnblist[ni].nb[kl];    // Neighbor index				
        double xij = xi - dcon[nj].x;   // x-direction distance
        double yij = yi - dcon[nj].y;   // y-direction distance
        // Apply periodic boundary conditions
        xij = xij - round(xij * dbox.xinv) * dbox.x;
        yij = yij - round(yij * dbox.yinv) * dbox.y;
		double rijsq = xij * xij + yij * yij;       // Squared distance
        double dij   = ( di + dcon[nj].d ) * 0.5;   // Average diameter
		
		// If particles overlap, calculate repulsive force
		if (rijsq < dij * dij)
        {
			double rij = sqrt(rijsq);  // Distance
            double fr  = pow(1.0 - rij / dij, dbox.alpha - 1.0) / dij / rij;  // Force magnitude
            // Accumulate forces
            dcon[ni].fx = dcon[ni].fx + fr * xij;  
            dcon[ni].fy = dcon[ni].fy + fr * yij;             
            forpre = forpre + rijsq * fr;  // For pressure calculation                	
			pot    =  pot   + pow (1.0 - rij / dij, dbox.alpha) / dbox.alpha;  // Potential energy
		}         
    }
    // Store thread results in shared memory
    block_forpre[tid] = forpre;
	block_pot[tid]    = pot;
    __syncthreads();  // Synchronize all threads within block

    // Parallel reduction to calculate block sums
    int jj = blocks / 2;
    while (jj != 0) {
        if (tid < jj) {
            block_forpre[tid] = block_forpre[tid] + block_forpre[tid+jj];
			block_pot[tid]    = block_pot[tid]    + block_pot[tid+jj];
        }
        __syncthreads();
        jj = jj / 2;
    }
    // First thread of block updates global sums using atomic operations
    if (tid == 0){
        atomicAdd (&dsyst.forpre, block_forpre[0]);
		atomicAdd (&dsyst.pot,    block_pot[0]);		
	}
    return;
}

/**
 * make_list - Build neighbor list
 * Parameters:
 *   dcon - Particle array on GPU
 *   dnblist - Neighbor list array on GPU
 */
__global__ void make_list (tpvec *dcon, tplist *dnblist)
{	
	int ni = blockIdx.x * blockDim.x + threadIdx.x;	// Global thread index
	if (ni >= dbox.natom) return;  // Boundary check		

	double xi = dcon[ni].x;  // Particle x-coordinate
	double yi = dcon[ni].y;  // Particle y-coordinate
	// Loop through all other particles
	for (int nj=0; nj < dbox.natom; nj++)
	{	
		if (nj == ni) continue;  // Skip self
		
		double xij = xi - dcon[nj].x;  // x-direction distance
        double yij = yi - dcon[nj].y;  // y-direction distance
        // Apply periodic boundary conditions
        xij = xij - round(xij * dbox.xinv) * dbox.x;
        yij = yij - round(yij * dbox.yinv) * dbox.y;
        double rijsq  = xij * xij + yij * yij;  // Squared distance
        double dijcut = dbox.ratio + NLCUT;  // Cutoff distance
        // If within cutoff distance, add to neighbor list
        if (rijsq < dijcut * dijcut)
        {
        	if (dnblist[ni].nbsum == LISTMAX-1) continue;  // Neighbor list full
            dnblist[ni].nbsum = dnblist[ni].nbsum + 1;  // Increment neighbor count
            dnblist[ni].nb[dnblist[ni].nbsum] = nj;  // Add to neighbor list
        }  
	}
    return;
}

/**
 * get_inital - Initialize particle system
 * Distributes particles randomly, sets diameters and box dimensions
 * Parameters:
 *   con - Particle array
 */
void get_inital (tpvec *con) 
{        
    srand(np); // Initialize random number generator

    // Set particle diameters
    tpvec *ni;
    for (ni = con; ni < con + box.natom; ni++)
    { 
        if (ni < con + (box.natom/2)) 
            ni->d = 1.0;  // Small particles
        else
            ni->d = 1.0 * box.ratio;  // Large particles
    }
    // Calculate total area of all particles
    double  sdisk = 0.0;
    for ( ni = con; ni < con + box.natom; ni++){
        sdisk += PI * (ni->d/2.0) * (ni->d/2.0); 
    }
    // Calculate box dimensions based on volume fraction
    box.vol   = sdisk / box.phi;
    double temp = sqrt(box.vol);
    box.x     = temp;
    box.y     = temp;    
    box.xinv  = 1.0 / temp;
    box.yinv  = 1.0 / temp;
    box.dens  = (double)box.natom / box.vol;
  
    // Distribute particles randomly
    for ( ni = con; ni < con + box.natom; ni++){
        ni->x = ((double)rand()/RAND_MAX - 0.5) * box.x;
        ni->y = ((double)rand()/RAND_MAX - 0.5) * box.y;
    }
    return;
}

/**
 * get_self - Initialize self-propulsion directions
 * Randomly assigns a direction to each particle, then ensures uniform overall direction
 * Parameters:
 *   heun - Heun integrator array
 */
void get_self (tpheun *heun) 
{
	srand ((unsigned)time(NULL));  // Initialize random number generator
	
	// Assign random direction to each particle
	for(int ni=0; ni<box.natom; ni++)
	{
		heun[ni].thex = (double)rand()/RAND_MAX * 2.0 * PI;  // Random angle
		heun[ni].cosx = cos(heun[ni].thex);  // x-direction component
		heun[ni].cosy = sin(heun[ni].thex);  // y-direction component
	} 
	// Iterate 1000 times to ensure uniform overall direction (no bulk drift)
	for(int k=0; k<1000; k++)
	{	
		// Calculate average direction
		double sumx = 0.0;
		double sumy = 0.0;
		for (int ni=0; ni<box.natom; ni++)
		{
			sumx = sumx + heun[ni].cosx;
			sumy = sumy + heun[ni].cosy;
		}	
		sumx = sumx / (double)box.natom;
		sumy = sumy / (double)box.natom;
		// Subtract average direction from each particle's direction
		for (int ni=0; ni<box.natom; ni++)
		{
			heun[ni].cosx = heun[ni].cosx - sumx;
			heun[ni].cosy = heun[ni].cosy - sumy;			
		} 
		// Normalize each particle's direction vector
		for (int ni=0; ni<box.natom; ni++)
		{
			double uin = sqrt(heun[ni].cosx * heun[ni].cosx + heun[ni].cosy * heun[ni].cosy);
			heun[ni].cosx = heun[ni].cosx / uin;
			heun[ni].cosy = heun[ni].cosy / uin;
		}
	}
    return;
}   

/**
 * heun_force - Calculate inter-particle forces in Heun integrator prediction step
 * CUDA kernel function, executed in parallel on GPU
 * Parameters:
 *   dheun - Heun integrator array with predicted positions
 *   dcon - Particle array
 *   dnblist - Neighbor list array
 */
__global__ void heun_force (tpheun *dheun, tpvec *dcon, tplist *dnblist)
{
    // Calculate global thread index (each thread processes one particle)
	int ni = blockIdx.x * blockDim.x + threadIdx.x;	
	// Exit function if index exceeds number of particles
	if (ni >= dbox.natom)
        return;

    // Initialize forces to zero
	dheun[ni].fx = 0.0;  
    dheun[ni].fy = 0.0;             
	double di = dcon[ni].d;            // Current particle diameter
	double xi = dheun[ni].x;           // Predicted position x-coordinate
	double yi = dheun[ni].y;           // Predicted position y-coordinate
	int thjmax = dnblist[ni].nbsum;    // Number of neighbor particles
        
    // Loop through all neighbor particles
	for (int kl=0; kl <= thjmax; kl++)
    {   
	    int nj = dnblist[ni].nb[kl];   // Neighbor particle index				
        double xij = xi - dheun[nj].x;    // Relative distance in x-direction
        double yij = yi - dheun[nj].y;    // Relative distance in y-direction
        // Apply periodic boundary conditions (minimum image convention)
        xij = xij - round(xij * dbox.xinv) * dbox.x;
        yij = yij - round(yij * dbox.yinv) * dbox.y;
        // Calculate squared distance
		double rijsq = xij * xij + yij * yij;
        // Calculate contact distance (average diameter of two particles)
        double dij = (di + dcon[nj].d) * 0.5;
        // If particles overlap (distance less than contact distance)
		if (rijsq < dij * dij)
        {
			double rij = sqrt(rijsq);  // Actual distance
            // Calculate repulsive force magnitude (soft-core repulsion)
            double fr = pow(1.0 - rij / dij, dbox.alpha - 1.0) / dij / rij;
            // Update force components based on direction
            dheun[ni].fx = dheun[ni].fx + fr * xij;  
            dheun[ni].fy = dheun[ni].fy + fr * yij;             
		}         
    }
    return;
}

/**
 * predic - Prediction step of Heun integrator
 * Calculate predicted positions using current positions and forces
 * Parameters:
 *   dcon - Particle array (current state)
 *   dheun - Heun integrator array (for storing predicted state)
 */
__global__ void predic (tpvec *dcon, tpheun *dheun) 
{
	int ni = blockIdx.x * blockDim.x + threadIdx.x;	
    if (ni >= dbox.natom) return;
	dheun[ni].x = dcon[ni].x + (dbox.fd * dheun[ni].cosx + dcon[ni].fx) * dbox.dt;
	dheun[ni].y = dcon[ni].y + (dbox.fd * dheun[ni].cosy + dcon[ni].fy) * dbox.dt;
    return;
}

/**
 * correc - Correction step of Heun integrator
 * Update positions using average of current and predicted forces
 * Parameters:
 *   dcon - Particle array (current state, will be updated)
 *   dheun - Heun integrator array (contains predicted state)
 */
__global__ void correc (tpvec *dcon, tpheun *dheun) 
{
	int ni = blockIdx.x * blockDim.x + threadIdx.x;	
    if (ni >= dbox.natom) return;
	dcon[ni].x = dcon[ni].x + (dbox.fd * dheun[ni].cosx + dcon[ni].fx / 2.0 + dheun[ni].fx / 2.0) * dbox.dt;
	dcon[ni].y = dcon[ni].y + (dbox.fd * dheun[ni].cosy + dcon[ni].fy / 2.0 + dheun[ni].fy / 2.0) * dbox.dt;
    return;	
}

/**
 * update - Update particle positions in neighbor list and reset neighbor list
 * Called before building new neighbor list
 * Parameters:
 *   dcon - Particle array
 *   dnblist - Neighbor list array
 */
__global__ void update (tpvec *dcon, tplist *dnblist)
{ 	
	int ni = blockIdx.x * blockDim.x + threadIdx.x;	
    if (ni >= dbox.natom) return;
    
	double xi = dcon[ni].x;
	double yi = dcon[ni].y;
    dnblist[ni].x = xi;
    dnblist[ni].y = yi;
	dnblist[ni].nbsum = -1;
    for (int n=0; n < LISTMAX; n++){
        dnblist[ni].nb[n] = 0;
    }
    return;
}



