#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define FOR_EACH_CELL for ( i=1 ; i<=N ; i++ ) { for ( j=1 ; j<=N ; j++ ) {
#define END_FOR }}

#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>

#include <OpenCL/OpenCL.h>

void add_source ( int N, float * x, float * s, float dt )
{
	int i, size=(N+2)*(N+2);
	for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
}



void set_bnd ( int N, int b, float * x, float* boundMask ,float vel)
{
	int i;
	for(i=0;i<N*N;i++)
	{
		if(boundMask[i]<=128.0)
		{
			x[i]=0.0;
		}
	}
	
	
	for ( i=1 ; i<=N ; i++ )
	{	
		//x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
		//x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
		x[IX(0  ,i)] = b==1 ? vel: x[IX(1,i)];
		x[IX(N+1,i)] = b==1 ? 0. : x[IX(N,i)];
		x[IX(i,0  )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
		x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
		//x[IX(i,0  )] = b==2 ? 0.05 : x[IX(1,i)];
		//x[IX(i,N+1)] = b==2 ? 0.05 : x[IX(N,i)];
	}
	x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
	x[IX(0  ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0  ,N)]);
	x[IX(N+1,0  )] = 0.5f*(x[IX(N,0  )]+x[IX(N+1,1)]);
	x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N)]);
}

void lin_solve ( int N, int b, float * x, float * x0, float a, float c,float* boundMask ,float vel)
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) {
		FOR_EACH_CELL
			x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
		END_FOR
		set_bnd ( N, b, x, boundMask,vel);
	}
}

void diffuse ( int N, int b, float * x, float * x0, float diff, float dt,float* boundMask,float vel)
{
	float a=dt*diff*N*N;
	lin_solve ( N, b, x, x0, a, 1+4*a,boundMask ,vel);
}

void advect ( int N, int b, float * d, float * d0, float * u, float * v, float dt,float* boundMask,float vel )
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = dt*N;
	FOR_EACH_CELL
		x = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
		if (x<0.5f) x=0.5f; if (x>N+0.5f) x=N+0.5f; i0=(int)x; i1=i0+1;
		if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
		s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
		d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+
					 s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
	END_FOR
	set_bnd ( N, b, d , boundMask,vel);
}

void project ( int N, float * u, float * v, float * p, float * div, float* boundMask,float vel)
{
	int i, j;

	FOR_EACH_CELL
		div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/N;
		p[IX(i,j)] = 0;
	END_FOR	
	set_bnd ( N, 0, div ,boundMask,vel); set_bnd ( N, 0, p,boundMask,vel);

	lin_solve ( N, 0, p, div, 1, 4, boundMask,vel);

	FOR_EACH_CELL
		u[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)]-p[IX(i-1,j)]);
		v[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)]-p[IX(i,j-1)]);
	END_FOR
	set_bnd ( N, 1, u , boundMask,vel); set_bnd ( N, 2, v, boundMask,vel );
}

void dens_step( int N,float * x, float * x0, float * u, float * v,float* boundMask, float diff, float dt ,float vel)
{

	add_source ( N, x, x0, dt );
	SWAP ( x0, x ); diffuse ( N, 0, x, x0, diff, dt ,boundMask,vel);
	SWAP ( x0, x ); advect ( N, 0, x, x0, u, v, dt ,boundMask,vel);
}

void vel_step( int N,float * u, float * v, float * u0, float * v0,float* boundMask, float visc, float dt ,float vel)
{

	add_source ( N, u, u0, dt ); add_source ( N, v, v0, dt );
	SWAP ( u0, u ); diffuse ( N, 1, u, u0, visc, dt ,boundMask,vel);
	SWAP ( v0, v ); diffuse ( N, 2, v, v0, visc, dt ,boundMask,vel);
	project ( N, u, v, u0, v0 ,boundMask,vel);
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( N, 1, u, u0, u0, v0, dt ,boundMask,vel); advect ( N, 2, v, v0, u0, v0, dt ,boundMask,vel);
	project ( N, u, v, u0, v0,boundMask ,vel);
}

void runstep( int N,float * u, float * v, float * u0, float * v0, float* dens,float* dens_prev,float* boundMask, float visc, float dt ,float diff,float vel)
{
	vel_step(N, u, v,u0,v0, boundMask, visc, dt,vel);
     dens_step(N,dens,dens_prev,u,v, boundMask, diff, dt,vel);
	int testVal=0;
	testVal = testOpenCL(N);
}


#pragma mark -
#pragma mark Utilities
char * load_program_source(const char *filename)
{ 
	
	struct stat statbuf;
	FILE *fh; 
	char *source; 
	
	fh = fopen(filename, "r");
	if (fh == 0)
		return 0; 
	
	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0'; 
	
	return source; 
} 

#pragma mark -
#pragma mark Main OpenCL Routine
int runCL(float * a, float * results, int n)
{
	cl_program program[1];
	cl_kernel kernel[2];
	
	cl_command_queue cmd_queue;
	cl_context   context;
	
	cl_device_id cpu = NULL, device = NULL;

	cl_int err = 0;
	size_t returned_size = 0;
	size_t buffer_size;
	
	cl_mem a_mem, ans_mem;
	
#pragma mark Device Information
	{
		// Find the CPU CL device, as a fallback
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);
		assert(err == CL_SUCCESS);
		
		// Find the GPU CL device, this is what we really want
		// If there is no GPU device is CL capable, fall back to CPU
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (err != CL_SUCCESS) device = cpu;
		assert(device);
	
		// Get some information about the returned device
		cl_char vendor_name[1024] = {0};
		cl_char device_name[1024] = {0};
		err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
							  vendor_name, &returned_size);
		err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
							  device_name, &returned_size);
		assert(err == CL_SUCCESS);
		printf("Connecting to %s %s...\n", vendor_name, device_name);
	}
	
#pragma mark Context and Command Queue
	{
		// Now create a context to perform our calculation with the 
		// specified device 
		context = clCreateContext(0, 1, &device, NULL, NULL, &err);
		assert(err == CL_SUCCESS);
		
		// And also a command queue for the context
		cmd_queue = clCreateCommandQueue(context, device, 0, NULL);
	}
	
#pragma mark Program and Kernel Creation
	{
		// Load the program source from disk
		// The kernel/program is the project directory and in Xcode the executable
		// is set to launch from that directory hence we use a relative path
		//const char * filename = "/Volumes/markdbenedict/Episode_3_source/example.cl";
		const char * filename = "example.cl";
		char *program_source = load_program_source(filename);
		program[0] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
											   NULL, &err);
		assert(err == CL_SUCCESS);
		
		err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
		assert(err == CL_SUCCESS);
		
		// Now create the kernel "objects" that we want to use in the example file 
		kernel[0] = clCreateKernel(program[0], "set_wing_bnd", &err);
	}
		
#pragma mark Memory Allocation
	{
		// Allocate memory on the device to hold our data and store the results into
		buffer_size = sizeof(float) * n;
		
		// Input array a
		a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
		err = clEnqueueWriteBuffer(cmd_queue, a_mem, CL_TRUE, 0, buffer_size,
								   (void*)a, 0, NULL, NULL);
		assert(err == CL_SUCCESS);
		
		// Results array
		ans_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
		
		// Get all of the stuff written and allocated 
		clFinish(cmd_queue);
	}
	
#pragma mark Kernel Arguments
	{
		// Now setup the arguments to our kernel
		err  = clSetKernelArg(kernel[0],  0, sizeof(cl_mem), &a_mem);
		err |= clSetKernelArg(kernel[0],  1, sizeof(cl_mem), &ans_mem);
		assert(err == CL_SUCCESS);
	}
	
#pragma mark Execution and Read
	{
		// Run the calculation by enqueuing it and forcing the 
		// command queue to complete the task
		size_t global_work_size = n;
		err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, 
									 &global_work_size, NULL, 0, NULL, NULL);
		assert(err == CL_SUCCESS);
		clFinish(cmd_queue);
		
		// Once finished read back the results from the answer 
		// array into the results array
		err = clEnqueueReadBuffer(cmd_queue, ans_mem, CL_TRUE, 0, buffer_size, 
								  results, 0, NULL, NULL);
		assert(err == CL_SUCCESS);
		clFinish(cmd_queue);
	}
	
#pragma mark Teardown
	{
		clReleaseMemObject(a_mem);
		clReleaseMemObject(ans_mem);
		
		clReleaseCommandQueue(cmd_queue);
		clReleaseContext(context);
	}
	return CL_SUCCESS;
}

int testOpenCL(int probSize)
{
	
	// Problem size
	int n = probSize;
	
	// Allocate some memory and a place for the results
	float * a = (float *)malloc(n*sizeof(float));
	float * results = (float *)malloc(n*sizeof(float));
	
	// Fill in the values
	int i=0;
	for(i;i<n;i++){
		a[i] = 2*(float)i;
		printf("%f\n",a[i]);
		results[i] = 1.f;
	}
	
	// Do the OpenCL calculation
	runCL(a, results, n);
	int sum=0;
	// Print out some results. For this example the values of all elements
	// should be the same as the value of n
	i=0;
	for(i;i<n;i++) 
	{
		printf("a=%f , results= %f\n",a[i],results[i]);
		sum=sum+results[i];
	}
	
	// Free up memory
	free(a);
	free(results);
	
	return sum;
}






