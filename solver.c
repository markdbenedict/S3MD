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

         
void testCUDA()
{
	//callC();	
}
 
void runstep( int N,float * u, float * v, float * u0, float * v0, float* dens,float* dens_prev,float* boundMask, float visc, float dt ,float diff,float vel)
{
  	vel_step(N, u, v,u0,v0, boundMask, visc, dt,vel);
	dens_step(N,dens,dens_prev,u,v, boundMask, diff, dt,vel);
}


int main()
{
	printf("in main, about to callCUDA");
	testCUDA();
	return 0;
}

