
__kernel void set_wing_bnd ( __global float* boundMask,__global float * x)
{
	int gid = get_global_id(0);
	if (boundMask[gid]<128.0)
	{
	   x[gid]=boundMask[gid];
	}
	

}