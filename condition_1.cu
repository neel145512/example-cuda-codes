//Neel V Zadafiya (1115533)
//Condition_1.cu
//Runtime: Visual Studio (Using NVCC)
//Assignment 3 Part 2 - Matrix Multiplication using CUDA
//GPU: RTX 2080 Ti 11 GB

//Libraries for cuda runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Standard C libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Function for matrix multiplication
//C(m,w) = A(m,n) X B(n,w)
__global__ void matMul(int *c, int *a, int *b, int *m, int *n, int *w)
{
	//Get unique index of thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Get row and column value from flatten matrix
	int row = x / *w;
	int col = x % *w;
	
	//Calculate offset for result matrix
    int offset = row * *w + col;
	
	//Initialize sum to zero
	int sum = 0;
	
	//Loop to calculate actual multiplication
	for(int i =0; i<*n; i++)
	{
		//C[i][j] += A[i][k] * B[k][j] ; Innermost loop of standard matrix multiplication algorithm
		sum += a[(row * (*n))+i] * b[col+(i* (*w))];
	}
	
	//Store the sum in respective cell
	c[offset] = sum;
}

//Main function
int main()
{
	//Initialize start time
	clock_t startTime = clock();
	
	//C(m,w) = A(m,n) X B(n,w)
	//Initialize conditions
	int m = 500;
	int n = 500;
	int w = 400;
	int N = 100;
	
	//Initialize pointers for host array
    int *a;
	int *b;
	int *c;
	
	//Allocate host memory for above pointers
	a = (int *)malloc(m*n*sizeof(int));
	b = (int *)malloc(n*w*sizeof(int));
	c = (int *)malloc(m*w*sizeof(int));
	
	//Initialize random number generator
	srand(time(0));
	
	//Generate random numbers for B
	for(int i=0;i<n*w;i++)
	{
		b[i] = rand() % 10;
	}
	
	//Initialize pointers for device array and dimentions
	int *d_a, *d_b, *d_c;
	int *d_m, *d_n, *d_w;
	
	//Allocate memory to B in device
	cudaMalloc((void **)&d_b, n * w * sizeof(int));
	
	//Allocate memory for dimentions in device
	cudaMalloc((void **)&d_m, sizeof(int));
	cudaMalloc((void **)&d_n, sizeof(int));
	cudaMalloc((void **)&d_w, sizeof(int));
	
	//Copy values of dimentions from host to device
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
	
	//Copy values of B from host to device
	cudaMemcpy(d_b, b, n * w * sizeof(int), cudaMemcpyHostToDevice);
	
	//Core loop to iterate N
	for(int z = 0; z<N; z++)
	{
	
		//Generate random numbers for A
		for(int i=0;i<m*n;i++)
		{
			a[i] = rand() % 10;
		}
		
		//Allocate memory to A and C in device
		cudaMalloc((void **)&d_a, m * n * sizeof(int));
		cudaMalloc((void **)&d_c, m * w * sizeof(int));
		
		//Copy values of A from host to device
		cudaMemcpy(d_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
		
		//Call kernel function
		matMul<<<200,1000>>>(d_c, d_a, d_b, d_m, d_n, d_w);
		
		//Transfer results back to host memory
		cudaMemcpy(c, d_c, m * w * sizeof(int), cudaMemcpyDeviceToHost);
		
		//Free memory of A and C from device
		cudaFree(d_a);
		cudaFree(d_c);
		
		//Un-comment the code given below to print output of matrices on every iteration
		
		/*printf("===========================\n");
		printf("Iteration : %d\n\n",z);
		
		printf("Values of A:\n");
		for(int i=0;i<m;i++)
		{
			for(int j=0;j<n;j++)
			{
				printf("%d ",a[i*n+j]);
			}
			printf("\n");
		}
		
		printf("\nValues of B:\n");
		
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<w;j++)
			{
				printf("%d ",b[i*w+j]);
			}
			printf("\n");
		}
		
		printf("\nValues of C:\n");
		
		for(int i=0;i<m;i++)
		{
			for(int j=0;j<w;j++)
			{
				printf("%d ",c[i*w+j]);
			}
			printf("\n");
		}
		printf("\n");*/
		
	}
	
	//Free memory of B and dimentions from device
	cudaFree(d_b);
	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_w);

	//Initialize end time
	clock_t endTime = clock();
	
	//Print time taken by the program
	printf("Elapsed: %f seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	
	//Return 0 to finisdh the main function
    return 0;
}