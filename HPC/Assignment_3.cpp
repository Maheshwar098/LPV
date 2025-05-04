#include <iostream>
#include <vector>
#include <chrono>
using namespace std; 
using namespace chrono ; 

vector<int>vector_add(vector<int>A, vector<int>B)
{
    int l = A.size();
    vector<int>C ; 
    for(int i = 0 ; i < l ; i++)
    {
        C.push_back(A[i] + B[i]);
    }
    return C ; 
}

void display_vector(vector<int>v)
{
    int l = v.size();
    for(int i = 0 ; i < l ; i++)
    {
        cout<< v[i] << "  ";
    }
    cout<<endl;
}

__global__ void  vector_add_parallel(int *X, int *Y, int *Z, int l)
{
    int tr_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(tr_id < l)
    {
        Z[tr_id]  = X[tr_id]  + Y[tr_id];
    }
}

int main()
{
    vector<int>A = {1,2,3,4};
    vector<int>B = {1,2,3,4};

    auto cpu_start = high_resolution_clock::now();
    vector<int>C = vector_add(A,B);
    auto cpu_stop = high_resolution_clock::now();

    duration<double, milli>elapsed_time = cpu_stop - cpu_start ; 
    cout<< "Sequential addition time : " << elapsed_time.count() << endl; 
    display_vector(C);

    // ==================================================================
    int l = A.size();
    int bytes_to_allocate = l * sizeof(int);
    vector<int>C2; 
    C2.resize(l);

    int *X, *Y, *Z ; 
    cudaMalloc(&X, bytes_to_allocate);
    cudaMalloc(&Y, bytes_to_allocate);
    cudaMalloc(&Z, bytes_to_allocate);

    cudaMemcpy(X, A.data(), bytes_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B.data(), bytes_to_allocate, cudaMemcpyHostToDevice);

    cudaEvent_t gpu_start, gpu_end ; 
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);
    vector_add_parallel<<<1,10>>>(X,Y,Z,l);
    cudaEventRecord(gpu_end);

    cudaMemcpy(C.data(), Z, bytes_to_allocate, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(gpu_end);

    float gpu_time_elapsed = 0 ; 
    cudaEventElapsedTime(&gpu_time_elapsed, gpu_start, gpu_end);
    cout<<"Parallel execution time : " << gpu_time_elapsed <<endl;
    display_vector(C);

}