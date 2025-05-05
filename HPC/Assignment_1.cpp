#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int calculate_sum(vector<int>v)
{
    int l = v.size();
    int sum = 0 ; 
    for(int i = 0 ; i < l ; i++)
    {
        sum = sum + v[i];
    }
    return sum ; 
}

float calculate_avg(vector<int>v)
{
    int sum = calculate_sum(v);
    int l = v.size();
    float avg = (float)(sum) / (float)(l);
    return avg;
}

int calculate_min(vector<int>v)
{
    int min_val = v[0];
    int l = v.size();
    for(int i = 0 ; i < l ; i++)
    {
        min_val = min(v[i], min_val);
    }
    return min_val;
}

int calculate_max(vector<int>v)
{
    int max_val = v[0];
    int l = v.size();

    for(int i = 0 ; i < l ; i++)
    {
        max_val = max(v[i], max_val);
    }
    return  max_val;
}

// =======================================================
int calculate_sum_parallel(vector<int>v)
{
    int l = v.size();
    int sum = 0 ; 

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0 ; i < l ; i++)
    {
        sum = sum + v[i];
    }
    return sum ; 
}

float calculate_avg_parallel(vector<int>v)
{
    int sum = calculate_sum_parallel(v);
    int l = v.size();
    float avg = (float)(sum) / (float)(l);
    return avg;
}

int calculate_min_parallel(vector<int>v)
{
    int min_val = v[0];
    int l = v.size();
    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0 ; i < l ; i++)
    {
        min_val = min(v[i], min_val);
    }
    return min_val;
}

int calculate_max_parallel(vector<int>v)
{
    int max_val = v[0];
    int l = v.size();
    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0 ; i < l ; i++)
    {
        max_val = max(v[i], max_val);
    }
    return  max_val;
}

int main()
{
    vector<int> v{4,2,1,7,5,6};

    double start_time = omp_get_wtime();
    int sum = calculate_sum(v);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time ; 
    cout << "Sequential sum calculation time : " << elapsed_time << endl;

    start_time = omp_get_wtime();
    int sum_parallel = calculate_sum_parallel(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Parallel sum calculation time : " << elapsed_time << endl;

    cout << "Sum : "<< sum << endl; 
    cout<<"---------------------------------"<<endl;
    
    // ============================================================

    start_time = omp_get_wtime();
    float avg = calculate_avg(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Sequential avg calculation time : " << elapsed_time << endl;

    start_time = omp_get_wtime();
    float avg_parallel = calculate_avg_parallel(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Paralle avg calculation time : " << elapsed_time << endl;

    cout << "Average : " << avg << endl;
    cout<<"---------------------------------"<<endl;

    // ============================================================

    start_time = omp_get_wtime();
    int minimum_val = calculate_min(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Sequential min calculation time : " << elapsed_time << endl;

    start_time = omp_get_wtime();
    int minimum_val_parallel = calculate_min_parallel(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Parallel min calculation time : " << elapsed_time << endl;

    cout << "Minimum value : " << minimum_val << endl;
    cout<<"---------------------------------"<<endl;
    // ============================================================

    start_time = omp_get_wtime();
    int maximum_val = calculate_max(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Sequential max calculation time : " << elapsed_time << endl;

    start_time = omp_get_wtime();
    int maximum_val_parallel = calculate_max_parallel(v);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time ; 
    cout << "Parallel max calculation time : " << elapsed_time << endl;
    cout << "Maximum value : " << maximum_val << endl;
    cout<<"---------------------------------"<<endl;
    // ============================================================


    return 0 ; 
}