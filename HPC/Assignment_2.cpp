#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void swap (int &a , int &b)
{
    int c = a ; 
    a = b ; 
    b = c  ;
}

vector<int> bubble_sort(vector<int>v)
{
    int l = v.size();
    for(int i = 0 ; i < l ; i++)
    {
        for(int j = 0 ; j < l-i-1 ; j++)
        {
            if(v[j] > v[j+1])
            {
                swap(v[j], v[j+1]);
            }
        }
    }
    return v;
}

vector<int> bubble_sort_parallel(vector<int>v)
{
    int l = v.size();

    for(int i = 0 ; i < l  ; i++)
    {
        int start = 0 ; 
        if(i%2 == 1)
        {
            start = 1 ;
        }

        #pragma omp parallel for
        for(int j = start ; j < l-1 ; j+=2)
        {
            if(v[j] > v[j+1])
            {
                swap(v[j], v[j+1]);
            }
        }
    }

    return v;
}

void display_vector(vector<int>v)
{
    int l = v.size();
    for(int i = 0 ; i < l ; i++)
    {
        cout<<v[i]<<"   ";
    }
    cout<<endl;
}

void merge(vector<int>&v, int start, int mid, int end)
{
    int i = start ; 
    int j = mid+1 ; 

    vector<int> temp ; 

    while(i<=mid && j <= end)
    {
        if(v[i] <= v[j])
        {
            temp.push_back(v[i]);
            i++;
        }
        else
        {
            temp.push_back(v[j]);
            j++;
        }
        
    }

    while(i<=mid)
    {
        temp.push_back(v[i]);
        i++;
    }

    while(j<=end)
    {
        temp.push_back(v[j]);
        j++;
    }

    for(i = start ; i <= end ; i++)
    {
        v[i] = temp[i-start];
    }


}

void merge_sort_util(vector<int>& v, int start, int end)
{
    if(end<=start)
    {
        return ; 
    }
    int mid = (start + end) / 2; 
    merge_sort_util(v,start, mid);
    merge_sort_util(v, mid+1, end);

    merge(v, start, mid, end);
}

vector<int>merge_sort(vector<int>v)
{
    int l = v.size();
    int start = 0 ;
    int end = l-1;

    merge_sort_util(v,start,end);

    return v;
}

void merge_sort_util_parallel(vector<int>& v, int start, int end)
{
    if(end<=start)
    {
        return ; 
    }
    int mid = (start + end) / 2; 
    #pragma omp parallel sections
    {
        #pragma omp section
        merge_sort_util_parallel(v,start, mid);

        #pragma omp section
        merge_sort_util_parallel(v, mid+1, end);
    }
    

    merge(v, start, mid, end);
}

vector<int>merge_sort_parallel(vector<int>v)
{
    int l = v.size();
    int start = 0 ;
    int end = l-1;

    merge_sort_util_parallel(v,start,end);

    return v;
}

vector<int>generate_vector(int size)
{
    vector<int>v ; 
    for(int i = 0 ; i < size ; i++)
    {
        v.push_back(rand() % 100 + 1);
    }

    return v;
}

int main()
{

    int vector_size = 10;
    vector<int>v=generate_vector(vector_size);

    double start_time = omp_get_wtime();
    vector<int>bubble_sorted_v = bubble_sort(v);
    double end_time = omp_get_wtime();
    double time_elapsed = end_time - start_time ; 
    display_vector(bubble_sorted_v);
    cout<< "Sequential Bubble sorting time : "<< time_elapsed <<endl;

    start_time = omp_get_wtime();
    vector<int>parallel_bubble_sorted_v = bubble_sort_parallel(v);
    end_time = omp_get_wtime();
    time_elapsed = end_time - start_time ; 
    display_vector(parallel_bubble_sorted_v);
    cout << "Parallel Bubble sorting time : " << time_elapsed << endl;

    cout<<"--------------------------------------------------"<<endl;

    // ==========================================================

    start_time = omp_get_wtime();
    vector<int>merge_sorted_v = merge_sort(v);
    end_time = omp_get_wtime();
    time_elapsed = end_time - start_time ; 
    display_vector(merge_sorted_v);
    cout<< "Sequential merge sorting time : " << time_elapsed<<endl;

    start_time = omp_get_wtime();
    vector<int>parallel_merge_sorted_v = merge_sort_parallel(v);
    end_time = omp_get_wtime();
    time_elapsed = end_time - start_time ; 
    display_vector(parallel_merge_sorted_v);
    cout<< "Parallel merge sorting time : " << time_elapsed<<endl;


    return 0 ; 

}