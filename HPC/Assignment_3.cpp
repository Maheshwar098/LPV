#include <iostream>
#include <vector>
#include <map>
#include <stack>
#include <queue>
#include <omp.h>
using namespace std;

class Graph
{
private:
    vector<int> vertices;
    map<int, vector<int>> adj_list;

public:
    void add_vertex(int vertex)
    {
        this->vertices.push_back(vertex);
    }

    void add_edge(int vertex_1, int vertex_2)
    {
        adj_list[vertex_1].push_back(vertex_2);
        adj_list[vertex_2].push_back(vertex_1);
    }

    vector<int> dfs(int start_vertex)
    {
        vector<int> dfs_traversal;
        map<int, bool> is_visited;
        stack<int> st;
        st.push(start_vertex);

        while (!st.empty())
        {
            int top = st.top();
            st.pop();
            if (!is_visited[top])
            {
                dfs_traversal.push_back(top);
                is_visited[top] = true;

                vector<int> adj_elements = adj_list[top];
                int l = adj_elements.size();
                for (int i = 0; i < l; i++)
                {
                    if (!is_visited[adj_elements[i]])
                    {
                        st.push(adj_elements[i]);
                    }
                }
            }
        }

        return dfs_traversal;
    }

    vector<int> dfs_parallel(int start_vertex)
    {
        vector<int> dfs_traversal;
        map<int, bool> is_visited;
        stack<int> st;

        st.push(start_vertex);
        
        while (!st.empty())
        {
            int top = st.top();
            st.pop();
            
            if (!is_visited[top])
            {
                dfs_traversal.push_back(top);
                is_visited[top] = true;

                vector<int> adj_elements = adj_list[top];
                int l = adj_elements.size();

                #pragma omp parallel for
                for (int i = 0; i < l; i++)
                {
                    bool check_visited ; 
                    #pragma omp critical 
                    {
                        check_visited = is_visited[adj_elements[i]];
                    }

                    if (!check_visited)
                    {
                        #pragma omp critical 
                        {
                            st.push(adj_elements[i]);
                        }
                        
                    }
                }
            }
        }

        return dfs_traversal;
    }

    vector<int> bfs(int start_vertex)
    {
        vector<int> bfs_traversal;

        queue<int> q;
        q.push(start_vertex);
        map<int, bool> is_visited;

        while (!q.empty())
        {
            int front = q.front();
            q.pop();
            if (!is_visited[front])
            {
                bfs_traversal.push_back(front);
                is_visited[front] = true;

                vector<int> adj_elements = adj_list[front];
                int l = adj_elements.size();
                for (int i = 0; i < l; i++)
                {
                    if (!is_visited[adj_elements[i]])
                    {
                        q.push(adj_elements[i]);
                    }
                }
            }
        }

        return bfs_traversal;
    }

    vector<int> bfs_parallel(int start_vertex)
    {
        vector<int> bfs_traversal;

        queue<int> q;
        q.push(start_vertex);
        map<int, bool> is_visited;

        while (!q.empty())
        {
            int front = q.front();
            q.pop();
            if (!is_visited[front])
            {
                bfs_traversal.push_back(front);
                is_visited[front] = true;

                vector<int> adj_elements = adj_list[front];
                int l = adj_elements.size();
                #pragma omp parallel for
                for (int i = 0; i < l; i++)
                {
                    bool check_visited ; 
                    #pragma omp critical
                    {
                        check_visited = is_visited[adj_elements[i]];
                    }
                    if (!check_visited)
                    {
                        #pragma omp critical
                        {
                            q.push(adj_elements[i]);
                        }
                        
                    }
                }
            }
        }

        return bfs_traversal;
    }
};

void display_vector(vector<int> v)
{
    int l = v.size();
    for (int i = 0; i < l; i++)
    {
        cout << v[i] << "  ";
    }
    cout << endl;
}

int main()
{
    Graph g;
    g.add_vertex(1);
    g.add_vertex(2);
    g.add_vertex(3);
    g.add_vertex(4);
    g.add_vertex(5);

    g.add_edge(1, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 4);
    g.add_edge(3, 5);
    g.add_edge(4, 5);
    g.add_edge(2, 6);

    double start_time = omp_get_wtime();
    vector<int> graph_dfs = g.dfs(1);
    double end_time = omp_get_wtime();
    double duration = end_time - start_time ;   
    cout << "DFS : " << endl;
    display_vector(graph_dfs);
    cout << "Execution time :  " << duration << endl;

    start_time = omp_get_wtime();
    vector<int> graph_dfs_parallel = g.dfs_parallel(1);
    end_time = omp_get_wtime();
    duration = end_time - start_time ;   
    cout << "DFS parallel : " << endl;
    display_vector(graph_dfs_parallel);
    cout << "Execution time :  " << duration << endl;

    // -------------------------------------------

    start_time = omp_get_wtime();
    vector<int> graph_bfs = g.bfs(1);
    end_time = omp_get_wtime();
    duration = end_time - start_time ;  
    cout << "BFS : " << endl;
    display_vector(graph_bfs);
    cout << "Execution time :  " << duration << endl;

    start_time = omp_get_wtime();
    vector<int> graph_bfs_parallel = g.bfs_parallel(1);
    end_time = omp_get_wtime();
    duration = end_time - start_time ;  
    cout << "BFS parallel : " << endl;
    display_vector(graph_bfs_parallel);
    cout << "Execution time :  " << duration << endl;

    return 0;
}