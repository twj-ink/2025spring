#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

int maxCapacityPath(const unordered_map<int,vector<pair<int,int>>>& g, int s, int t, int n) {
    vector<int> capacity(n + 1, 0);
    vector<bool> visited(n + 1, false);
    capacity[s] = INF;
    
    priority_queue<pair<int,int>> pq;
    pq.push({INF, s});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (visited[u]) continue;
        visited[u] = true;
        
        if (u == t) return capacity[t];
        
        if (g.find(u) != g.end()) {
            for (const auto& p : g.at(u)) {
                int v = p.first;
                int w = p.second;
                if (!visited[v] && capacity[v] < min(capacity[u], w)) {
                    capacity[v] = min(capacity[u], w);
                    pq.push({capacity[v], v});
                }
            }
        }
    }
    return -1;
}
    

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t;
    cin>>t;
    for (int i = 0; i < t;i++){
        int op,x;
        cin>>op>>x;
        set<int> s;
        auto lo = s.lower_bound(x);
        auto it = s.begin();
        switch (op)
        {
        case 5:
            /* code */
            s.insert(x);
            break;
        case 1:
            if (s.size()>1) cout << distance(s.begin(), lo) + 2 << endl;
            else cout << distance(s.begin(), lo) + 1 << endl;
            break;
        case 2:
            auto it = s.begin();
            for (int i = 0;i<x-1;i++) {
                it++;
            }
            cout<<*it<<endl;
            break;

        }
    }

    // int n,m;
    // cin>>n>>m;
    // unordered_map<int,vector<pair<int,int>>> g;
    // for (int i = 0; i < m;i ++) {
    //     int u,v,w;
    //     cin>>u>>v>>w;
    //     g[u].push_back(make_pair(v,w));
    //     g[v].push_back(make_pair(u,w));
    // }
    // unordered_map<int,unordered_map<int,int>> d;
    // int q;
    // cin>>q;
    // for (int j = 0;j<q;j++) {
    //     int s,e;
    //     cin>>s>>e;
    //     if (d.find(s)!=d.end() && d[s].find(e)!=d[s].end()) {
    //         cout<<d[s][e]<<endl;
    //         continue;
    //     }
    //     int ans = maxCapacityPath(g,s,e,n);
    //     d[s][e]=ans;
    //     cout<<ans<<endl;
    // }
    
    return 0;
}