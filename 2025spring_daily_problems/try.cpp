#include <bits/stdc++.h>
using namespace std;

const int INF = 0x3f3f3f3f;

using Pair = pair<int,int>;
using vvi  = vector<vector<int>>;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n,m;
    cin >> n >> m;

    unordered_map<int, vector<Pair>> g;
    for (int i = 0; i < m; i++) {
        int u,v,w;
        cin>>u>>v>>w;
        g[u].push_back({v,w});
    }

    priority_queue<Pair, vector<Pair>, greater<Pair>> heap;
    heap.push({0,1});
    vector<int> dist(n + 1, INF);
    dist[1] = 0;

    while (true) {
        int d,u;
        auto p = heap.top(); heap.pop();
        d = p.first, u = p.second;
        if (u == n) {
            cout << d << endl;
            break;
        }
        if (d > dist[u]) {
            continue;
        }
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                heap.push({dist[v], v});
            }
        }
    }

    return 0;
}