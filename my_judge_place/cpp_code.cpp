#include <bits/stdc++.h>
using namespace std;
#define debug(...) [](auto...a){ ((cout << a << ' '), ...) << endl;}(#__VA_ARGS__, ":", __VA_ARGS__)
#define debugv(v) do {cout<< #v <<" : {"; for(int izxc=0;izxc<v.size();++izxc) {cout << v[izxc];if(izxc+1!=v.size()) cout << ","; }cout <<"}"<< endl;} while(0)
#define debugvv(vv) do {cout << #vv << " = {\n";for (auto &v : vv) {cout << "  { ";for (auto &x : v) cout << x << ' ';cout << "}\n";}cout << "}\n";} while(0)

void solve() {
    int n;
    cin >> n;
    vector<int> s(n);
    for (int i = 0; i < n; i++) cin >> s[i];
    
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    int t;
    cin >> t;
    // t=1;
    while (t--) {
        solve();
    }
    return 0;
}