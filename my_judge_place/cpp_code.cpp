#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <stack>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <set>
#include <numeric> //iota
#include <cmath> //pow->float

using namespace std;

using dictii = unordered_map<int, int>;
using vi = vector<int>;

const int MAXi = ~(1<<31);

bool can_find(int L, const vector<int>& a, int k) {
    const long long base = 1000003;
    const long long mod  = 1000000007;

    int n = a.size();
    if (L == 0) return true;

    long long power = 1;
    for (int i = 0; i < L; i++) power = (power * base) % mod;

    long long h = 0;
    for (int i = 0; i < L; i++) {
        h = (h * base + (a[i]+1)) % mod;
    }

    unordered_map<long long,int> cnt;
    cnt.reserve(n*2);
    cnt[h] = 1;

    for (int i = L; i < n; i++) {
        h = (h * base + (a[i]+1)) % mod;
        h = (h - (power * (a[i-L]+1)) % mod + mod) % mod;
        int c = ++cnt[h];
        if (c >= k) return true;
    }

    return false;
}



int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0);

    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // unordered_map<string, int> d;
    // for (int L = 1; L <= (n-1); L++) {
    //     for (int i = 0; i < n; i++) {
    //         int j = i + L - 1;
    //         if (j <= n - 1) {
    //             string sub = s.substr(i, L);
    //             // cout<<sub<<endl;
    //             d[sub]++;
    //         }
    //     }
    // }
    int lo = 1, hi = n, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (can_find(mid, a, k)) {
            ans = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    cout << ans << endl;

    // int ans = 0;
    // for (auto& [sub, cnt] : d) {
    //     if (cnt >= k) {
    //         ans = max(ans, (int)sub.size());
    //     }
    // }

    // cout << ans << endl;

    return 0;    
}