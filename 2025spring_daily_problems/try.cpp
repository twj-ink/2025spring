#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    char val;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
    TreeNode(char val_) : val(val_) {};
};

int idx = 0;
TreeNode* build(const string& s) {
    if (idx >= s.size()) {
        return nullptr;
    }
    if (s[idx] == '#') {
        return nullptr;
    }
    TreeNode* root = new TreeNode(s[idx]);
    idx++;
    root->left = build(s);
    idx++;
    root->right = build(s);
    return root;
}

string inos = "";
string poss = "";
vector<char> ans;

void ino(TreeNode* root) {
    if (root!=nullptr) {
        // cout << root->val << endl;
        ino(root->left);
        inos+=root->val;
        ino(root->right);
    }
}
void pos(TreeNode* root) {
    if (root!=nullptr) {
        pos(root->left);
        pos(root->right);
        poss+=root->val;
    }
}
void bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    // ans.push_back(root->val);
    while (!q.empty()) {
        auto node = q.front();q.pop();
        ans.push_back(node->val);
        if (node->left!=nullptr) {
            // ans.push_back(node->left->)
            q.push(node->left);
        }
        if (node->right!=nullptr) {
            q.push(node->right);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    string s;
    cin >> s;

    TreeNode* root = build(s);
    ino(root); pos(root);
    bfs(root);
    cout << inos << endl;
    cout << poss << endl;
    for (auto i:ans) cout<<i;
    cout << endl;
    // int t;
    // cin >> t;
    // while (t--) {
    //     solve();
    // }

    return 0;
}