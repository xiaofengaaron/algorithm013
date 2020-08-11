## 总结

1. 知识点

![Untitled Diagram](C:\Users\aaron\Downloads\Untitled Diagram.png)



2. 知识体会

- 哈希表：哈希表的查询非常快 O(1)，作为缓存非常高效
- 树：树可以让搜索变的更快，从链表的O(n) 进化到 二叉搜索树的 O(logN)，相当于升维了，空间换时间；遍历树一般使用递归helper函数
- 堆：适用于有优先级的取出场景；查询最大/最小值 O(1)，但是删除或者添加值因为要维护其堆的顺序，时间复杂度为 O(logN)；实现原理有多种，性能最好的是严格斐波那契堆，常考的为二叉堆（完全二叉树 + 父节点大于所有子节点）



## 题目

1. [有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/description/)（亚马逊、Facebook、谷歌在半年内面试中考过）

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 时间复杂度 O(n)
        # 空间复杂度 O(1)
        
        # 保证长度相同
        if len(s) != len(t):
            return False
        
        # 遇到s中+1，遇到y中-1
        temp = defaultdict(lambda: 0)
        for i in range(len(s)):
            temp[s[i]] += 1 
            temp[t[i]] -= 1
        
        # 检查是否每个元素值都为0
        for i in temp.values():
            if i != 0:
                return False
        return True
```

2. [前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        
        # 利用hashmap的dict计算频次 -- 时间 O(N)，空间 O(1)
        freq = defaultdict(lambda: 0)
        for i in nums:
            freq[i] += 1
        
        # 利用堆来获取前n个最大值 -- 时间 O(N*logK) 每次heap操作需要logK,一共有N个元素; 空间 O(N+k)
        return heapq.nlargest(k, freq, key = lambda x: freq[x])
```

3. [二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)（亚马逊、字节跳动、微软在半年内面试中考过）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

#  时间复杂度 O(n)  空间复杂度 O(n) 
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.helper(root, res)
        return res
    
    def helper(self, root:TreeNode, res):
        if root:
            if root.left:
                self.helper(root.left, res)
            res.append(root.val)
            if root.right:
                self.helper(root.right, res)
```

