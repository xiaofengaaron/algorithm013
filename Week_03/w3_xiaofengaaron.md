## 总结

1. 递归 

递归解决的是可以将一个问题拆成多个子问题，并且子问题解决过程是相同的。



递归的执行逻辑是：从顶部到底部执行，然后从底部再返回顶部，返回最终结果



写递归的时候需要有下面思路（有顺序）：

- 当前层逻辑 current logic

  - 你有什么选择 choices
  - 选择有什么限制条件 constraint

- 传递给下一层（参数的变化）drill down

- 终止条件 termination

  - 触底返回

  

2. 树的递归

写树的递归的时候，每一个node 就是一层

- 先关注当前node如果操作
- 向下传输值 drill down
- 向上传输值 return



3. 分治 / 回溯

- 分治 current logic 中涉及到多个 recursion
- 回溯 current logic 中涉及到多个 需要进行循环的 recursion，因为要循环，所以 参数需要 append 后再 pop



## 作业

1. [二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        # 时间复杂度 O(N), 空间复杂度 O(logN)
        
        # 如果root就是要找的节点，则返回root，向上传递；如果为空，返回None，无需下面继续操作
        if root in (None, p, q): return root
        
        # search drill down 下钻左右子树分别递归查找
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 如果左右都有节点返回，则是最近公共祖先；如果有一个返回，需要向上传递，继续查找
        if left or right: return left or right
        if left and right: return root
        
```



2. [组合](https://leetcode-cn.com/problems/combinations/)

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:

        # 时间复杂度 O(k*C_n^k)
        # 回溯的核心是，明确当前层需要做什么事情。
        # 此问题是每层 有一些 choices 需要尝试； 尝试完一个choice后传递给下一层，然再试下一个choice
        def bp(start, output):
            if len(output) == k:
                outputs.append(output[:])
            
            for i in range(start, n+1):
                output.append(i)
                bp(i+1, output)
                output.pop()
            
        outputs = []
        bp(start=1, output=[])
        
        return outputs
```



3. [全排列 II ](https://leetcode-cn.com/problems/permutations-ii/)

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 时间复杂度 O(N! * N)
        # 空间复杂度 O(N)
        def bp(ind, output):
            if len(output) == len(nums) and output not in outputs:
                outputs.append(output[:])
            
            # 每个格子的choice是：nums的元素，但未被选过的，也就是 not in ind
            # ind 记录被选过格子的index，output记录被选过元素的值
            for i in range(0,len(nums)):
                if i not in ind:
                    output.append(nums[i])
                    ind.append(i)
                    bp(ind, output)
                    output.pop()
                    ind.pop()
        outputs = []
        bp(ind = [] ,output =[])
        return outputs
    
```

