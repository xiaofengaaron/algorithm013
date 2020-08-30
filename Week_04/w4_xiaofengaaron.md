## 总结

1. DFS / BFS

DFS 的步骤（递归-遵循递归基本步骤）-- 递归是天然自上而下的

- 终止条件：如果访问到了最底层 None
- 当前层的选择 + 限制条件：若未被访问过，访问当前节点，操作是直接 append / add
- 基于当前的选择 drill down 到更深层 - recursion



DFS 非递归步骤

- idea：维护一个stack，后进先处理；每处理当前element的时候，会扩展出更多children，穷尽了所有children才会，返回处理水平的元素
- while stack: 当stack还有元素的时候，一直处理
- pop 出当前处理元素（加入visited）
- 得到当前元素children
- 把 children push 进去



BFS步骤

- idea: 维护一个队列，先进先出，每处理当前元素的时候，会把当前要处理的元素放在队尾，保证水平元素可以依次优先处理。
- while queue: 当queue 还有元素的时候，一直处理
- pop 当前元素（加入visited）
- 得到当前元素的children
- 把 children push 进去



2. 贪心算法

贪心算法基于当前的状态，寻找出对当前最优的方案，然后顺序遍历，时间空间复杂度最优；但是只限于解决部分可以证明的问题。



3. 二分查找

查找条件：1. 有序 2. 有上下界 3. 能够通过索引访问

具体步骤

- 若无需，进行排序
- 将上下界分别存储 left, right
- while left <= right: 
  - 通过索引获得 mid 值
  - 比较 target 在 mid 左边还是右边
  - 更新 left or right 



## 作业

1. [分发饼干](https://leetcode-cn.com/problems/assign-cookies/description/)

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        
        # O(NlogN + MlogM), N = #ofChildern M = #ofCookie
        g.sort() # children
        s.sort() # cookie
        
        gi = 0
        si = 0
        # cookie 一直循环，如果cookie 满足 kid，那么 gi + 1， 相当于 满足条件的 kid number
        while gi < len(g) and si < len(s):
            if g[gi] <= s[si]:
                gi += 1
            si += 1
        return gi
```



2. [模拟行走机器人](https://leetcode-cn.com/problems/walking-robot-simulation/description/)

```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        
        # 时间复杂度 O(N + K)
        # 空间复杂度 O(K)
        
        # 设置初始值
        x, y, res = 0, 0, 0
        
        # 设置方向映射 # east, north, west, south
        dx = [1, 0, -1, 0] 
        dy = [0, 1, 0, -1] 
        cur_dir = 1
        
        # 设置障碍映射
        obs = set(map(tuple, obstacles))
        
        # 基于输入值，改变方向；基于方向，改变坐标
        for cmd in commands:
            # 改变方向
            if cmd == -2:
                cur_dir = (cur_dir + 1) % 4
            elif cmd == -1:
                cur_dir = (cur_dir + 3 ) % 4 
            else:
                # 基于方向改变坐标		
                for _ in range(cmd):
                    xy_next = (x + dx[cur_dir], y + dy[cur_dir])
                    if xy_next in obs:
                        break
                    x, y = xy_next
                    
                res = max(res, (x**2 + y**2))
        return res
```



3. [买卖股票的最佳时机 II ](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/description/)

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 时间复杂度 O(N) 
        # 空间复杂度 O(1)
        max_profit = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        return max_profit
```



4. [岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        # 时间复杂度 O(N*M) N-number of rows, M-number of columns
        # 空间复杂度 O(N*M) 
        if not grid or not grid[0]: return 0
        
        row, col = len(grid), len(grid[0])
        islands = 0
        visited = set()
        
        def dfs(r, c):
            if r not in range(row) or c not in range(col) or grid[r][c]=="0" or (r,c) in visited:
                return
            
            visited.add((r,c))
            
            for i in [(1,0),(-1,0),(0,1),(0,-1)]:
                dfs(r+i[0], c+i[1])
            
        for i in range(row):
            for j in range(col):
                if (i,j) not in visited and grid[i][j] == "1":
                    islands += 1
                    dfs(i,j)
                   
        return islands 
```



5. [搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        # 时间复杂度 O(log(n*m))
        # 空间复杂度 O(1)
        # binary search bound
        m = len(matrix)
        if m == 0: return False
        n = len(matrix[0])

        left, right = 0, n*m -1
        
        # search
        # 持续判断 mid 与 target 的大小，最差情况 target = left/right; 最后一步才会分辨出
        while left <= right:
            mid = (right + left) // 2
            mid_row, mid_col = mid // n, mid % n
            
            if target == matrix[mid_row][mid_col]:
                return True
            elif target < matrix[mid_row][mid_col]:
                right = mid - 1 
            else:
                left = mid + 1 
        
        return False
```

