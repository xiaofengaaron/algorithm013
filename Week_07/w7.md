## 总结

Trie -- prefix数据类型

- 问题：自动补全，拼写检查，打字预测
- why not hashtable：1. hashtable适合找一个key，非多个 2. data size变大，collision 变多，time 为 O(n), n is number of keys inserted
- why Trie：1. 可以找多个keys 2. 快 O(m), m is key length

剪枝 -- 搜索优化

- 问题：递归过程中，重复的计算非常多，会造成 exponential 的时间复杂度
- memorization：将计算过的数据存下来，下次遇到后不再递归计算，而是O(1)直接查询
- 注意：作为递归的优化，一般也可以用DP来解决

并查集 union find

- 问题：已知两两元素关系，如何快速判断两个元素是否有关联。类似于facebook的用户网络
- 步骤：
  - 元素关联：使用一个数组，index 是自己的id，value 是指向的另一个元素的index
  - 初始化：每个元素的value都指向自己
  - union：基于两两元素的关系，建立union graph；找到每个元素的root并连接，root之间连接
  - find：每个元素找到root，比较root是否相同
  - 注意：对于组建的graph，可以进行路径优化；也就是找到root后，把路径中间的元素直接指向root

BFS 和 DFS 理解

- 二者都是先找到neighbour（neighbour 可以自己定义），然后neighbour的元素进行drill down 搜索
- BFS：在queue中，陆续把neighbour元素一起放进去，先处理第一个元素A，处理完，把A的neighbour放进对尾，再处理队头第二个元素B。【AB】--> 【B A1A2】--> 【A1A2 B1B2】
- DFS：在stack中，陆续把neighbour元素一起放进去，先处理最后的元素A，处理完，再把A的neighbour放进队尾，先处理最后的元素B，以此类推。【AB】--> 【A B1B2】--> 【A B1】
- 递归DFS：在recursion中，陆续对neighbour元素进行递归，递归有严格的顺序的，会先到达第一个neighbour元素递归的底部再一层一层返回。与Stack不同的是，recursion先处理最底层的一个元素；而stack是顺序处理，最后处理最底层的元素



## 题目



```python
# 实现Trie树
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        
	# 时间复杂度 O(m) m为word长度
    # 空间复杂度 O(m)
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for w in word:
            node = node.setdefault(w, {})
        node["#"] = "#"

	# 时间复杂度 O(m)
    # 空间复杂度 O(1)
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for w in word:
            if w not in node:
                return False
            node = node[w]
        return "#" in node
        
	# 时间复杂度 O(m)
    # 空间复杂度 O(1)
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for w in prefix:
            if w not in node:
                return False
            node = node[w]
        return True
```



```python
# 爬楼梯 -- 高级搜索
# 时间复杂度 O(n) -- size of regression tree
# 空间复杂度 O(n) -- depth of regression tree
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = [0] * (n+1)
        return self.climb_stairs(0, n, memo)

    def climb_stairs(self, cur, n, memo):
        # cur 位置到 n-stair 有多少种可能
        if cur > n: return 0
        if cur == n: return 1
        
        if memo[cur] == 0: 
            memo[cur] = self.climb_stairs(cur+1, n, memo)+ self.climb_stairs(cur+2, n, memo) 

        return memo[cur]

```



```python
# 朋友圈 -- 并查集
# 时间复杂度 O(n^3) -- matrix is n^2, unionfind is n
# 空间复杂度 O(n)
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        # 并查集初始化
        if not M: return 0
        n = len(M)
        p = [i for i in range(n)]
        
        # 合并
        def union(i,j):
            p1 = parent(i)
            p2 = parent(j)
            p[p1] = p2

        def parent(i):
            root = i
            while p[root] != root:
                root = p[root]
            while p[i] != i:
                i, p[i] = p[i], root
            return root
        
        # 使用 unionfind 查找
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    union(i,j)
        return len(set([parent(i) for i in range(n)]))
```



```python
# 岛屿的数量 -- 高级搜索
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # 特殊情况处理
        if grid == []: return 0
        # 初始化
        row = len(grid)
        col = len(grid[0])
        num = 0
        visited = set()

        # BFS logic
        def bfs(i,j):
            q = collections.deque()
            q.append((i,j))
            visited.add((i,j))
            while q:
                # 取出当前节点
                x0, y0 = q.popleft()
                
                # 加入孩子
                directions = [(0,-1), (0,1), (1,0), (-1,0)]
                for d in directions:
                    x,y = x0 + d[0], y0 + d[1]
                    if 0<=x<=row-1 and 0<=y<=col-1 and grid[x][y]=="1" and (x,y) not in visited: 
                        continue
                    q.append((x,y))
                    visited.add((x,y))

        # 遍历每个格子，并调用BFS
        for i in range(row):
            for j in range(col):
                if (i,j) not in visited and grid[i][j] == "1":
                    num += 1
                    bfs(i,j)
        return num
```



```python
# 括号生成 -- 高级搜索
# 时间复杂度 O(2^(2n))
# 空间复杂度 O(n) -- recursion的状态树深度为2n
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # backtracking
        open_cnt = 0
        close_cnt = 0
        result = []
        self.backtracking(result, "", open_cnt, close_cnt, n)
        return result

    def backtracking(self, result, cur_str, open_cnt, close_cnt, maxp):
        # termination
        if len(cur_str) == 2*maxp:
            result.append(cur_str)
            return 

        # drill down
        if close_cnt < open_cnt:
            self.backtracking(result, cur_str+")", open_cnt, close_cnt+1, maxp)

        if open_cnt < maxp:
            self.backtracking(result, cur_str+"(", open_cnt+1, close_cnt, maxp)
```



```python
# 单词接龙 -- 高级搜索
# 时间复杂度 O(N*M^2)
# 空间复杂度 O(N*M^2)
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # mask dict in wordlist
        if wordList == []: return 0
        word_dict = {}
        for word in wordList:
            for i in range(len(word)):
                masked_word = word[0:i] + "*" + word[i+1:]
                word_dict.setdefault(masked_word, [])
                word_dict[masked_word].append(word)

        # bfs search
        q = deque([(beginWord,1)])
        visited = set([])
        while q:
            word, level = q.popleft()
            for i in range(len(word)):
                masked_word = word[0:i]+"*"+word[i+1:]
                for next_word in word_dict.setdefault(masked_word,[]):
                    if next_word == endWord: 
                        return level + 1
                    if next_word not in visited:
                        visited.add(next_word)
                        q.append((next_word, level+1))
                word_dict[masked_word] = []
        return 0
```



```python
# N皇后 -- 高级搜索
# 时间复杂度 O(N!)
# 空间复杂度 O(N)
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def dfs(queens, xy_sum, xy_dif):
            x = len(queens)
            if x == n:
                results.append(queens)
                return 
            for y in range(n):
                if x+y not in xy_sum and x-y not in xy_dif and y not in queens:
                    bfs(queens+[y], xy_sum+[x+y], xy_dif+[x-y])

        results = []
        dfs([],[],[])
        return [["."*col + "Q" + "."*(n-col-1) for col in result] for result in results]
```



