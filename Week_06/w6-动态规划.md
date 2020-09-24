## 总结-动态规划

DP解决的问题

- 普通递归过程中，有大量的重复计算，时间复杂度通常是 exponential 级别的；这里最核心的问题是普通递归会从头一层一层的进行，多个递归自然重复的多；而DP是将原始问题分成了多个子问题，每个子问题依赖于其他子问题结果，这样每次计算就是【站在巨人的肩膀上】，而不是从头计算。

DP解决步骤

- 把大问题分解成有递推关系的多个子问题
- 递推计算并存储中间结果（dp矩阵或者dp数组），输出最后一个格子的结果

具体程序模板

- 建立DP矩阵
- 边界初始化 - 为了能启动DP方程
- DP方程在DP矩阵中进行递推计算
- 注意：每一个格子依赖前面的格子，可能根据当前值的情况不同计算公式也不同

## 题目

```python
# Longest Valid Parentheses
# 时间复杂度 O(N)
# 空间复杂度 O(N)
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        if not s: return 0
        
        # DP矩阵 + 边界初始化
        dp = [0] * len(s)
        
        # DP方程计算，基于不同场景（当前为")"时，前一个为"("或者")"）
        for i in range(1,len(s)):
            if s[i] == ")" and s[i-1] == "(":
                dp[i] = dp[i-2]+2 if i>=2 else 2

            if s[i] == ")" and s[i-1] == ")" and i-dp[i-1]>=1:
                if s[i-dp[i-1]-1] == "(":
                    dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
        return max(dp)

```



```python
# Edit Distance
# 时间复杂度 O(n*m)
# 空间复杂度 O(n*m)
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)

        if m == 0: return n
        if n == 0: return m

        # 建立dp矩阵
        dp = [[0]*(m+1) for _ in range(n+1)]
        
        # 边界初始化
        for i in range(n+1):
            dp[i][0] = i
        for i in range(m+1):
            dp[0][i] = i
        
        # 递推，word1前i个字符和word2前j个字符，改变1个字符前的最小的edit距离
        for i in range(1,n+1):
            for j in range(1, m+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]-1)
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[n][m]
```



