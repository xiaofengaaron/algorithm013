## 总结

**高级动态规划**

动态规划步骤：

- divide&conquer（定义子问题（状态），并分别解决）
- optimal substructure（递推状态转移方程，合并子问题推出当前问题解）

难点：dp subproblem 难定义，状态转移方程难推

高级难点：1. subproblem 维度更多，二维、三维等 2. 状态方程更加复杂

**字符串算法**

高级算法：把string看作数组，问题与非字符场景无区别（DP常用）。一般DP的index是字符串index，value根据情况定义。例题：最长回文字串，正则表达式匹配、公共子串。

字符串匹配算法

- 暴力法：枚举起点后，用第二个字符串一次比较 O(N*M)
- Rabin-Karp：加入了哈希函数，如果哈希函数不同，直接不用比较；若相同，再逐一比较 O(N)
- KMP：若字符串不匹配时，则挪到字符串中首尾相同的最大位置，重新开始，加速了匹配过程。



## 题目



```python
# 最长上升子序列
# 时间复杂度 O(n^2)
# 空间复杂度 O(n)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # legal check
        if nums == []: return 0

        # init dp 
        dp = [0] * len(nums)

        # populate dp
        for i in range(len(nums)):
            tmp = 0
            for j in range(0,i):
                if nums[j] < nums[i]:
                    tmp = max(tmp, dp[j])
            dp[i] = tmp + 1
        # final result
        return max(dp)
```





```python
# 最长回文子串
# 时间复杂度 O(n^2)
# 空间复杂度 O(n^2)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # init dp and res
        res = ""
        dp = [[False]*len(s) for _ in s]

        # loop strings with two pointers with dp
        # dp[i][j] weather the substring is palindromic
        # populate dp
        for j in range(len(s)):
            for i in range(j+1):
                if s[i] == s[j] and (j-i<3 or dp[i+1][j-1]):
                    dp[i][j] = True
                    if len(res) < j-i+1:
                        res = s[i:j+1]
        return res 
```





```python
# 最长有效括号
# @lc code=start
# 时间复杂度 O(N)
# 空间复杂度 O(N)
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        if not s: return 0
        dp = [0] * len(s)
        for i in range(1,len(s)):
            if s[i] == ")" and s[i-1] == "(":
                dp[i] = dp[i-2]+2 if i>=2 else 2

            if s[i] == ")" and s[i-1] == ")" and i-dp[i-1]>=1:
                if s[i-dp[i-1]-1] == "(":
                    dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
        return max(dp
```



```python
# 字符串第一个唯一字符
# 时间复杂度 O(N)
# 空间复杂度 O(1) 因为字母是有限的26个
class Solution:
    def firstUniqChar(self, s: str) -> int:
        cnt = {}
        for c in s:
            if c not in cnt:
                cnt[c] = 1
            else:
                cnt[c] += 1
        for index in range(len(s)):
            if cnt[s[index]] == 1:
                return index
        return -1
```





```python
# 翻转字符串中的单词
# 时间复杂度 O(N)
# 空间复杂度 O(N)
class Solution:
    def reverseWords(self, s: str) -> str:
        # ini pointers
        left = 0
        right = len(s)-1

        # remove trailing zeros
        while s[left] == " ":
            left += 1
        while s[right] == " ":
            right -= 1
        
        # use stack to reverse word
        q = deque([]) 
        tmp = []
        while left <= right:
            if s[left] != " ":
                tmp.append(s[left])

            if (s[left] == " " or left==right) and tmp != []:
                q.appendleft("".join(tmp))
                tmp = []
            
            left += 1
        return " ".join(q)
```

