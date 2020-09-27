## 总结

位运算

- 主要用法：1. 二进制代替数组来追踪状态 2. 通过位运算来操作bit的值更新状态
- 主要操作符： >> << & | ~ ^ 依次为 右移、左移、与、或、非、异或



布隆过滤器&LRU cache 

- Bloom Filter
  - 与哈希表区别：
    - 相同：都可以判断元素是否在集合中
    - 不同：但哈希表还可以查询value相关信息；效率比哈希表高，因为bf是模糊判断（可判断不在，无法100%判断在）
  - 应用：比特币网络（transaction 是否在node里） / 分布式系统 Hadoop、search engine / Redis / 垃圾邮件、评论等过滤
- LRU cache
  - 一个有序且有限的缓存（可用orderedDict实现），新使用过的排在最前面，如果超出了capacity的限制，删除最不常用的 least recently used (LRU)

排序算法

- O(N^2)
  - 冒泡排序：两两比较
  - 选择排序：选择最小的互换
  - 插入排序：将当前值插入有序序列
  - 
- O(logN)
  - 快速排序：选择一个pivot，把元素小于pivot的放在左边，大于pivot的放在右边，递归重复操作
  - 归并排序：选择中间元素，把原始长度为n数组递归分成n份，利用递归再合并，合并时候完成排序
  - 堆排序：利用第一个元素是最小值的性质，依次从数组中取出最小值



## 题目

```python
# 力扣排行榜
# top函数 时间复杂度 O(N*logN) N为 number of players
# 空间复杂度 O(N)
class Leaderboard:
    
    def __init__(self):
        self.board = {}

    def addScore(self, playerId: int, score: int) -> None:
        self.board.setdefault(playerId, 0)
        self.board[playerId] += score

    def top(self, K: int) -> int:
        return sum([v for k, v in sorted(self.board.items(), key=lambda item: item[1], reverse=True)][0:K])

    def reset(self, playerId: int) -> None:
        self.board[playerId] = 0
        
```



```python
# LRU 缓存机制
# 时间复杂度 put O(1)
# 空间复杂度 put O(capacity)

from collections import OrderedDict
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.cache.keys():
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        
```





```python
# 排序算法 -- quick search
# quick sort
def quick_sort(arr, low, high):
	if len(arr) == 1: return arr
	# partition--sorting
	if low < high:
		pivot_index = partition(arr, low, high)

		# recursion for each partition
		quick_sort(arr, low, pivot_index-1)
		quick_sort(arr, pivot_index+1, high)
	return arr

def partition(arr, low, high):
	# pointer to swap with pivot
	i = low-1
	# pivot
	pivot_value = arr[high]
	for j in range(low, high):
		if arr[j] < pivot_value:
			i += 1
			arr[i], arr[j] = arr[j] , arr[i]
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i+1
```



```python
# 排序算法 -- merge search

# merge sort
def merge_sort(arr):
	# 递归将arr分成两个一组
	n = len(arr)
	if n <= 1: return arr
	mid = n // 2
	left = arr[0:mid]
	right = arr[mid:]
	return merge(merge_sort(left), merge_sort(right))

def merge(left, right):
	l, r, left_len, right_len = 0, 0, len(left), len(right)

	result = []
	while l < left_len and r < right_len:
		if left[l] < right[r]:
			result.append(left[l])
			l += 1
		else:
			result.append(right[r])
			r += 1
	while l < left_len:
		result.append(left[l])
		l += 1
	while r < right_len:
		result.append(right[r])
		r += 1
	return result
```



```python
# 插入排序 -- insertion sort

def insertion_sort(arr):
	n = len(arr)
	for i in range(1,n):
		pre_index = i-1
		current = arr[i]
		while pre_index >= 0 and arr[pre_index] > current:
			arr[pre_index+1] = arr[pre_index]
			pre_index -= 1
		arr[pre_index+1] = current
	return arr
```



```python
# 选择排序 
def selection_sort(arr):
	n = len(arr)
	for i in range(n-1):
		min_index = i
		for j in range(i+1,n):
			if arr[j] < arr[min_index]:
				min_index = j
		arr[min_index], arr[i] = arr[i], arr[min_index]
	return arr

```



```python
# 冒泡排序
def bubble_sort(arr):
	n = len(arr)
	for i in range(0,n-1):
		for j in range(0,n-1-i):
			if arr[j] > arr[j+1]:
				arr[j], arr[j+1] = arr[j+1], arr[j]
	return arr
```



