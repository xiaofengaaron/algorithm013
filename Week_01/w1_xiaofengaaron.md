## 总结

一、知识要点

1. 数组：search快 O(1)，因为有index。 add/delete 慢 O(n)，因为要元素copy并移动。
2. 链表：search慢 O(n)，因为要从头遍历。add/delete 快 O(1)，因为只需要更改指针地址。
3. 跳表：【有没有search快但是 add/del 也不慢的结构】？ 跳表空间换时间，利用多维索引来加速查询 O(logn)；但因结构复杂，每次add/delete 都需要重新维护结构，故 add/delete 时间也为 O(logn)
4. 栈 stack：后进先出，适用于【最近相关性】场景，比如 正反扩号问题
5. 队列 queue：先进先出，适用于【广度优先搜索】场景，或者有任务排队场景
6. 双端队列 deque：在Python中， stack 和 queue 都可以用 deque 来实现
7. 优先队列：第一个数据为max/min 元素，适用于【快速取出排名靠前元素】的操作，比如双十一会员优先，可以保证其请求快速得到响应。实现方式有多种，可以用堆heap来实现，取出高优元素O(1)，add/delete O(logn)，构造 O(n)



二、题目解法技巧

1. 重复性规律：根据题意，找到重复动作，并尝试简化。
2. 暴力遍历技巧：过程中设置缓存，可以加快速度（dict or hashmap）。
3. 链表技巧：关注需要更换位置的node，设置 node.next 指向对应node，一般需要 node 互换位置。
4. 指针移动：指针在当前位置进行操作后，再移动到下一个位置，进行相同操作（一般使用while循环，并常需要双指针和中间变量）。
5. 数字循环：取模 % 可以保证数字在预定范围内。
6. 



三、学习方法总结

1. 做题，利用遗忘曲线，在固定时间点，过遍数，来将短期记忆的只是变成长期记忆**神经元**
2. 将知识理解后，按照**树型**的总分结构整理归纳，也需要过遍数来加强记忆





## 题目

**简单**

1. 用 add first 或 add last 这套新的 API 改写 Deque 的代码

```java
import java.util.Deque;
import java.util.LinkedList;

public class deque_newapi {
    public static void main(String[] args) {
        Deque<String> deque = new LinkedList<String>();

        deque.addFirst("a");
        deque.addFirst("b");
        deque.addFirst("c");
        System.out.println(deque);

        String str = deque.peek();
        System.out.println(str);
        System.out.println(deque);

        while(deque.size() >0){
            System.out.println(deque.removeFirst());
        }
        System.out.println(deque);
    }
}
```



2. 分析 Queue 和 Priority Queue 的源码

- Queue 
  - 在python中的实现，直接调用的 Deque的功能 append 和 popleft

```python
    # Initialize the queue representation
    def _init(self, maxsize):
        ########################### 初始化使用了 deque来实现的（deque 是用C实现的）
        self.queue = deque()
    
    def _qsize(self):
        return len(self.queue)

    # Put a new item in the queue
    def _put(self, item):
        ########################### 加入元素，在list最后append
        self.queue.append(item)

    # Get an item from the queue
    def _get(self):
         ########################### 获取元素，在list前面pop出来
        return self.queue.popleft()
   
```

- PriorityQueue 
  - 使用的是heapq 中 heappush和 heappop实现的，二者都是利用 heap 堆来实现的
  - heappush
    - 先在heap尾部添加元素  heap.append(item)，然后再上浮 _siftdown(heap, 0, len(heap)-1)
  - heappop 
    - 代码注释

```python
class PriorityQueue(Queue):
'''Variant of Queue that retrieves open entries in priority order (lowest first).
Entries are typically tuples of the form:  (priority number, data).
'''
        def _init(self, maxsize):
            self.queue = []

        def _qsize(self):
            return len(self.queue)

        def _put(self, item):
            ########################### 添加元素，调用了heappush
            heappush(self.queue, item)

        def _get(self):
            ########################### 获取元素，调用了heappop
            return heappop(self.queue)


def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    ########################### 获取最后一个元素
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty

    if heap: 
        ########################### 返回堆顶端元素
        returnitem = heap[0] 
        ########################### 将最后一个元素替换成堆顶端，然后下沉，堆结构更新
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt
```



3. [删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)（Facebook、字节跳动、微软在半年内面试中考过）

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        cnt = 0
        for i in range(len(nums)-1):
            if nums[i] != nums[i+1]:
                cnt += 1
                nums[cnt] = nums[i+1]
        return cnt + 1
```



4. [旋转数组](https://leetcode-cn.com/problems/rotate-array/)（微软、亚马逊、PayPal 在半年内面试中考过）

```
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        # solution 1: 利用模来将多出来的元素插到前面
        n = len(nums)
        k = k % n
        a = [0] * n
        for i in range(n):
            a[(k+i) % n] = nums[i] 
        nums[:] = a
        '''
        
        # solution 2: 直接互换复制
        n = len(nums)
        k = k % n
        nums[:] = nums[n-k:] + nums[:n-k]
```



5. [合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)（亚马逊、字节跳动在半年内面试常考）

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        
        if l2 is None:
            return l1
        
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```



6. [合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)（Facebook 在半年内面试常考）

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 把nums1 作为母数组，利用双指针以此从后往前写入
        while n > 0 and m > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[n+m-1] = nums1[m-1]
                m-=1
            else:
                nums1[n+m-1] = nums2[n-1]
                n-=1
        if n > 0:
            nums1[:n] = nums2[:n]
```



7. [两数之和](https://leetcode-cn.com/problems/two-sum/)（亚马逊、字节跳动、谷歌、Facebook、苹果、微软在半年内面试中高频常考）

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        // 将查过的数据存在dict中，以便后续使用
        mydict = {}
        for i in range(0, len(nums)):
            if target - nums[i] in mydict.keys():
                return [mydict[target - nums[i]], i]
            else:
                mydict[nums[i]] = i
```



8. [移动零](https://leetcode-cn.com/problems/move-zeroes/)（Facebook、亚马逊、苹果在半年内面试中考过）

```
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 把所有非零元素前置
        j = 0
        for i in range(0, len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        # 把数组中剩下的值置零
        for i in range(j, len(nums)):
            nums[i] = 0
```



9. [加一](https://leetcode-cn.com/problems/plus-one/)（谷歌、字节跳动、Facebook 在半年内面试中考过）

```
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits
            
        # 如果每个数字都是9的话，再最前面加1，并返回
        return [1] + digits
```

**中等**

10. [设计循环双端队列](https://leetcode.com/problems/design-circular-deque)（Facebook 在 1 年内面试中考过）

```
class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.front = 0
        self.rear = 0
        self.size = 0
        self.capacity = k
        self.data = [0]*self.capacity
        

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        
        if self.isEmpty():
            self.data[self.front] = value
        else:
            self.front = (self.front - 1) % self.capacity
            self.data[self.front] = value
        self.size += 1
        return True
        

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        
        if self.isEmpty():
            self.data[self.rear] = value
        else:
            self.rear = (self.rear + 1 ) % self.capacity
            self.data[self.rear] = value
        self.size += 1
        return True

        
    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.data[self.front] = 0
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        if self.isEmpty():
            self.rear = self.front
        return True
        

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        
        self.data[self.rear] = 0
        self.rear = (self.rear - 1 ) % self.capacity
        self.size -= 1
        if self.isEmpty():
            self.front = self.rear
        return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.isEmpty():
            return -1
        return self.data[self.front]
        

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.isEmpty():
            return -1
        return self.data[self.rear]
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return self.size == 0
        
        

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return self.size == self.capacity
        
```

**困难**

11. [接雨水](https://leetcode.com/problems/trapping-rain-water/)（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）

```
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right, left_max, right_max = 0,len(height)-1,0,0
        rain = 0
        
        ###### 双指针解法，左右指针分别往中间移动，值小的移动（保证两边往最高的柱子靠拢）
        ###### 指针走过的范围维护一个 历史max，比历史max小的都会积水，大的更新max值
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    rain += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    rain += right_max - height[right]
                right -= 1
            
        return rain
```

