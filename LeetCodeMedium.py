# https://leetcode.com/problems/word-ladder/
# WAtch in fast mode https://www.youtube.com/watch?v=So1rCcavhS0
#https://www.youtube.com/watch?v=O3hUyXyHeBw

#
# Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:
#
# Only one letter can be changed at a time.
# Each transformed word must exist in the word list.

#
# Example 1:
#
# Input:
# beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"]
#
# Output: 5
#
# Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
# return its length 5.
# Example 2:
#
# Input:
# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log"]
#
# Output: 0
#
# Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.



from collections import defaultdict
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0


        # Since all words are of same length.
        L = len(beginWord)

        # Generate Adjacent

        # Dictionary to hold combination of words that can be formed,
        # from any given word. By changing one letter at a time.
        # A defaultdict works exactly like a normal dict, but it is initialized with a function (“default factory”) that takes no arguments and
        # provides the default value for a nonexistent key. A defaultdict will never raise a KeyError
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                # Key is the generic word
                # Value is a list of words which have the same intermediate generic word.
                tmp = word[:i] + "*" + word[i+1:]
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)


        # Queue for BFS
        import collections
        queue = collections.deque([(beginWord, 1)])
        # Visited to make sure we don't repeat processing same word.
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                # Intermediate words for current word
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]

                # Next states are all the words which share the same intermediate state.
                for word in all_combo_dict[intermediate_word]:
                    # If at any point if we find what we are looking for
                    # i.e. the end word - we can return with the answer.
                    if word == endWord:
                        return level + 1
                    # Otherwise, add it to the BFS Queue. Also mark it visited
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                all_combo_dict[intermediate_word] = []
        return 0

beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
obj = Solution()
#print(obj.ladderLength(beginWord, endWord, wordList))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# https://leetcode.com/problems/lru-cache/

#  Youtube: https://www.youtube.com/watch?v=kGlSZdDfm8M&list=PLfmc45xFf1Al9hzTTD7qBa5mk1mjuBY32&index=17&t=0s
# Youtube: https://www.youtube.com/watch?v=sDP_pReYNEc

class dll(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache(object):
    def __init__(self, capacity):
        self.head = dll(-1,-1)
        self.tail = self.head
        self.hash = {}
        self.capacity = capacity
        self.length = 0

    def get(self, key):
        if key not in self.hash:
            return -1

        node = self.hash[key]
        value = node.value

        while node.next:
            node.prev.next = node.next
            node.next.prev = node.prev

            self.tail.next = node
            node.prev = self.tail
            node.next = None
            self.tail = node
        return value

    def put(self, key, value):
        if key in self.hash:
            node = self.hash[key]
            node.value = value

            while node.next:
                node.prev.next = node.next
                node.next.prev = node.prev

                self.tail.next = node
                node.prev = self.tail
                node.next = None
                self.tail = node
        else:
            node = dll(key, value)
            self.hash[key] = node

            self.tail.next = node
            node.prev = self.tail
            node.next = None
            self.tail = node
            self.length += 1
            if self.length > self.capacity:
                remove = self.head.next
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
                del self.hash[key]
                self.length -= 1

# ++++++++++++++++++++++++++++++

class dll(object):

    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.head = dll(-1, -1)
        self.tail = self.head
        self.hash = {}
        self.capacity = capacity
        self.length = 0

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.hash:
            return -1
        node = self.hash[key]
        val = node.val
        while node.next:
            node.prev.next = node.next
            node.next.prev = node.prev
            self.tail.next = node
            node.prev = self.tail
            node.next = None
            self.tail = node
        return val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.hash:
            node = self.hash[key]
            node.val = value
            while node.next:
                node.prev.next = node.next
                node.next.prev = node.prev
                self.tail.next = node
                node.prev = self.tail
                node.next = None
                self.tail = node
        else:
            node = dll(key, value)
            self.hash[key] = node
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
            self.length += 1
            if self.length > self.capacity:
                remove = self.head.next
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
                del self.hash[remove.key]
                self.length -= 1

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# ++++++++++++++++++++++++++++++++++++++++
# Hard
# LFU Cache

# https://leetcode.com/problems/lfu-cache/
#https://leetcode.com/problems/lfu-cache/discuss/635047/Python-3-Hashmap-and-Double-Linked-List

# https://www.youtube.com/watch?v=rSwmpWaJPG0
# https://www.youtube.com/watch?v=su3E22YwLB4&list=PL5s8CPoNdSOuMtUw6iGIL_xfIna_vcPJa&index=39


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Partition Labels

# A string S of lowercase English letters is given. We want to partition this string into as many parts as possible
# so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

# Example 1:
#
# Input: S = "ababcbacadefegdehijhklij"
# Output: [9,7,8]
# Explanation:
# The partition is "ababcbaca", "defegde", "hijhklij".
# This is a partition so that each letter appears in at most one part.
# A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.

# Youtube: https://www.youtube.com/watch?v=ED4ateJu86I&t=314s

# Complexity Analysis
#
# Time Complexity: O(N), where NN is the length of SS.
#
# Space Complexity: O(1) to keep data structure last of not more than 26 characters.

def partitionLabels(S):
    d = {letter:index for index, letter in enumerate(S)}
    outline = []
    count = 0
    pos = 0
    for index, letter in enumerate(S):
        count += 1
        pos = max(pos, d[letter])
        if index == pos:
            outline.append(count)
            count = 0
    return outline


input = "ababcbacadefegdehijhklij"
#print(partitionLabels(input))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# https://leetcode.com/problems/merge-intervals/
# Merge Intervals
# Given a collection of intervals, merge all overlapping intervals.

# Youtube; https://www.youtube.com/watch?v=O44oSOfNvLM
#
# Example 1:
#
# Input: [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
# Example 2:
#
# Input: [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.


# Complexity Analysis
#
# Time complexity : O(nlog n)
#
# Other than the sort invocation, we do a simple linear scan of the list, so the runtime is dominated by the O(nlgn)O(nlgn)
# complexity of sorting.
#
# Space complexity : O(1) (or O(n)
#
# If we can sort intervals in place, we do not need more than constant additional space.
# Otherwise, we must allocate linear space to store a copy of intervals and sort that

def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    print(intervals[0][1])
    i = 1
    while i < len(intervals):
        if intervals[i][0] <= intervals[i - 1][1]:
            # intervals[i-1][0] = min(intervals[i-1][0], intervals[i][0])
            intervals[i - 1][1] = max(intervals[i - 1][1], intervals[i][1])

            intervals.pop(i)
        else:
            i += 1
    return intervals

input=[[1,4],[4,5]]
#print(merge(input))

#+++++++++++++++++++++++
# Meeting Rooms II
# https://leetcode.com/problems/meeting-rooms-ii/
# Youtube: https://www.youtube.com/watch?v=-WJFVe_1TRw

# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.
#
# Example 1:
#
# Input: [[0, 30],[5, 10],[15, 20]]
# Output: 2
# Example 2:
#
# Input: [[7,10],[2,4]]
# Output: 1

# Algorithm
#
# Sort the given meetings by their start time.
# Initialize a new min-heap and add the first meeting's ending time to the heap. We simply need to keep track of the ending times as that tells us when a meeting room will get free.
# For every meeting room check if the minimum element of the heap i.e. the room at the top of the heap is free or not.
# If the room is free, then we extract the topmost element and add it back with the ending time of the current meeting we are processing.
# If not, then we allocate a new room and add it to the heap.
# After processing all the meetings, the size of the heap will tell us the number of rooms allocated. This will be the minimum number of rooms needed to accommodate all the meetings

import  heapq
def meetingrooms(intervals):
    # If there is no meeting to schedule then no room needs to be allocated.
    if not intervals:
        return 0

    # The heap initialization
    free_rooms = []

    # Sort the meetings in increasing order of their start time.
    intervals.sort(key=lambda x: x[0])
    #print(intervals)

    #print(intervals[0][1])
    # Add the first meeting. We have to give a new room to the first meeting.
    heapq.heappush(free_rooms, intervals[0][1])
    #print(free_rooms)
    # For all the remaining meeting rooms
    for i in intervals[1:]:
        # If the room due to free up the earliest is free, assign that room to this meeting.
        if free_rooms[0] <= i[0]:
            #print("Pop", i[0])
            heapq.heappop(free_rooms)

        # If a new room is to be assigned, then also we add to the heap,
        # If an old room is allocated, then also we have to add to the heap with updated end time.
        heapq.heappush(free_rooms, i[1])

    # The size of the heap tells us the minimum rooms required for all the meetings.
    return len(free_rooms)

input = [[0, 30],[5, 10],[15, 20]]
#print(meetingrooms(input))


#++++++++++++++++++++++++++++++++++++++++++=
# https://leetcode.com/problems/subarray-sum-equals-k/solution/

# Subarray Sum Equals K

# Youtube: https://www.youtube.com/watch?v=bqN9yB0vF08https://www.youtube.com/watch?v=bqN9yB0vF08
#
# Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.
#
# Example 1:
#
# Input:nums = [1,1,1], k = 2
# Output: 2

# Approach 2: Using Cumulative Sum
# Algorithm
#
# Instead of determining the sum of elements everytime for every new subarray considered, we can make use of a cumulative sum array , sumsum. Then, in order to calculate the sum of elements lying between two indices, we can subtract the cumulative sum corresponding to the two indices to obtain the sum directly, instead of iterating over the subarray to obtain the sum.
#
# In this implementation, we make use of a cumulative sum array, sumsum, such that sum[i]sum[i] is used to store the cumulative sum of numsnums array upto the element corresponding to the (i-1)^{th}(i−1)
# th
#   index. Thus, to determine the sum of elements for the subarray nums[i:j]nums[i:j], we can directly use sum[j+1] - sum[i]sum[j+1]−sum[i].
#

#



def subarraySum( nums, k ):
    sumDict = {0: 1} # Watch youtbe
    count = 0
    sm = 0

    for num in nums:
        sm += num
        if sm - k in sumDict:
            count += sumDict[sm - k]
        sumDict[sm] = sumDict.get(sm, 0) + 1
    return count

nums = [3, 4, 7, 2, -3, 1, 4, 2, 1]
print(subarraySum(nums, 7))
