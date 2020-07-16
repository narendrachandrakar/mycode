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



