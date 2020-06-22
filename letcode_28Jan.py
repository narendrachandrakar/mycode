#https://github.com/kamyu104/LeetCode/blob/master/Python/two-sum.py
#https://leetcode.com/problems/two-sum/description/


class Solution(object):
    def twoSum(self, nums, target):
        lookup = {}
        for i, num in enumerate(nums):
            if target - num in lookup:
                return [lookup[target -num], i]
            lookup[num] = i
        return None



# if __name__ == "__main__":
#     nums = [2,7,11,15]
#     target = 9
#     print(Solution().twoSum(nums, target))


# 3sum.py
# https://www.geeksforgeeks.org/find-a-triplet-that-sum-to-a-given-value/
#https://leetcode.com/problems/3sum/

#  Great explain  https://www.geeksforgeeks.org/find-a-triplet-that-sum-to-a-given-value/
# 1.Sort the given array.
# 2.Loop over the array and fix the first element of the possible triplet, arr[i].
# 3.Then fix two pointers, one at i + 1 and the other at n – 1. And look at the sum,
# a.If the sum is smaller than the required sum, increment the first pointer.
# b.Else, If the sum is bigger, Decrease the end pointer to reduce the sum.
# c. Else, if the sum of elements at two-pointer is equal to given sum then print the triplet and break.
#

def sum3(arr, sum):
    arrSize = len(arr)
    arr.sort()
    r = arrSize - 1
    for i in range(0, arrSize - 2):
        l = i + 1

        while (l < r):
            result = arr[i] + arr[l] + arr[r]
            if (result == sum):
                print(arr[i], arr[l], arr[r])
                return True
            if (result < sum):
                l += 1
            else:
                r -= 1
        return False


arr = [1, 4, 45, 6, 10, 8]
sum = 13

#print(sum3(arr, sum))

nums = [-1, 0, 1, 2, -1, -4]


def threeSum(nums):
    nums.sort()
    rst, leng, target = [], len(nums), 0
    for i in range(leng - 2):
        # this is very elegant to avoid repeated case.
        if i == 0 or (nums[i] != nums[i - 1]):
            if nums[i] + nums[i + 1] + nums[i + 2] > target: break
            if nums[i] + nums[-1] + nums[-2] < target: continue
            curtar = target - nums[i]
            l, r = i + 1, leng - 1
            while l < r:
                cursum = nums[l] + nums[r]
                if cursum > curtar:
                    r -= 1
                elif cursum < curtar:
                    l += 1
                else:
                    rst.append((nums[i], nums[l], nums[r]))
                    # this is actually not correct, it should be l + 1 < r, the triple version is bug-free
                    while l < r and nums[l + 1] == nums[l]: l += 1
                    while l < r and nums[r - 1] == nums[r]: r -= 1
                    l, r = l + 1, r - 1
    return rst

# https://leetcode.com/problems/3sum/discuss/624953/Python3-Solution

def threeSum11(nums):
        nums.sort()
        arr = []

        for idx in range(len(nums) - 2): # beacuse two-pointer:l=i+1 and r=len(nums)-1
            if ((idx == 0) or\
                    (idx > 0 and nums[idx] != nums[idx - 1])):
                l = idx + 1
                r = len(nums) - 1

                while (l < r):
                    sum_t = nums[idx] + nums[l] + nums[r]
                    if sum_t == 0:
                        arr.append([nums[idx], nums[l], nums[r]])
                        # skip duplicate value
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r - 1] == nums[r]:
                            r -= 1
                        l += 1
                        r -= 1

                    elif sum_t < 0:
                        l += 1
                    else:
                        r -= 1
        return arr
#print(threeSum11(nums))

#+++++++++++++++++++++++++++++++
#https://leetcode.com/problems/reverse-string/
# Write a function that reverses a string. The input string is given as an array of characters char[].
#
# Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
# Example 1:
#
# Input: ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]


def reverseString(s):
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s)-1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left, right = left + 1, right - 1
        print(s)

s=["h","e","l","l","o"]
#(reverseString(s))

#+++++++++++++++++++++++++++++++++++
#Maximum Depth of Binary Tree

# https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Given a binary tree, find its maximum depth.
#
# The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
#
# Note: A leaf is a node with no children.
#
# Example:
#
# Given binary tree [3,9,20,null,null,15,7],
#
#     3
#    / \
#   9  20
#     /  \
#    15   7

#  1 + max( left + right )

# Video : https://www.youtube.com/watch?v=to2XMEXE1ms

class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        else:
            return 1  + max(self.maxDepth(root.left), self.maxDepth(root.right))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Single Number
#
# Given a non-empty array of integers, every element appears twice except for one. Find that single one.
#
# Note:
#
# Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
#
# Example 1:
#
# Input: [2,2,1]
# Output: 1
# Example 2:
#
# Input: [4,1,2,1,2]
# Output: 4

# Youtube
#https://www.youtube.com/watch?v=r0CAz6MdgEg
#watch after 3:40

# 1. Bit Manipulation

# XOR operation
# 0 ^ 0 = 0
# 0 ^ 1 = 1
# 1 ^ 0 = 1
# 1 ^ 1 = 0
# 0000
# 0001
# 0010
# 0011
# 0100
# 0101

# Time complexity : O(n). We only iterate through \text{nums}nums,
# so the time complexity is the number of elements in \text{nums}nums.
#
# Space complexity : O(1).

def singleNumber(nums):
    ans = 0
    for i in nums:
        ans ^= i
    return ans

nums = [4,1,2,1,2]
#print(singleNumber(nums))

# 2 : Hash Table

# Algorithm
#
# We use hash table to avoid the O(n) time required for searching the elements.
#
# Iterate through all elements in nums and set up key/value pair.
# Return the element which appeared only once.

def hashSingleNumber(nums):
    from collections import defaultdict

    hashTable = defaultdict(int)

    for i in nums:
        hashTable[i] += 1

    for i in hashTable:
        if hashTable[i] == 1:
            return i
#print(hashSingleNumber(nums))
#
# Complexity Analysis
#
# Time complexity : O(n - 1) = O(n). Time complexity of for loop is O(n).
# Time complexity of hash table(dictionary in python) operation pop is O(1).
#
# Space complexity : O(n). The space required by hash\_table hash_table is equal to the number
# of elements in \text{nums}nums.


# ++=================================================================================

def fizzBuzz(n):
    result = []

    for i in range(1, n+1):
        if i % 15 == 0:
            result.append("fizzBuzz")
        elif i % 5 == 0:
            result.append("Buzz")
        elif i % 3 == 0:
            result.append("Fizz")
        else:
            result.append(str(i))
    return result

#print(fizzBuzz(1))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# https://leetcode.com/problems/reverse-linked-list/
# Youtube: https://www.youtube.com/watch?v=XDO6I8jxHtA&frags=pl%2Cwn

# Reverse a singly linked list.
#
# Example:
#
# Input: 1->2->3->4->5->NULL
# Output: 5->4->3->2->1->NULL

class Solution:
    def reverseList(self, head):
        prev = None
        while head:
            tmp = head
            head = head.next
            tmp.next = prev
            prev = tmp
        return prev

# Complexity analysis
#
# Time complexity : O(n). Assume that n is the list's length, the time complexity is O(n).
#
# Space complexity : O(1)
#

#+++++++++++++++++++++++++++++++++++=
#https://leetcode.com/problems/majority-element/

# # Brute Force algorithm
# Algorithm
#
# The brute force algorithm iterates over the array, and then iterates again for each number to count its occurrences.
# As soon as a number is found to have appeared more than any other can possibly have appeared, return it.


# Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
#
# You may assume that the array is non-empty and the majority element always exist in the array.
#
# Example 1:
#
# Input: [3,2,3]
# Output: 3
# Example 2:
#
# Input: [2,2,1,1,1,2,2]
# Output: 2

def majorityElement(nums):
    majority_count = len(nums) // 2
    for num in nums:
        count = sum(1 for elem in nums if elem == num)
        if count > majority_count:
            return num


# nums=[3,2,3]
# print(nums)
# print(majorityElement([1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6]))


def majorityElems(nums):

    myDict = {}

    for num in nums:
        if num not in myDict:
            myDict[num] = 1
        else:
            myDict[num] = +1
    #print(myDict)
    return max(myDict, key=myDict.get)
#print(majorityElems([2,2,1,1,1,2,2]))
#++++++++++++++++++++++++++++++++++++=

# https://leetcode.com/problems/move-zeroes/

# Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
#
# Example:
#
# Input: [0,1,0,3,12]
# Output: [1,3,12,0,0]


def moveZeroes(nums):
    pos = 0

    for i in range(len(nums)):
        if nums[i]:
            nums[i], nums[pos] = nums[pos], nums[i]
            pos += 1
    return nums

nums = [0,1,0,2,0,4,5]
#print(moveZeroes(nums))

#++++++++++++++++++++++++++++++++++++++++++

# https://leetcode.com/problems/valid-anagram/

# An anagram of a string is another string that contains same characters, only the order of characters can be different

# Given two strings s and t , write a function to determine if t is an anagram of s.
#
# Example 1:
#
# Input: s = "anagram", t = "nagaram"
# Output: true
# Example 2:
#
# Input: s = "rat", t = "car"
# Output: false

# Youtube: https://www.youtube.com/watch?v=1ns7UFp1o54


def isAnagram(s, t):

    if len(s) != len(t):
        return False

    count = {}

    for ch in s:
        if ch not in count:
            count[ch] = 0
        count[ch] += 1

    for ch in t:
        if ch not in count:
            count[ch] = 0
        count[ch] -= 1

    for key in count.keys():
        if count[key] != 0:
            return False
    return True

# watch youtube to see time and Space Complexity https://www.youtube.com/watch?v=1ns7UFp1o54
#Time Complexity: we are using 2 for loop -> Q(n) + Q(n) = Q(n)
#Speace Complexity:  Q(1)

s = "anagram"
t = "nagaram"
#print(isAnagram(s, t))

#+++++++++++++++++++++++++++++++++++++++++++++++

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

# firt watch https://www.youtube.com/watch?v=XPCXGT4u6Qc
# watch from 1:13 https://www.youtube.com/watch?v=vxIMqdR8flY

# Say you have an array prices for which the ith element is the price of a given stock on day i.
#
# Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).
#
# Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
#
# Example 1:
#       -7 -6 4 -2 3 -2 = 4 +3 =7
# Input: [7,1,5,3,6,4]
# Output: 7
# Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
#              Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
# Example 2:
#          1 1 1 1
# Input: [1,2,3,4,5]
# Output: 4
# Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
#              Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
#              engaging multiple transactions at the same time. You must sell before buying again.
#          1  3  1   2 1  = 6
input = [1, 2, 5, 6, 4, 5]
output = 6

# Note: watch above youtube

def maxProfit(prices):
    profit = 0

    if prices is None:
        return profit

    for i in range(1, len(prices)):
        if prices[i-1] < prices[i]:
            profit += prices[i] - prices[i-1]
    return profit

# Time complexity : O(n). Single pass.
#
# Space complexity: O(1). Constant space needed.
#print(maxProfit(input))

#+++++++++++++++++++++++++++++

# https://leetcode.com/problems/contains-duplicate/

# Given an array of integers, find if the array contains any duplicates.
#
# Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
#
# Example 1:
#
# Input: [1,2,3,1]
# Output: true
# Example 2:
#
# Input: [1,2,3,4]
# Output: false

def containsDuplicate(nums):
    dDict = {}

    for i in range(len(nums)):
        if nums[i] in dDict:
            return True
        else:
            dDict[nums[i]]  = 1
    return False
    #return len(nums) > len(set(nums))

# Time complexity : O(n)O(n). We do search() and insert() for nn times and each operation takes constant time.
#
# Space complexity : O(n)O(n). The space used by a hash table is linear with the number of elements in it.

#print(containsDuplicate([1,2,2,3]))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Roman to Integer

# Youtube: https://www.youtube.com/watch?v=MUUc4GFvlL0


# https://leetcode.com/problems/roman-to-integer/
# Example 1:
#
# Input: "III"
# Output: 3
# Example 2:
#
# Input: "IV"
# Output: 4
# Example 4:
#
# Input: "LVIII"
# Output: 58
# Explanation: L = 50, V= 5, III = 3.
# Example 5:
#
# Input: "MCMXCIV"
# Output: 1994
# Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.


def romanToInt(s):
    dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    prev, curr, total = 0, 0, 0

    for i in range(len(s)):
        curr = dict[s[i]]
        if curr > prev:
            total = total + curr - 2 * prev
        else:
            total += curr
        prev = curr
    return total

# Watch Youtube
#print(romanToInt("XLIX"))

# Explanation:
# X = 10 , L = 50 , I = 1, X=10
#           prev                curr     total
#             0                   0         0
# 1.( X)   0 -> 10               10       0 + 10 -2 = 10
# 2.(L)     50 ( bcz if cond)    50       10+ 50 -2*10 = 40
# 3 (I)       1                   1        40 + 1 = 41
# 4 (X)       10                  10        41+ + 10 - 2 = 49

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=

# https://leetcode.com/problems/first-unique-character-in-a-string/

#  First Unique Character in a String

# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
#
# Examples:
#
# s = "leetcode"
# return 0.
#
# s = "loveleetcode",
# return 2.

def firstUniqChar(inputs):
    import  collections
    count = collections.Counter(inputs)

    for i in range(len(inputs)):
        if count[inputs[i]] == 1:
            return i
    return -1

# Complexity Analysis
#
# Time complexity : O(N) since we go through the string of length N two times.
# Space complexity : O(1) because English alphabet contains 26 letters.

def firstUniqChar2(inputs):
    dict = {}

    for i in range(len(inputs)):
        if inputs[i] not in dict:
            dict[inputs[i]] = 1
        else:
            dict[inputs[i]] += 1

    for i in range(len(inputs)):
        if dict[inputs[i]] == 1:
            return i
    return -1

#print(firstUniqChar("loveleetcode"))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pascal's Triangle
# https://leetcode.com/problems/pascals-triangle/
#  https://www.youtube.com/watch?v=xcq9RoBth1o


# Logic
# here i is row & j is coloum
# 1. if j == 0 ->> 1
# 2. if i == j ->> 1
# 3   (i - 1)(j) + ( i - 1)( j - 1)

# i j
#   1
#   1 1
#   1 2 1
#   1 3 3 1
#   1 4 6 4 1


def generate(numRows):
    result = []
    for i in range(numRows):
        result.append([])
        for j in range(i + 1):
            if j in (0, i):   # Logic 1 & 2
                result[i].append(1)
            else:
               # temp1 = result[i - 1][j - 1]
                #temp2 = result[i - 1][j]
                result[i].append(result[i - 1][j - 1] + result[i - 1][j]) # Logic 3
    return result
#print(generate(5))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Missing Number
# https://leetcode.com/problems/missing-number/
#
# Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
#
# Example 1:
#
# Input: [3,0,1]
# Output: 2
# Example 2:
#
Input = [9,6,4,2,3,5,7,0,1]
# Output: 8

def missingNumber(nums):
    num_set = set(nums)
    n = len(nums) + 1

    for num in range(n):
        if num not in num_set:
            return num

#print(missingNumber(Input))

#Complexity Analysis:

# Time complexity : O(n)
# Because the set allows for O(1) containment queries, the main loop runs in O(n) time. Creating num_set costs O(n) time
# , as each set insertion runs in amortized O(1) time , so the overall runtime is O( n + n ) = O(n)

# Space complexity : O(n)
#
# nums contains n-1 distinct elements, so it costs O(n) space to store a set containing all of them.

# ++++++++++++++++++++++++++++++++++++++++++

# https://leetcode.com/problems/search-a-2d-matrix-ii/
# Search a 2D Matrix II

# Check youtube to understand the logic https://www.youtube.com/watch?v=Ohke9-qwAKU

# Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
#
# Integers in each row are sorted in ascending from left to right.
# Integers in each column are sorted in ascending from top to bottom.

# Example:
#
# Consider the following matrix:
#
# [
#   [1,   4,  7, 11, 15],
#   [2,   5,  8, 12, 19],
#   [3,   6,  9, 16, 22],
#   [10, 13, 14, 17, 24],
#   [18, 21, 23, 26, 30]
# ]
# Given target = 5, return true.
#
# Given target = 20, return false.


def serachMatrix(matrix, target):

    # Checking if row should not empty len(matrix)
    # checking if col should mot empty len(matrix[0])
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return False

    row = 0
    col = len(matrix[0]) - 1

    while (row <= len(matrix) - 1) and col >= 0:
        if matrix[row][col] > target:   # Check youtube to understand the logic https://www.youtube.com/watch?v=Ohke9-qwAKU
            col -= 1
        elif matrix[row][col] < target:
            row += 1
        else:
            return True
    return False


# Time complexity : O(n+m)
# The key to the time complexity analysis is noticing that, on every iteration (during which we do not return true)
# either row or col is is decremented/incremented exactly once. Because row can only be decremented mm times and col can only be
# incremented n times before causing the while loop to terminate, the loop cannot run for more than n+m iterations.
# Because all other work is constant, the overall time complexity is linear in the sum of the dimensions of the matrix.

# Space complexity : O(1)
#Because this approach only manipulates a few pointers, its memory footprint is constant.

# def searchMatrix(self, matrix, target):
#         for row in matrix:
#             if target in row:
#                 return True
#
#         return False

input = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]

#print(serachMatrix(input, 5))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Intersection of Two Arrays II
# https://leetcode.com/problems/intersection-of-two-arrays-ii/
# Youtube:  https://www.youtube.com/watch?v=XtSUDAtB-rI


def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    nums1.sort()
    nums2.sort()
    i, j = 0, 0
    res = []

    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            i += 1
        elif nums1[i] > nums2[j]:
            j += 1
        else:
            res.append(nums1[i])
            i += 1
            j += 1
    return res








#+++++++++++++++++++++  Extra Python code ++++++++++++++++++++++++++

# Python Program to Print Pattern – Print Number, Star, Pyramid, Diamond and Letter Pattern
# https://pynative.com/print-pattern-python-examples/

#Python Pattern Programs - Printing Stars '*' in Right Angle Triangle Shape | Star Pattern

# 1. # Note : Right Angle Triangle Shape or Half pyramid pattern with Star (asterisk)

def pypart(rows):
    # myList = []
    # for i in range(1, row + 1):
    #     myList.append("*" * i)
    # print("\n".join(myList))

    # for i in range(0, rows):
    #     for j in range(0, i + 1):
    #         print("*", end=' ')
    #
    #     print("\r")

    for j in range(1, rows + 1):
        print("* " * j)


# Driver Code
n = 5
#pypart(n)

# 3  Left Half-Pyramid Pattern with Star (asterisk)

def leftHalfPyramid(rows):
    k = 2 * n - 2

    for i in range(0, rows):
        for j in range(0, k):
            print(end=" ")
        k = k - 2
        for j in range(0, i+1):
            print("*", end=" ")
        print("\r")
#print(leftHalfPyramid(5))

# 3 Printing Triangle Or Equilateral triangle pattern with Star (asterisk)

# Function to demonstrate printing pattern triangle
#  Watch this one https://www.youtube.com/watch?v=k_B-5Aad7EU&list=PLzgPDYo_3xuliFyI5dZKgYB99SMUscGyp&index=3

def triangle(n):
    # number of spaces
    k = 2 * n - 2

    # outer loop to handle number of rows
    for i in range(0, n):

        # inner loop to handle number spaces
        # values changing acc. to requirement
        for j in range(0, k):
            print(end=" ")

            # decrementing k after each loop
        k = k - 1

        # inner loop to handle number of columns
        # values changing acc. to outer loop
        for j in range(0, i + 1):
            # printing stars
            print("* ", end="")

            # ending line after each row
        print("\r")

    # Driver Code


n = 5
#print(triangle(n))

# 4 Downward Half-Pyramid Pattern with Star (asterisk)

def downwardHalfPyramid(rows):

    for i in range(rows+1, 0 , -1):
        for j in range(0, i -1):
            print("*", end=" ")
        print(" ")

#print(downwardHalfPyramid(5))

# Star Pattern 5: Print Right start Pattern with Star (asterisk)

def rightStarPattern(rows):

    for i in range(0, rows):
        for j in range(0, i + 1):
            print("*", end=" ")
        print("\r")
    # We are already on 5 th row, so we do not need to increment rows
    for i in range(rows, 0, -1):
        for j in range(0, i - 1):
            print("*", end=" ")
        print("\r")

#print(rightStarPattern(5))


#  2: Print Number pattern in Python  -> https://pynative.com/print-pattern-python-examples/


