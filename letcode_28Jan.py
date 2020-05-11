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


