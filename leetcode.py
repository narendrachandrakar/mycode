#https://github.com/kamyu104/LeetCode/blob/master/Python/two-sum.py
#https://leetcode.com/problems/two-sum/description/
#===============================================================================
# from __builtin__ import str
# from Carbon.Aliases import false
# from scipy.io.matlab.miobase import arr_dtype_number
# from StdSuites.AppleScript_Suite import result
# from click.testing import Result
# from mailcap import subst
#===============================================================================
from array import array



def twoSum( nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        lookup = {}
        for i, num in enumerate(nums):
            if target - num in lookup:
                #print(num, i
                print(lookup)
                return [lookup[target - num], i]
            lookup[num] = i
             
nums = [3,2,4] 
target = 6
#print(twoSum(nums, target))



class Solution:
    # @return an integer
    def trailingZeroes(self, n):
        result = 0
        while n > 0:
            result += n / 5
            n /= 5
        return result
    
#===============================================================================
# if __name__ == "__main__":
#     print(Solution().trailingZeroes(100))
#===============================================================================

#https://leetcode.com/problems/reverse-integer/description/
#https://github.com/kamyu104/LeetCode/blob/master/Python/reverse-integer.py
# Reverse Integer
def reverse2(x):
         """
         :type x: int
         :rtype: int
         """
         if x < 0:
             return -reverse2(-x)
         else:
             x = int(str(x)[::-1])
         x = 0 if abs(x) > 0x7FFFFFFF else x # abs mens absolute values
         return x

print(reverse2(-1243)  



class Solution:
     # @return a boolean
     def isPalindrome(self, x):
         if x < 0:
             return False
         copy, reverse = x, 0
          
         while copy:
             reverse *= 10
             reverse += copy % 10
             copy //= 10
          
         return x == reverse
  
#if __name__ == "__main__":
  #  print('teset')
    # print(Solution().isPalindrome(12321))
     #print(Solution().isPalindrome(12320))
    #print(Solution().isPalindrome(-12321))

#https://github.com/kamyu104/LeetCode/blob/master/Python/longest-common-prefix.py
#https://www.youtube.com/watch?v=FRVE4593nDM&pbjreload=10
#https://leetcode.com/problems/longest-common-prefix/description/
#Longest Common Prefix
class Solution(object):
      def longestCommonPrefix(self, input):
          """
          :type strs: List[str]
          :rtype: str
          """
          if not input:
              return ""

          for i in xrange(len(input[0])):
              for string in input[1:]:
                  if i >= len(string) or string[i] != input[0][i]:
                      return input[0][:i]
          return input[0]
                    
#===============================================================================
# 
#       def CommonPre(self, input):
#           if not input:
#                  return ""
#           for i in xrange(len(input)):
#              for string in input[1:]:
#                  if i > len(string) or string[i] != input[0][i]:
#                       return input[0][:i]
#===============================================================================

    
#if __name__ == "__main__":
 #print(Solution().longestCommonPrefix(["hello", "heaven", "heavy"]))
# print(Solution().longestCommonPrefix(["apple pie available", "apple pies"]))
 #print(Solution().longestCommonPrefix(["apples", "appleses"]))

 
class Solution:
    # @return a boolean
    def isValid(self, s):
        stack, lookup = [], {"(": ")", "{": "}", "[": "]"}
        for parenthese in s:
            if parenthese in lookup:
                stack.append(parenthese)
            elif len(stack) == 0 or lookup[stack.pop()] != parenthese:
                return False
        return len(stack) == 0
     
#===============================================================================
# if __name__ == "__main__":
#     print(Solution().isValid("()[]{}"))
#     print(Solution().isValid("()[{]}"))
#===============================================================================

def isValid1( s ):
   stack = []
   lookup = {"{":"}", "[":"]", "(":")"}   
   for para in s: 
       if para in lookup:
            stack.append(para)
       elif len(stack) == 0 or lookup[stack.pop()] != para:
            return False
   
   return len(stack) == 0  

#print isValid1("()[]{}")

#===============================================================================
# def count_substring():
#     string = "ABCDCDC"
#     sub_string = "CDC"
#     counter=0
#     i=0
#     while i<len(string):
#         print i, 
#         x = string.find(sub_string,i)
#         print x
#         if string.find(sub_string,i)>=0:
#             i=string.find(sub_string,i)+1
#             print i 
#             counter+=1
#         else: break
#     return counter
# print count_substring()f
#===============================================================================

#https://github.com/kamyu104/LeetCode/blob/master/Python/valid-parentheses.py
#https://leetcode.com/problems/valid-parentheses/description/


class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
         
        last, i = 0, 1
        while i < len(nums):
            if nums[last] != nums[i]:
                last +=1
                nums[last] = nums[i]
            i += 1
        return last+1
     
#===============================================================================
# if __name__ == "__main__":
#     print Solution().removeDuplicates([1, 1, 2,3,3,4])

            
 

#===============================================================================
# class Solution:
#     # @param    A       a list of integers
#     # @param    elem    an integer, value need to be removed
#     # @return an integer
#     def removeElement(self, A, elem):
#         i, last = 0, len(A) - 1
#         while i <= last:
#             if A[i] == elem:
#                 print A[i], A[last], A
#                 A[i], A[last] = A[last], A[i]
#                 print A[i], A[last], A
#                 last -= 1
#             else:
#                 i += 1
#         print A
#         return last + 1
#     
# if __name__ == "__main__":
#     print Solution().removeElement([1, 2, 3, 4, 5, 2, 2], 2)
#===============================================================================
#===============================================================================
# 
# class Solution2(object):
#     def strStr(self, haystack, needle):
#         """
#         :type haystack: str
#         :type needle: str
#         :rtype: int
#         """
#         try:
#             return haystack.index(needle)
#         except:
#             return -1
# 
#     
# if __name__ == "__main__":
#     print Solution2().strStr("a", "")
#     #print Solution2().strStr("abababcdab", "ababcdx")
#     print Solution2().strStr("aaaaa", "bba")
#     print Solution2().strStr("hello", "o")
#===============================================================================
#===============================================================================
# 
# class Solution(object):
#     def searchInsert(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: int
#         """
#         left, right = 0, len(nums) - 1
#         while left <= right:
#             mid = left + (right - left) / 2
#             print mid, left, right
#             if nums[mid] >= target:
#                 right = mid - 1
#             else:
#                 left = mid + 1
# 
#         return left
# 
# 
# if __name__ == "__main__":
#     print Solution().searchInsert([1, 3, 5, 6], 5)
#    # print Solution().searchInsert([1, 3, 5, 6], 2)
#    # print Solution().searchInsert([1, 3, 5, 6], 7)
#    # print Solution().searchInsert([1, 3, 5, 6], 0)
#===============================================================================


#===============================================================================
# def HammingDistance(x,y):
#     distance = 0
#     z = x ^ y
#     while z:
#         distance += 1
#         z &= z-1
#     return distance
# 
# print HammingDistance(1,4)
#         
#===============================================================================


def singleNumber( nums ):
        """
        :type nums: List[int]
        :rtype: int
        """
        #print sum(nums)
        x = sum(set(nums)) * 2
        #print x
       # return (sum(set(nums)) * 2 - sum(nums)) / 2
        return 2 * sum(set(nums)) - sum(nums)

#print singleNumber([1, 1, 2, 2, 3])


class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for i in nums:
            a ^= i
        return a

import operator

def singleNum1(A):
    return reduce(operator.xor, A)

#print singleNum1([1, 1, 2, 2, 3])

# https://leetcode.com/problems/move-zeroes/description/
def moveZero(nums):
    pos = 0
    
    for i in xrange(len(nums)):
        if nums[i]:
            #nums[pos] = nums[i]
            nums[i], nums[pos] = nums[pos], nums[i]
            pos += 1
            
    
    return nums
#print moveZero([0,1,4,0,23,4,0]) 


def movezero( nums ):
    zeroList = []
    nonZeroList = []
    for i in xrange(len(nums)):
        if nums[i]:
            nonZeroList.append(nums[i])
        else:
            zeroList.append(nums[i])
    finalList = nonZeroList + zeroList
    return finalList

#print movezero([0,1,4,0,23,4,0]) 
 

def fundDisappNums(nums):
    #print set(range(1, len(nums)+1))
    return list(set(range(1, len(nums)+1)) - set(nums))

#print fundDisappNums([4, 3, 2, 7, 8, 2, 3, 1])

#print fundDisappNums([9,6,4,2,3,5,7,0,1])
#  Fizz Buzz

def fizzbuzz(n):
    result = []

    for i in xrange(1, n+1):
        if i % 15 == 0:
            result.append("fizzbuzz")
        elif i % 5 == 0:
            result.append("Buzz")
        elif i % 3 == 0:
            result.append("Fizz")
        else:
            result.append(str(i))
    print result
    
#fizzbuzz(20)




#===============================================================================
# INPUT    OUTPUT
# A    B    A XOR B
# 0    0    0
# 0    1    1
# 1    0    1
# 1    1    0
#===============================================================================

# Sum of Two Integers
# https://leetcode.com/problems/sum-of-two-integers/description/
def Add(x, y):

    # Iterate till there is no carry 
    while (y != 0):
    
        # carry now contains common set bits of x and y
        carry = x & y

        # Sum of bits of x and y where at least one of the bits is not set
        x = x ^ y

        # Carry is shifted by one so that adding it to x gives the required sum
        y = carry << 1
    
    return x

#print(Add(2, 3))


#Excel Sheet Column Number
def titleToNumber1(s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        for i in xrange(len(s)):
            result *= 26
            result += ord(s[i]) - ord('A') + 1
        return result

#print titleToNumber1("AAAB")



def titleToNumber( s):
        """
        :type s: str
        :rtype: int
        """
        s = s.upper()
        LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        maps = {}
        for i in range(26):
            maps[LETTERS[i]] = i+ 1
            
        print maps
            
        col = 0 
        for l in s:
            col = 26*col + maps[l]
        
        return col
#print titleToNumber("AAAB")


        
    


#https://www.youtube.com/watch?v=kwaMvB7nhJw
def maxProfit1(prices):
        profit = 0
        #print len(prices) 
        for i in xrange(len(prices) - 1):
            profit += max(0, prices[i + 1] - prices[i])     
        return profit

#print maxProfit1([7, 1, 5, 3, 6, 4])

# https://github.com/kamyu104/LeetCode/blob/master/Python/best-time-to-buy-and-sell-stock.py
# http://www.goodtecher.com/leetcode-121-best-time-buy-sell-stock-java/
#https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
def maxProfit(prices):
        max_profit, min_price = 0, float("inf")  #How can I represent an infinite number in Python using float("inf")
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)  
        return max_profit
#print maxProfit([7, 1, 5, 3, 6, 4])
#print maxProfit1([3, 2, 1, 4, 2, 5, 6])

def maxProfit2(prices):
        return sum(map(lambda x: max(prices[x + 1] - prices[x], 0), range(len(prices[:-1]))))

#print maxProfit2([3, 2, 1, 4, 2, 5, 6])
# start loop from 1, if nums same increase count
# othervise  count decrease and iF CNT 0 than idx will be curent i and cnt will be 1
def majorityElement( nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        idx, cnt = 0, 1
        
        #print len(nums)
        for i in xrange(1, len(nums)):
            if nums[idx] == nums[i]:
                cnt += 1
            else:
                cnt -= 1
                if cnt == 0:
                    idx = i
                    cnt = 1
        
        return nums[idx]
    

    
def majorityElm(nums ):
    from collections import Counter
    count = Counter(nums)

    print type(count)
    max = count.most_common(1)
    print max[0][0]
    
    #max1= max(zip(count.keys(), count.values()))
    #print max1
#print majorityElement([1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6])



def romanToInt( s):
        numeral_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C":100, "D": 500, "M": 1000}
        decimal = 0
        for i in xrange(len(s)):
            if i > 0 and numeral_map[s[i]] > numeral_map[s[i - 1]]:
                decimal += numeral_map[s[i]] - 2 * numeral_map[s[i - 1]]
            else:
                decimal += numeral_map[s[i]]
        return decimal
    
#print romanToInt("IIVX")

# use first string and create map and add count if occer more than one
# use the second string and reduce the char in map and If its not in map add - 1 also need to check 
# if value is less than 0 if Its not anangram
def isAnagram(s, t):
 
    
    if len(s) != len(t):
        return false
    
    count = {}
    
    for c in s:
        if c.lower() in count:
            count[c.lower()] += 1
        else:
            count[c.lower()] = 1
            
    for c in t:
        if c.lower() in count:
            count[c.lower()] -= 1
        else:
            count[c.lower()] = -1
        if count[c.lower()] < 0:
            return false
    return True

#print isAnagram("anagram","nagaram")



def isPalindrome(str):
    str = str.lower()
    if str == str[::-1]:
        return True
    else:
        return False
    
#print isAnangram("civic")    
    
# Create dict with all char and add count 
# after travese the str and check dict with char count is eqaul to 1  
        
def firstUniqChar(s):
        """
        :type s: str
        :rtype: int
        """
        letters = {}
        #print letters
        for c in s:
            letters[c] = letters[c] + 1 if c in letters else 1
        for i in xrange(len(s)):
            if letters[s[i]] == 1:
                return i
        return -1
    
#print firstUniqChar("loveleetcode") 




def containDuplicate(nums):
    return len(nums) > len(set(nums))      
    
#print containDuplicate([1,2,2,3,4,5])


# set with len+1 array minus orign array
def missingNumber(nums):
    return list(set(range(1, len(nums)+1)) - set(nums))

#print missNums([1,0,3])


#===============================================================================
# 
# def intersection2(nums1, nums2):
#         """
#         :type nums1: List[int]
#         :type nums2: List[int]
#         :rtype: List[int]
#         """
#         return list(set(nums1) & set(nums2))
# 
# print intersection2([1, 2, 2, 1],[2, 2])
#===============================================================================

#https://github.com/kamyu104/LeetCode/blob/master/Python/intersection-of-two-arrays-ii.py
#https://leetcode.com/problems/intersection-of-two-arrays-ii/description/
#Intersection of Two Arrays II
def intersect(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    nums1.sort(), nums2.sort()  # Make sure it is sorted, doesn't count in time.

    res = []
    
    it1, it2 = 0, 0
    while it1 < len(nums1) and it2 < len(nums2):
        if nums1[it1] < nums2[it2]:
            it1 += 1
        elif nums1[it1] > nums2[it2]:
            it2 += 1
        else:
           # res += nums1[it1],
            res.append(nums1[it1])
            it1 += 1
            it2 += 1
    
    return res
#print intersect([1, 2, 2,3,1],[2,2,3,5])





#https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

class Solution:
    # @param {integer} n
    # @return {boolean}
    def isHappy(self, n):
        lookup = {}
        while n != 1 and n not in lookup:
            lookup[n] = True
           # print lookup
            n = self.nextNumber(n)
        return n == 1
    
    def nextNumber(self, n):
        print n
        new = 0
        for char in str(n):
            new += int(char)**2
        return new

x = Solution()
#print x.isHappy(19)


# https://www.geeksforgeeks.org/count-ways-reach-nth-stair/

def climbStairs( n):
        if n == 1:
            return 1
        if n == 2:
            return 2
        return climbStairs(n - 1) + climbStairs(n - 2)
    

#print climbStairs(3)


# power-of-three.py
# http://bookshadow.com/weblog/2016/01/08/leetcode-power-three/
import math
def isPowerOfThree( n):
       #print round(math.log(n,3))
       return n > 0 and  3 ** round(math.log(n,3)) == n
#print isPowerOfThree(27)


#+++++++++++++
# Merge Two Sorted Lists

class ListNode1(object):
    def __init__(self, x):
        self.val = x
        self.next = None
         
    def __repr__(self):
        return "{}->{}".format(self.val, self.next)
     
class Solution1(object):
    def mergeList(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeList(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeList(l2.next, l1)
            return l2
    
#===============================================================================
# if __name__ == "__main__":
#  l1 = ListNode1(0)
#  #print l1.next
#  l1.next = ListNode1(1)
#  l2 = ListNode1(2)
#  l2.next = ListNode1(3)
#  print Solution1().mergeList(l1, l2)
#===============================================================================


#+++++++++
# number-of-1-bits.py

def hammingWeight(n):
    result = 0
    while n:
        n = n & (n -1 )
        result = result + 1
    return result
    
#print hammingWeight(11)


#+++++++
#Good explain  https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/

def maxSubArraySum(arr):
     
    max_so_far = 0
    max_ending_here = 0
    size = len(arr)
     
    for i in range(0, size):
        max_ending_here = max_ending_here + arr[i]
        if max_ending_here < 0:
            max_ending_here = 0
         
        # Do not compare for all elements. Compare only   
        # when  max_ending_here > 0
        elif (max_so_far < max_ending_here):
            max_so_far = max_ending_here
             
    return max_so_far


#print maxSubArraySum([-2,1,-3,4,-1,2,1,-5,4])



#++++++++++
#https://www.geeksforgeeks.org/find-maximum-possible-stolen-value-houses/
#===============================================================================
# dp[i] = max (hval[i] + dp[i-2], dp[i-1])
# 
# hval[i] + dp[i-2] is the case when thief
# decided to rob house i. In that situation 
# maximum value will be the current value of
# house + maximum value stolen till last 
# robbery at house not adjacent to house 
# i which will be house i-2.  
#  
# dp[i-1] is the case when thief decided not 
# to rob house i. So he will check adjacent 
# house for maximum value stolen till now.         
#===============================================================================
# House Robber      
def maximize_loot(hval):
    n = len(hval)
    if n == 0:
        return 0
    if n == 1:
        return hval[0]
    if n == 2:
        return max(hval[0], hval[1])
 
    # dp[i] represent the maximum value stolen so
    # for after reaching house i.
    dp = [0]*n
 
    # Initialize the dp[0] and dp[1]
    dp[0] = hval[0]
    dp[1] = max(hval[0], hval[1])
     
    # Fill remaining positions
    for i in range(2, n):
        dp[i] = max(hval[i]+dp[i-2], dp[i-1])
        #print dp
    return dp[-1]

hval = [6, 7, 1, 3, 8, 2, 4]
#print maximize_loot(hval)

# number of houses
n = len(hval)
#print("Maximum loot value : {}".\
    #    format(maximize_loot1(hval, n)))

#++++++++
#Pascal's Triangle
# https://github.com/kamyu104/LeetCode/blob/master/Python/pascals-triangle.py
# https://www.geeksforgeeks.org/pascal-triangle/
def generate(numRows):
        result = []
        for i in xrange(numRows):
            result.append([])
            for j in xrange(i + 1):
                if j in (0, i):
                    result[i].append(1)
                else:
                    result[i].append(result[i - 1][j - 1] + result[i - 1][j])
        return result
#print generate(5)

#+++++++

        
        

#https://github.com/kamyu104/LeetCode/blob/master/Python/plus-one.py
def plusOne(digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        for i in reversed(xrange(len(digits))):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits
        digits[0] = 1
        digits.append(0)
        return digits
    
    
#print plusOne([9, 9, 9, 9,9])
#print plusOne([4,3,2,1])



#============

#https://www.geeksforgeeks.org/count-trailing-zeroes-factorial-number/
# Please read the explain above link
# factorial-trailing-zeroes.py

def findTrailingZeros(n):
    count = 0
    i = 5
    while n/i >= 1:
         count += n/i
         i = i * 5
    return int(count)

#print findTrailingZeros(100)
#x

class Solution:
    # @return a string
    def countAndSay(self, n):
        s='1'
        for i in range(1,n):
            s=self.cal(s)
        return s
        
    def cal(self,s):          
        cnt=1
        length=len(s)
        ans=''
        for i ,c in enumerate(s):
            if i+1 < length and s[i]!=s[i+1]:
                ans=ans+str(cnt)+c
                cnt=1
            elif i+1 <length:
                cnt=cnt+1
                
        ans=ans+str(cnt)+c    
        return ans
    
#===============================================================================
# if __name__ == "__main__":
#     for i in xrange(1, 5):
#         print Solution().countAndSay(i)
#===============================================================================


def removeDuplicate(arr):
    if not arr:
        return 0
    last, i = 0, 1
    while i < len(arr):
        if arr[last] != arr[i]:
            last = last + 1
            arr[last] = arr[i]
            
        i += 1
    return last+1

def removeDup( arrs ):
    if not arrs:
      return arrs
    
    last = 0
    i = 1
    
    while i < len(arrs):
         if arrs[last] != arrs[i]:
             last = last+1
             arrs[last] = arrs[i]
         i += 1
    print arrs[:last+1]
    return last+1


#print removeDup([1, 2, 2, 3, 4, 4, 4, 5, 5])

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # @param head, a ListNode
    # @return a boolean
    def hasCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
            if fast is slow:
                return True
        return False

if __name__ == "__main__":
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = head.next
#    print Solution().hasCycle(head)
   
# +++++


class Solution:
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing
    def merge(self, A, m, B, n):
        last, i, j = m + n - 1, m - 1, n - 1
        
        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[last] = A[i]
                last, i = last - 1, i - 1
            else:
                A[last] = B[j]
                last, j = last - 1, j - 1
        
        while j >= 0:
                A[last] = B[j]
                last, j = last - 1, j - 1
        #print A
        
    
   
            
    
#===============================================================================
# 
# if __name__ == "__main__":
#      A = [1, 3, 5, 0, 0, 0, 0]
#      B = [2, 4, 6, 7]
#      Solution().merge(A, 3, B, 4)
#      print A
#===============================================================================


#===============================================================================
# def merge( C, m , D, n):
#     last = m + n -1
#     i = m -1
#     j = n -1
#     while i >= 0 and j >= 0:
#          if C[i] > D[j]:
#              C[last] = C[i] 
#              last = last -1
#              i = i - 1
#          else:
#              C[last] = D[j]
#              last = last - 1
#              j = j - 1
#             
#     while j >= 0:
#         C[last] = D[j]
#         last = last - 1
#         j = j- 1
#     print C
#         
# 
# C = [1,3,5,0,0, 0]
# D = [2, 4, 6, 7]
# merge(C, 3, D, 4)
#===============================================================================


#===============================================================================
# class MinStack:
#     def __init__(self):
#         self.min = None
#         self.stack = []
#         
#     # @param x, an integer
#     # @return an integer
#     def push(self, x):
#         if not self.stack:
#             self.stack.append(0)
#             self.min = x
#         else:
#             self.stack.append(x - self.min)
#             if x < self.min:
#                 self.min = x
# 
#     # @return nothing
#     def pop(self):
#         x = self.stack.pop()
#         if x < 0:
#             self.min = self.min - x
# 
#     # @return an integer
#     def top(self):
#         x = self.stack[-1]
#         if x > 0:
#             return x + self.min
#         else:
#             return self.min
#         
#     # @return an integer
#     def getMin(self):
#         return self.min
#===============================================================================
# https://codesays.com/2014/solution-to-min-stack-by-leetcode/
# Min Stack
class MinStack1:
    def __init__(self):
        self.data = []          # Store all the data
        self.minData  = []      # Store the minimum element so far
 
    # @param x, an integer
    # @return nothing
    def push(self, x):
        self.data.append(x)
 
        # Check if we need to update the minimum value
        if len(self.minData) == 0 or x <= self.minData[-1]:
            self.minData.append(x)
 
    # @return an integer
    def pop(self):
        if len(self.data) == 0:
            # Empty stack
            return None
        else:
            if self.data[-1] == self.minData[-1]:   self.minData.pop()
            return self.data.pop()
 
    # @return an integer
    def top(self):
        if len(self.data) == 0: return None             # Empty stack
        else:                   return self.data[-1]
 
    # @return an integer
    def getMin(self):
        if len(self.minData) == 0:  return None         # Empty stack
        else:                       return self.minData[-1]



class MiniStock11:
    def __init__(self):
       self.data = []
       self.minData = []
    
    def push(self, x):
       self.data.append(x)
       
       if len(self.minData) == 0  or x <= self.minData[-1]:
               self.minData.append(x)
            
    def pop(self):
      if len(self.data) == 0:
              return None
      else:
          if  self.data[-1] == self.minData[-1]: self.minData.pop()
          return self.data.pop()
           
    def top(self):
       if len(self.data) == 0: return None
       else:
            return self.data[-1]
    
    def getMin(self):
        if len(self.minData) == 0: return None
        else:
             return self.minData[-1]
#===============================================================================
# if __name__ == "__main__":
#     stack = MinStack1()
#     stack.push(1)
#     #print [stack.top(), stack.getMin()]
#     stack.push(12)
#     stack.push(-1)
#     print [stack.top(), stack.getMin()]
#     stack.pop()
#     print [stack.top(), stack.getMin()]
#===============================================================================
  
#++++++++++++

#===============================================================================
# def reverseBits1( n):
#         result = 0
#         for i in xrange(32):
#             result <<= 1
#             result |= n & 1
#             n >>= 1
#         return result
#     
#===============================================================================
# https://www.geeksforgeeks.org/reverse-bits-positive-integer-number-python/

def reverseBits(num,bitSize):
 
     # convert number into binary representation
     # output will be like bin(10) = '0b10101'
     binary = bin(num)
     print binary
 
     # skip first two characters of binary
     # representation string and reverse
     # remaining string and then append zeros
     # after it. binary[-1:1:-1]  --> start
     # from last character and reverse it until
     # second last character from left
     reverse = binary[-1:1:-1]
     print reverse
     reverse = reverse + (bitSize - len(reverse))*'0'
     print reverse
 
     # converts reversed binary string into integer
     print int(reverse,2)
#reverseBits(1, 32)

#++++++++++
#strStr() https://github.com/kamyu104/LeetCode/blob/master/Python/implement-strstr.py
def strStr2( haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        try:
           return haystack.index(needle)
        except:
            return -1
        

#print strStr2("aaaaa", "bba")
#print strStr2("hello", "ll")
#+++++++++++++++
# https://github.com/kamyu104/LeetCode/blob/master/Python/sqrtx.py Sqrt(x)
def mySqrt( x):
        """
        :type x: int
        :rtype: int
        """
        if x < 2:
            return x
        
        left, right = 1, x // 2
        while left <= right:
            mid = left + (right - left) // 2
            if mid > x / mid:
                right = mid - 1
            else:
                left = mid + 1

        return left - 1

            
#print mySqrt(10)

#++++++++++++++++

#https://github.com/kamyu104/LeetCode/blob/master/Python/valid-palindrome.py

def isPalindrome(s):
        i = 0, 
        j = len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i, j = i + 1, j - 1
        return True

   


def isPalindrome1(w):
   w = w.lower()
   return w == w[::-1]

#print isPalindrome("a++a");

#++++++++++++++++
# https://www.geeksforgeeks.org/program-find-sum-prime-numbers-1-n/ 
# Count Primes
def sumOfPrimes(n):
    # list to store prime numbers
    prime = [True] * (n + 1)
     
    # Create a boolean array "prime[0..n]"
    # and initialize all entries it as true.
    # A value in prime[i] will finally be
    # false if i is Not a prime, else true.
     
    p = 2
    while p * p <= n:
        # If prime[p] is not changed, then
        # it is a prime
        print p
        if prime[p] == True:
            # Update all multiples of p
            i = p * 2
            print i
            while i <= n:
                prime[i] = False
                i += p
        p += 1   
          
    # Return sum of primes generated through
    # Sieve.
    sum = 0
    for i in range (2, n + 1):
        if(prime[i]):
            sum += i
    return sum
    
n = 11
#print sumOfPrimes(n)

#+++++++++++++=
#https://github.com/kamyu104/LeetCode/blob/master/Python/rotate-array.py
# Rotate Array
def rotate2( nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        print nums
        print len(nums)
        print nums[len(nums)- k:]
        nums[:] = nums[len(nums)- k:] + nums[:len(nums) - k]  # nums[4:] + nums[:4]
        #return nums
        

nums = [1, 2, 3, 4, 5, 6, 7]
#===============================================================================
# rotate2(nums, 3)
# print nums
#===============================================================================


#===============================================================================
# class Solution:
#     """
#     :type nums: List[int]
#     :type k: int
#     :rtype: void Do not return anything, modify nums in-place instead.
#     """
# 
#     def rotate(self, nums, k):
#         k %= len(nums)
#         self.reverse(nums, 0, len(nums))
#         self.reverse(nums, 0, k)
#         self.reverse(nums, k, len(nums))
# 
#     def reverse(self, nums, start, end):
#         while start < end:
#             nums[start], nums[end - 1] = nums[end - 1], nums[start]
#             start += 1
#             end -= 1
# nums = [1, 2, 3, 4, 5, 6, 7]
# Solution().rotate(nums, 3)
# print nums
#===============================================================================

# 3sum.py 
# https://www.geeksforgeeks.org/find-a-triplet-that-sum-to-a-given-value/


def sum3(arr, sum):
    arrSize = len(arr)
    arr.sort()
    r = arrSize -1
    for i in range(0, arrSize -2):
        l = i + 1
       
        while (l < r):
            result = arr[i]+ arr[l] + arr[r]
            if (result == sum ):
                print arr[i], arr[l] , arr[r]
                return True
            if (result < sum ):
                l += 1
            else:
                r -= 1
        return False

arr = [1, 4, 45, 6, 10, 8]
sum = 13
 
#print sum3(arr, sum)
    
                  
#===============================================================================
# def findTriplets(arr, n):
#  
#     found = False
#     # sort array elements
#     arr.sort()
#     for i in range(0, n-1): 
#         # initialize left and right
#         l = i + 1
#         r = n - 1
#         x = arr[i]
#         while (l < r):
#          
#             if (x + arr[l] + arr[r] == 0):
#                 # print elements if it's sum is zero
#                 print(x, arr[l], arr[r])
#                 l+=1
#                 r-=1
#                 found = True           
#  
#             # If sum of three elements is less
#             # than zero then increment in left
#             elif (x + arr[l] + arr[r] < 0):
#                 l+=1
#             # if sum is greater than zero than
#             # decrement in right side
#             else:
#                 r-=1
#          
#     if (found == False):
#         print(" No Triplet Found")
#     
# #arr = [0, -1, 2, -3, 1]
# arr = [-1, 0, 1, 2, -1, -4]
# 
# n = len(arr)
#  
# print findTriplets(arr, n)
#===============================================================================



nums = [-1, 0, 1, 2, -1, -4]
    
#nums = [1, 4, 45, 6, 10, 8]
#sum_ = 22
sum_ = 0
result = []

nums.sort()

r=len(nums)-1
for i in range(len(nums)-2):
    l = i + 1  # we don't want l and i to be the same value.
               # for each value of i, l starts one greater
               # and increments from there.
    while (l < r):
        sum_ = nums[i] + nums[l] + nums[r]
        if (sum_ < 0):
            l = l + 1
        if (sum_ > 0):
            r = r - 1
        if not sum_:  # 0 is False in a boolean context
            result.append([nums[i],nums[l],nums[r]])
            l = l + 1  # increment l when we find a combination that works
#print result

            
#===============================================================================
# print result
# for i in result:
#     print i
#===============================================================================

#++++++++++++++=
# Product of Array Except Self
# https://www.youtube.com/watch?v=R745JLPox_A

# https://www.geeksforgeeks.org/a-product-array-puzzle/
# https://leetcode.com/problems/product-of-array-except-self/description/
#===============================================================================
# Algorithm:
# 1) Construct a temporary array left[] such that left[i] contains product of all elements on left of arr[i] excluding arr[i].
# 2) Construct another temporary array right[] such that right[i] contains product of all elements on on right of arr[i] excluding arr[i].
# 3) To get prod[], multiply left[] and right[].
#===============================================================================

def productExceptSelf( nums):
        if not nums:
            return []
            
        left_product = [1] * len(nums)
        for i in xrange(1, len(nums)):
            left_product[i] = left_product[i - 1] * nums[i - 1]
            
        right_product = 1
        for i in xrange(len(nums) - 2, -1, -1):
            right_product *= nums[i + 1]
            left_product[i] = left_product[i] * right_product

        return left_product
    
#print productExceptSelf([1,2,3,4])
      
#++++++++++
# https://github.com/kamyu104/LeetCode/blob/master/Python/subsets.py
# Subsets

def subsets(nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        result = [[]]
        for i in xrange(len(nums)):
            #size = len(result)
            for j in xrange(len(result)):
                result.append(list(result[j]))
                result[-1].append(nums[i])
        return result
        
#print subsets([1,2,3])  

#++++++++++++++++++
# https://www.tangjikai.com/algorithms/leetcode-215-kth-largest-element-in-an-array

#===============================================================================
# Step 1: select nums[0] as pivot
# Step 2: tail means the end of window that all elements are larger than pivot
# Step 3: swap pivot with nums[tail]
# If tail + 1 == k: return pivot. Since (tail) elements larger than pivot, so pivot is (tail + 1)th largest
# If tail + 1 < k: result is among the window excluding pivot, so cut off nums to nums[:tail](nums[tail] is pivot, kicked out)
# If tail + 1 > k: result is outside of window, so cut off nums to nums[tail+1:].
#===============================================================================


def findKthLargest(nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        pivot = nums[0]
        tail = 0

        for i in range(1, len(nums)):
            if nums[i] > pivot:
                tail += 1
                nums[tail], nums[i] = nums[i], nums[tail]
        
        nums[tail], nums[0] = nums[0], nums[tail]
        
        if tail + 1 == k:
            return pivot
        elif tail + 1 < k:
            return findKthLargest(nums[tail+1:], k - tail - 1)
        else:
            return findKthLargest(nums[:tail], k)  #excluding pivot
        
#print findKthLargest([3,2,1,5,6,4], 2)




     
             
#print reverse1(s)

# https://leetcode.com/problems/product-of-array-except-self/description/


#++++

# Python program to find top k elements in a stream
 
# Function to print top k numbers
import collections
def topKFrequent( nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counts = collections.Counter(nums)
        buckets = [[] for _ in xrange(len(nums)+1)]
        for i, count in counts.iteritems():
            buckets[count].append(i)
            
        result = []
        for i in reversed(xrange(len(buckets))):
            for j in xrange(len(buckets[i])):
                result.append(buckets[i][j])
                if len(result) == k:
                    return result
        return result
#print topKFrequent([1,1,1,2,2,3], 2)
#print topKFrequent([5, 2, 1, 3, 4], 4)

# This code is contributed by Sachin Bisht

        

# decorator      
#===============================================================================
# def dollar(func):
#     def new(*args):
#         return "$" + str(func(*args))
#     
#     return new
# 
# @dollar
# def price(amount, tax):
#     return amount + tax
# 
# print price(100, 0.1)
#===============================================================================


def isPrime( numbers ):
    if numbers == 1:
        return False
    for t in range(2, numbers):
        if numbers % t == 0:
            return False
    return True

#print [ n for n in range(10) if isPrime(n)]
            
#print [ n for n in range(1,100) if isPrime(n)]

def reversWord( ):
    str = "narendra chandrakar"
    str1 = "narendra chandrakar"
    str = str[::-1]
    str = str.split()
    rWards = []
    
    for word in str:
        rWards.append(word[::-1])
 
    
    print ' '.join(rWards)
    #print ' '.join([m[::-1] for m in str1[::-1].split() ])

#reversWord()

stirng = "narendra"
#print "".join([string(i) for i in range(len(string) - 1)])

def reverse12(word):
    #string = "narendrachandrakar"
    if len(word) <= 1:
         return word
     
    return reverse12(word[1:]) + word[0]
#===============================================================================
# word = "narendrachan"
# print  reverse12(word)
#===============================================================================
a = [10,30,50,70,70,30,40,30,10]

d = {}

for k in a:
    d[k] = 0
    
for k in a:
    d[k] += 1
    
#print [k for k,v in d.items() if v > 1 ]
x=sorted(d,key=d.__getitem__) # Sorted key with values
#print x

def fact(n):
    if n == 0:
        return 1
    return n * fact(n-1)
#print fact(5)

# recursive
def fibr(n):
   if n == 0 or n == 1: return n
   return fibr(n-2)+fibr(n-1)

#===============================================================================
# for i in range(10):
#    print (fibr(i), fibr(i))
#===============================================================================

#####

# Reverse String
#https://leetcode.com/problems/reverse-string/description/

def reverse(string):
   # string = string[::-1]
    string = "".join(reversed(string))
    return string
 
s = "Geeksforgeeks"
#print reverse(s)

def reverse1(str):
    if len(str) == 0:
        return str
    return reverse1(str[1:]) + str[0]
    #return "".join([string[i] for i in range(len(string)-1, -1, -1)])

def reverse12( string ):
     if len(string) == 0:
        return string
     #return reverse12(string[1:]) + string[0]
     #return "".join(reversed(string))
     return "".join([string[i] for i in range(len(string)-1, -1, -1)])
     #return string[::-1]
 
#print reverse12("naremdra")

## This is the text editor interface. 
## Anything you type or change here will be seen by the other person in real time.p

import os 

string = "narendra"

def reverse( string ):
    string = string[::-1]
    return string 
    
def reverse1( string ):
    if len(string) == 0:
         return string
    return "".join(reversed(string))
    
def reverse2( string ):
    if len( string ) == 0:
        return string 
    #return "".join(string[i] for i in range(len(string)-1, -1, -1))
    return reverse2(string[1:]) + string[0]

#print reverse2(string)

def reversed1( input ):
    if len(input) == 0:
        return None
    #return "".join([input[i] for i in range(len(input)-1, -1, -1)])
    return "".join([input[i] for i in range(len(input)-1,-1,-1)])
    #return reversed1(input[1:]) + input[0]
    #return reversed(input)
    #return input[::-1]
#print reversed1(string)


def dollar(fn):
    def new(*args):
        return "$" + str(fn(*args))
    return new
    
@dollar
def price(amount, tax):
    return amount + tax

#print price(100,10)

# https://www.geeksforgeeks.org/class-method-vs-static-method-python/

#Static
class A:
    total = 0
    
    def __init__(self,name):
        self.name = name
        A.total += 1
    @staticmethod
    def status():
        print "Total INS", A.total
        
        
#===============================================================================
# if __name__ == "__main__":
#    obj1 = A("1")
#    obj2 = A("2")
#    A.status()    
#===============================================================================

# https://stackoverflow.com/questions/33217326/finding-a-substring-of-a-string-in-python-without-inbuilt-functions
# Finding a substring of a string in Python without inbuilt functions

def index(input, subStr):
    start, end = 0,0
    
    while start < len(input):
          if input[start + end] != subStr[end]:
               start += 1
               end = 0
               continue
          end += 1
          if end == len(subStr):
               print subStr[:start]
               return start
    return -1


#print index("dedicate", 'cat')
#print index('hello world', 'word')
#print index("geeksforgeeks", "for")


       
# https://www.geeksforgeeks.org/python-print-sublists-list/

    
def sub_lists(list1):
 
    # store all the sublists 
    sublist = [[]]
     
    # first loop 
    for i in range(len(list1)):
         
        # second loop 
        for j in range(i + 1, len(list1)):
             
            # slice the subarray 
            sub = list1[i:j]
            sublist.append(sub)
    print sublist
    sublist.append(list1)
    print  sublist
    


    
#===============================================================================
# l1 = [1, 2, 3, 4]
# print(sub_lists(l1))
#===============================================================================




# https://www.geeksforgeeks.org/longest-prefix-also-suffix/
# Python3 program to find length 
# of the longest prefix which 
# is also suffix
 
#===============================================================================
# Input : aabcdaabc
# Output : 4
# The string "aabc" is the longest
# prefix which is also suffix.
# 
# Input : abcab
# Output : 2
# 
# Input : aaaa
# Output : 2
#===============================================================================
 
def longestPrefixSuffix(s) :
    n = len(s)
     
    for res in range(n // 2, 0, -1) :
        print res
         
        # Check for shorter lengths
        # of first half.
        prefix = s[0: res]
        suffix = s[n - res: n]
        print prefix
        print suffix
         
        if (prefix == suffix) :
            #print prefix -> bla
            return res
             
 
    # if no prefix and suffix match 
    # occurs
    return 0


     
s = "blablabla"
#s= "abab"

#print(longestPrefixSuffix(s))

# https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""    # match will contians only match b/n string
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)):  # This will check answer should be up to date
                    answer = match
                match = ""
    return answer


#print longestSubstringFinder("apple pie available", "apple pies")
#print longestSubstringFinder("the how apples", "how appleses")
 
 # https://www.jianshu.com/p/4406bf26366e
class Solution(object):
    def repeatedSubstringPattern(self, str):
        """
        :type str: str
        :rtype: bool
        """
        def split(str, width):
            res = []
            start = 0
            while (start + width < len(str)):
                res.append(str[start:start + width])
                start += width
            res.append(str[start:])
            return res

        for i in range(1, len(str)):
            split_list = split(str, i)
            print split_list
            import collections
            dict = collections.Counter(split_list)
            print dict
            if len(dict) == 1:
                return True
            else:
                continue
        return False
        
       
    
#obj = Solution()
#print obj.repeatedSubstringPattern("abab")

# https://www.w3resource.com/python-exercises/python-basic-exercise-70.php

#===============================================================================
# import glob
# import os
# 
# files = glob.glob("*.txt")
# # This will make list as time
# files.sort(key=os.path.getmtime)
# print("\n".join(files))
# 
# # Now to get max time which modified
# lastModified = max(file, key=os.path.getctime)
# print lastModified
# 
# # Now to get min time which modified
# minModified = min(file, os.path.getctime)
# print minModified
# 
# # Max size which modified
# maxFileSized = max(file, key=os.path.getsize)
# print maxFileSized 
#===============================================================================

# +++++++++++++++++
#Python Programming Tutorial - 41 - Min, Max, and Sorting Dictionar
# https://www.youtube.com/watch?v=U5I-2UZBOVg&list=PLfmc45xFf1AkZFPyIXUpSTXtnQCh2wWGM&index=3

comDict = {"app":"200", "cis":"100", "saf":"50"}

#print zip(comDict.keys(),comDict.values())
# Find max by value
#print max(zip(comDict.values(), comDict.keys()))

# Min
#print min(zip(comDict.values(), comDict.keys()))
# sorted 
#print sorted(zip(comDict.values(), comDict.keys()))

# https://www.geeksforgeeks.org/bubble-sort/

def bubbleSort(arrs):
    for i in range(len(arrs)):
        for j in range(len(arrs)-i-1):
            if arrs[j] > arrs[j+1]:
                arrs[j], arrs[j+1] = arrs[j+1], arrs[j]
   # print arrs
    
arr = [64, 34, 25, 12, 22, 11, 90]
  
#bubbleSort(arr)

# Python program for implementation of Insertion Sort
 
# Function to do insertion sort
# https://www.youtube.com/watch?v=Nkw6Jg_Gi4w&t=47s
def insertionSort(arr):
 
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
 
        key = arr[i]
 
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >=0 and key < arr[j] :
                arr[j+1] = arr[j]
                j -= 1
        arr[j+1] = key
    print arr
#insertionSort(arr)

# https://www.geeksforgeeks.org/selection-sort/
# https://www.youtube.com/watch?v=mI3KgJy_d7Y&t=137s
def selectionsort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j 
        arr[i], arr[min_idx] = arr[min_idx],arr[i]
    print arr
#selectionsort(arr)   

# https://github.com/joeyajames/Python/blob/master/Mergesort.py

               

# https://www.youtube.com/watch?v=5eO5-w2g1Ik&t=0s&list=PLfmc45xFf1AkZFPyIXUpSTXtnQCh2wWGM&index=6
# Python Programming Tutorial - 54 - Finding Most Frequent Items

from collections import Counter

words = "narendra the good the happy the good was chandrakar"
words = words.split()

count = Counter(words)
#print count
max_three = count.most_common(3)
#print max_three









# Copying from one text file to another using Python

# https://stackoverflow.com/questions/15343743/copying-from-one-text-file-to-another-using-python
#===============================================================================
# 
# with open("out.txt", 'w') as fw, open("in.txt", 'r') as fr:
#     fw.writelines([l for l in fr if "narendra" in l])
#     
# with open("out.txt", 'r') as fr:
#     for line in fr:
#         print line
#===============================================================================

# MAke file

#===============================================================================
# CC = gcc
# CFLAGs = -g -Wall
# 
# TARGET = myprog
# 
# all: ${TARGET}
# 
# ${TARGET}: ${TARGET}.c
#             ${CC} {CFLAGs} -o ${TARGET} ${TARGET}.c
#             
# clean: ${RM} ${TARGET}
#===============================================================================