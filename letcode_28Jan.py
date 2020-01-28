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

def threeSum11(nums):
        nums.sort()
        arr = []

        for idx in range(len(nums) - 2):
            if ((idx == 0) or (idx > 0 and nums[idx] != nums[idx - 1])):
                l = idx + 1
                r = len(nums) - 1

                while (l < r):
                    sum_t = nums[idx] + nums[l] + nums[r]
                    if sum_t == 0:
                        arr.append([nums[idx], nums[l], nums[r]])

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
print(threeSum11(nums))



