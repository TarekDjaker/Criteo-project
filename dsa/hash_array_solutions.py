"""
DSA Solutions - Hash & Array Patterns
Optimized for Criteo Interview (25 min max per problem)
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import heapq


class HashArraySolutions:
    """Collection of hash/array problems commonly asked in interviews"""

    def two_sum_all_pairs(self, nums: List[int], target: int) -> List[Tuple[int, int]]:
        """
        Find ALL pairs that sum to target (Criteo loves this variant)
        Time: O(n), Space: O(n)

        Pattern: Hash table for complement lookup
        Edge cases: duplicates, negative numbers
        """
        seen = {}
        pairs = []

        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                for j in seen[complement]:
                    pairs.append((j, i))

            if num not in seen:
                seen[num] = []
            seen[num].append(i)

        return pairs

    def top_k_frequent_elements(self, elements: List[int], k: int) -> List[int]:
        """
        Find k most frequent elements (useful for feature selection)
        Time: O(n log k), Space: O(n)

        Pattern: Counter + Min Heap
        Real use: Top K ad categories, user segments
        """
        if not elements or k <= 0:
            return []

        count = Counter(elements)

        # Min heap of size k
        heap = []
        for num, freq in count.items():
            heapq.heappush(heap, (freq, num))
            if len(heap) > k:
                heapq.heappop(heap)

        return [num for freq, num in heap]

    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """
        Maximum in each window of size k (CTR over time windows)
        Time: O(n), Space: O(k)

        Pattern: Monotonic deque
        Real use: Rolling CTR, peak detection
        """
        from collections import deque

        if not nums or k <= 0:
            return []

        dq = deque()  # Store indices
        result = []

        for i, num in enumerate(nums):
            # Remove indices outside window
            while dq and dq[0] < i - k + 1:
                dq.popleft()

            # Maintain decreasing order
            while dq and nums[dq[-1]] < num:
                dq.pop()

            dq.append(i)

            # Add to result after first window
            if i >= k - 1:
                result.append(nums[dq[0]])

        return result

    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        """
        Count subarrays with sum = k (revenue targeting)
        Time: O(n), Space: O(n)

        Pattern: Prefix sum + hash
        Real use: Budget allocation, campaign targeting
        """
        count = 0
        prefix_sum = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1  # Empty subarray

        for num in nums:
            prefix_sum += num

            # Check if (prefix_sum - k) exists
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]

            sum_count[prefix_sum] += 1

        return count

    def merge_intervals_with_weights(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Merge overlapping intervals (ad scheduling)
        Time: O(n log n), Space: O(n)

        Pattern: Sort + merge
        Real use: Campaign scheduling, bid windows
        """
        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            last = merged[-1]

            if current[0] <= last[1]:  # Overlap
                last[1] = max(last[1], current[1])
                # If intervals have weights, sum them
                if len(current) > 2:
                    if len(last) == 2:
                        last.append(current[2])
                    else:
                        last[2] += current[2]
            else:
                merged.append(current)

        return merged

    def find_duplicate_transactions(self, transactions: List[Dict]) -> List[List[Dict]]:
        """
        Find duplicate/similar transactions (fraud detection)
        Time: O(n log n), Space: O(n)

        Pattern: Sort + two pointers
        Real use: Click fraud detection, duplicate bid detection
        """
        if not transactions:
            return []

        # Sort by user_id and timestamp
        transactions.sort(key=lambda x: (x.get('user_id', ''), x.get('timestamp', 0)))

        duplicates = []
        i = 0

        while i < len(transactions):
            j = i + 1
            group = [transactions[i]]

            # Find all similar transactions within time window
            while j < len(transactions):
                if (transactions[j]['user_id'] == transactions[i]['user_id'] and
                    abs(transactions[j]['timestamp'] - transactions[i]['timestamp']) <= 60):
                    group.append(transactions[j])
                    j += 1
                else:
                    break

            if len(group) > 1:
                duplicates.append(group)

            i = j if j > i + 1 else i + 1

        return duplicates


class ArrayOptimizations:
    """Advanced array techniques for performance-critical code"""

    def kadane_with_indices(self, nums: List[int]) -> Tuple[int, int, int]:
        """
        Maximum subarray sum with start/end indices
        Time: O(n), Space: O(1)

        Real use: Best performing time window for campaigns
        """
        if not nums:
            return 0, -1, -1

        max_sum = float('-inf')
        current_sum = 0
        start = 0
        end = 0
        temp_start = 0

        for i, num in enumerate(nums):
            current_sum += num

            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i

            if current_sum < 0:
                current_sum = 0
                temp_start = i + 1

        return max_sum, start, end

    def dutch_flag_partition(self, nums: List[int], pivot: int) -> None:
        """
        3-way partition around pivot (in-place)
        Time: O(n), Space: O(1)

        Real use: Segment users (low/medium/high value)
        """
        left = 0    # Next position for values < pivot
        right = len(nums) - 1  # Next position for values > pivot
        current = 0

        while current <= right:
            if nums[current] < pivot:
                nums[left], nums[current] = nums[current], nums[left]
                left += 1
                current += 1
            elif nums[current] > pivot:
                nums[current], nums[right] = nums[right], nums[current]
                right -= 1
                # Don't increment current, need to check swapped value
            else:
                current += 1

    def reservoir_sampling(self, stream: List[int], k: int) -> List[int]:
        """
        Sample k elements from stream uniformly
        Time: O(n), Space: O(k)

        Real use: A/B test assignment, sampling for training
        """
        import random

        if k <= 0:
            return []

        reservoir = []

        for i, item in enumerate(stream):
            if i < k:
                reservoir.append(item)
            else:
                # Replace with probability k/(i+1)
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = item

        return reservoir


# Test functions with Criteo-specific examples
def test_solutions():
    """Quick test suite for interview practice"""

    solver = HashArraySolutions()
    optimizer = ArrayOptimizations()

    # Test 1: Two sum for bid optimization
    prices = [100, 200, 150, 250, 50]
    budget = 300
    pairs = solver.two_sum_all_pairs(prices, budget)
    print(f"Bid pairs summing to {budget}: {pairs}")

    # Test 2: Top K categories
    categories = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
    top_k = solver.top_k_frequent_elements(categories, 2)
    print(f"Top 2 categories: {top_k}")

    # Test 3: Rolling CTR window
    ctrs = [0.02, 0.03, 0.025, 0.04, 0.01, 0.035]
    window_max = solver.sliding_window_maximum(ctrs, 3)
    print(f"Max CTR in 3-day windows: {window_max}")

    # Test 4: Campaign performance window
    daily_revenue = [100, -20, 150, -10, 200, -50, 180]
    max_rev, start, end = optimizer.kadane_with_indices(daily_revenue)
    print(f"Best campaign period: days {start}-{end}, revenue: {max_rev}")

    # Test 5: User segmentation
    user_values = [5, 2, 8, 3, 9, 1, 7, 4, 6]
    optimizer.dutch_flag_partition(user_values, 5)
    print(f"Users segmented by value 5: {user_values}")

    # Test 6: A/B test sampling
    users = list(range(1000))
    sample = optimizer.reservoir_sampling(users, 100)
    print(f"A/B test sample size: {len(sample)}")


if __name__ == "__main__":
    test_solutions()

    print("\n✅ Hash/Array module ready!")
    print("Time complexities documented inline")
    print("Real-world Criteo use cases included")