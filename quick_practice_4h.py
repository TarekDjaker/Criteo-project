#!/usr/bin/env python3
"""
🚀 QUICK PRACTICE - 4H SPRINT CRITEO
Tous les problèmes essentiels en un seul fichier
Exécutez et pratiquez directement!
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import hashlib


print("=" * 80)
print("⚡ CRITEO INTERVIEW - QUICK PRACTICE SESSION")
print("=" * 80)
print("\nRésolvez chaque problème en temps limité!")
print("Les solutions sont fournies - comparez après avoir essayé.\n")

# ==============================================================================
# SECTION 1: DSA PROBLEMS
# ==============================================================================

class DSAPractice:
    """Problèmes DSA essentiels pour Criteo"""

    @staticmethod
    def practice_two_sum_all_pairs():
        """Problem 1: Two Sum All Pairs (12 min max)"""
        print("\n" + "="*60)
        print("📝 PROBLEM 1: Two Sum All Pairs")
        print("Time limit: 12 minutes")
        print("-"*60)

        print("""
Task: Find ALL pairs of indices that sum to target
Use case: Find ad pairs within budget

Example:
Input: nums = [100, 200, 100, 150], target = 300
Output: [(0,1), (2,1)]  # nums[0] + nums[1] = 300, nums[2] + nums[1] = 300
        """)

        # VOTRE SOLUTION ICI
        def two_sum_all_pairs(nums: List[int], target: int) -> List[Tuple[int, int]]:
            """
            TODO: Implement this
            Time: O(n), Space: O(n)
            """
            # START CODING HERE
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
            # END CODING

        # Tests
        test_cases = [
            ([100, 200, 100, 150], 300),
            ([1, 2, 3, 4, 5], 5),
            ([2, 2, 2, 2], 4),
        ]

        print("\nTesting your solution:")
        for nums, target in test_cases:
            result = two_sum_all_pairs(nums, target)
            print(f"Input: {nums}, Target: {target}")
            print(f"Output: {result}\n")

        return two_sum_all_pairs

    @staticmethod
    def practice_sliding_window_max():
        """Problem 2: Sliding Window Maximum (13 min max)"""
        print("\n" + "="*60)
        print("📝 PROBLEM 2: Sliding Window Maximum")
        print("Time limit: 13 minutes")
        print("-"*60)

        print("""
Task: Find maximum in each window of size k
Use case: Max CTR in rolling time windows

Example:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
        """)

        def sliding_window_max(nums: List[int], k: int) -> List[int]:
            """
            TODO: Implement using monotonic deque
            Time: O(n), Space: O(k)
            """
            # START CODING HERE
            from collections import deque

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
            # END CODING

        # Test
        test_nums = [1, 3, -1, -3, 5, 3, 6, 7]
        k = 3
        result = sliding_window_max(test_nums, k)
        print(f"\nInput: {test_nums}, k={k}")
        print(f"Output: {result}")
        print(f"Expected: [3, 3, 5, 5, 6, 7]")

        return sliding_window_max

    @staticmethod
    def practice_user_clustering():
        """Problem 3: User Network Clustering (15 min max)"""
        print("\n" + "="*60)
        print("📝 PROBLEM 3: User Network Clustering")
        print("Time limit: 15 minutes")
        print("-"*60)

        print("""
Task: Find all connected user clusters
Use case: Lookalike audience generation

Example:
Input: edges = [(1,2), (2,3), (4,5)]
Output: [[1,2,3], [4,5]]
        """)

        def find_user_clusters(edges: List[Tuple[int, int]]) -> List[List[int]]:
            """
            TODO: Implement using BFS/DFS
            """
            # START CODING HERE
            from collections import defaultdict, deque

            # Build graph
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            visited = set()
            clusters = []

            def bfs(start):
                cluster = []
                queue = deque([start])
                visited.add(start)

                while queue:
                    user = queue.popleft()
                    cluster.append(user)

                    for neighbor in graph[user]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                return cluster

            # Find all users
            all_users = set()
            for u, v in edges:
                all_users.add(u)
                all_users.add(v)

            # Find clusters
            for user in all_users:
                if user not in visited:
                    cluster = bfs(user)
                    clusters.append(sorted(cluster))

            return clusters
            # END CODING

        # Test
        edges = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8)]
        result = find_user_clusters(edges)
        print(f"\nEdges: {edges}")
        print(f"Clusters: {result}")
        print(f"Expected: [[1,2,3], [4,5,6], [7,8]]")

        return find_user_clusters


# ==============================================================================
# SECTION 2: CTR & ML PATTERNS
# ==============================================================================

class CTRPractice:
    """CTR modeling patterns essentiels"""

    @staticmethod
    def practice_hash_trick():
        """Hash Trick Implementation (5 min)"""
        print("\n" + "="*60)
        print("📝 HASH TRICK PATTERN")
        print("Time limit: 5 minutes")
        print("-"*60)

        def hash_trick(value: str, buckets: int = 1_000_000) -> int:
            """
            Implement hash trick for high cardinality features
            """
            if pd.isna(value) or value is None:
                return 0

            hash_val = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
            return hash_val % buckets

        # Test
        test_values = ["user_123", "user_456", None, "user_123", ""]
        print("\nHash Trick Results:")
        for val in test_values:
            hashed = hash_trick(val)
            print(f"  '{val}' -> {hashed}")

        return hash_trick

    @staticmethod
    def practice_metrics():
        """CTR Metrics Calculation (5 min)"""
        print("\n" + "="*60)
        print("📝 CTR METRICS CALCULATION")
        print("Time limit: 5 minutes")
        print("-"*60)

        def calculate_metrics(y_true, y_pred):
            """Calculate key CTR metrics"""
            from sklearn.metrics import log_loss, roc_auc_score

            metrics = {
                'logloss': log_loss(y_true, y_pred),
                'auc': roc_auc_score(y_true, y_pred),
                'calibration': y_pred.mean() / (y_true.mean() + 1e-10)
            }

            # Lift @ 10%
            threshold = np.percentile(y_pred, 90)
            top_10_ctr = y_true[y_pred >= threshold].mean()
            baseline_ctr = y_true.mean()
            metrics['lift_10'] = top_10_ctr / (baseline_ctr + 1e-10)

            return metrics

        # Test with synthetic data
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.05, 1000)  # 5% CTR
        y_pred = np.random.beta(2, 20, 1000)  # Predictions

        metrics = calculate_metrics(y_true, y_pred)
        print("\nMetrics on synthetic data:")
        for metric, value in metrics.items():
            target = {"logloss": "< 0.44", "auc": "> 0.80",
                     "calibration": "≈ 1.0", "lift_10": "> 2.0"}
            print(f"  {metric}: {value:.3f} (target: {target[metric]})")

        return calculate_metrics


# ==============================================================================
# SECTION 3: BIDDING PATTERNS
# ==============================================================================

class BiddingPractice:
    """First-price bidding patterns"""

    @staticmethod
    def practice_bid_shading():
        """Bid Shading Calculation (5 min)"""
        print("\n" + "="*60)
        print("📝 BID SHADING CALCULATION")
        print("Time limit: 5 minutes")
        print("-"*60)

        def calculate_shaded_bid(pCTR, pCVR, value, margin=0.3):
            """
            Calculate bid with shading for first-price auction
            """
            # Expected value
            expected_value = pCTR * pCVR * value

            # Max bid (with margin)
            max_bid = expected_value * (1 - margin)

            # Shading factor (typically 0.75-0.85)
            shading_factor = 0.8

            # Final bid
            shaded_bid = max_bid * shading_factor

            return {
                'expected_value': expected_value,
                'max_bid': max_bid,
                'shaded_bid': shaded_bid,
                'savings': max_bid - shaded_bid,
                'savings_pct': (max_bid - shaded_bid) / max_bid * 100
            }

        # Test scenarios
        scenarios = [
            {'pCTR': 0.05, 'pCVR': 0.02, 'value': 50},  # Standard
            {'pCTR': 0.10, 'pCVR': 0.01, 'value': 100},  # High CTR
            {'pCTR': 0.02, 'pCVR': 0.05, 'value': 200},  # High CVR
        ]

        print("\nBid Shading Results:")
        for i, params in enumerate(scenarios, 1):
            result = calculate_shaded_bid(**params)
            print(f"\nScenario {i}: pCTR={params['pCTR']}, pCVR={params['pCVR']}, Value=${params['value']}")
            print(f"  Max bid: ${result['max_bid']:.2f}")
            print(f"  Shaded bid: ${result['shaded_bid']:.2f}")
            print(f"  Savings: ${result['savings']:.2f} ({result['savings_pct']:.1f}%)")

        return calculate_shaded_bid


# ==============================================================================
# SECTION 4: QUICK QUIZ
# ==============================================================================

def quick_quiz():
    """Test rapide de connaissances (5 min)"""
    print("\n" + "="*80)
    print("🧠 QUICK KNOWLEDGE QUIZ")
    print("="*80)

    questions = [
        {
            "q": "1. Criteo dataset size?",
            "a": "45M samples, 13 numerical + 26 categorical features"
        },
        {
            "q": "2. Target latency for DeepKNN?",
            "a": "< 50ms p99"
        },
        {
            "q": "3. When did industry move to first-price?",
            "a": "2019"
        },
        {
            "q": "4. Demographic Parity threshold?",
            "a": "< 0.7%"
        },
        {
            "q": "5. Bid shading savings at Criteo?",
            "a": "15-20%"
        },
        {
            "q": "6. LogLoss target?",
            "a": "< 0.44"
        },
        {
            "q": "7. AUC target?",
            "a": "> 0.80"
        },
        {
            "q": "8. DeepKNN components?",
            "a": "Two-tower encoders + Vector DB (Faiss)"
        }
    ]

    print("\nAnswer mentally, then press Enter to see the answer:\n")
    score = 0

    for q_dict in questions:
        print(f"❓ {q_dict['q']}")
        input()
        print(f"✅ {q_dict['a']}\n")

        correct = input("Did you get it right? (y/n): ").strip().lower()
        if correct == 'y':
            score += 1

    print(f"\n📊 Your Score: {score}/{len(questions)}")

    if score == len(questions):
        print("🎉 Perfect! You're ready!")
    elif score >= len(questions) * 0.8:
        print("👍 Great job! Review the missed ones.")
    else:
        print("📚 Keep studying! You'll get there!")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run the complete practice session"""

    print("\n" + "="*80)
    print("🚀 STARTING 4H PRACTICE SESSION")
    print("="*80)

    # Track timing
    start_time = time.time()

    # DSA Practice
    print("\n\n" + "🔷"*30)
    print("PART 1: DSA PROBLEMS")
    print("🔷"*30)

    dsa = DSAPractice()
    dsa.practice_two_sum_all_pairs()
    input("\n✅ Press Enter to continue...")

    dsa.practice_sliding_window_max()
    input("\n✅ Press Enter to continue...")

    dsa.practice_user_clustering()
    input("\n✅ Press Enter to continue...")

    # CTR Practice
    print("\n\n" + "🔷"*30)
    print("PART 2: CTR & ML")
    print("🔷"*30)

    ctr = CTRPractice()
    ctr.practice_hash_trick()
    input("\n✅ Press Enter to continue...")

    ctr.practice_metrics()
    input("\n✅ Press Enter to continue...")

    # Bidding Practice
    print("\n\n" + "🔷"*30)
    print("PART 3: BIDDING")
    print("🔷"*30)

    bidding = BiddingPractice()
    bidding.practice_bid_shading()
    input("\n✅ Press Enter to continue...")

    # Quick Quiz
    print("\n\n" + "🔷"*30)
    print("PART 4: KNOWLEDGE CHECK")
    print("🔷"*30)

    quick_quiz()

    # Summary
    elapsed = (time.time() - start_time) / 60
    print("\n" + "="*80)
    print("✅ PRACTICE SESSION COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed:.1f} minutes")

    print("\n📋 CHECKLIST:")
    print("✅ DSA patterns practiced")
    print("✅ Hash trick implemented")
    print("✅ Metrics calculated")
    print("✅ Bid shading understood")
    print("✅ Knowledge tested")

    print("\n🎯 NEXT STEPS:")
    print("1. Review any weak areas")
    print("2. Practice the 60-second pitch")
    print("3. Prepare your questions")
    print("4. Get a good night's sleep")

    print("\n💪 YOU'RE READY TO ACE THAT INTERVIEW!")


if __name__ == "__main__":
    main()