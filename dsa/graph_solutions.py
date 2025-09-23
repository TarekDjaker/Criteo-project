"""
Graph Solutions - BFS/DFS/Advanced Patterns
Optimized for Criteo Interview (30 min max per problem)
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict
import heapq


class GraphSolutions:
    """Core graph algorithms for ad-tech problems"""

    def user_similarity_network(self, edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Build adjacency list for user similarity graph
        Time: O(E), Space: O(V + E)

        Real use: User clustering, lookalike audiences
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)  # Undirected for similarity
        return graph

    def find_connected_users_bfs(self, graph: Dict[int, List[int]], start: int) -> Set[int]:
        """
        Find all users in same cluster (BFS)
        Time: O(V + E), Space: O(V)

        Pattern: Level-order traversal
        Real use: Audience expansion, segment discovery
        """
        if start not in graph:
            return set()

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            user = queue.popleft()

            for neighbor in graph[user]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def detect_click_fraud_cycle(self, clicks: List[Tuple[int, int]]) -> bool:
        """
        Detect cycles in click patterns (fraud detection)
        Time: O(V + E), Space: O(V)

        Pattern: DFS with color marking
        Real use: Click fraud, bot detection
        """
        graph = defaultdict(list)
        for source, target in clicks:
            graph[source].append(target)

        # Colors: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
        colors = defaultdict(int)

        def dfs(node: int) -> bool:
            colors[node] = 1  # Mark as visiting

            for neighbor in graph[node]:
                if colors[neighbor] == 1:  # Back edge found
                    return True
                if colors[neighbor] == 0 and dfs(neighbor):
                    return True

            colors[node] = 2  # Mark as visited
            return False

        # Check all components
        for node in graph:
            if colors[node] == 0:
                if dfs(node):
                    return True

        return False

    def campaign_dependency_order(self, campaigns: int, dependencies: List[Tuple[int, int]]) -> List[int]:
        """
        Topological sort for campaign scheduling
        Time: O(V + E), Space: O(V)

        Pattern: Kahn's algorithm (BFS-based)
        Real use: Campaign launch order, budget allocation sequence
        """
        graph = defaultdict(list)
        in_degree = [0] * campaigns

        for prereq, campaign in dependencies:
            graph[prereq].append(campaign)
            in_degree[campaign] += 1

        queue = deque([i for i in range(campaigns) if in_degree[i] == 0])
        result = []

        while queue:
            campaign = queue.popleft()
            result.append(campaign)

            for next_campaign in graph[campaign]:
                in_degree[next_campaign] -= 1
                if in_degree[next_campaign] == 0:
                    queue.append(next_campaign)

        return result if len(result) == campaigns else []  # Empty if cycle exists

    def shortest_conversion_path(self, graph: Dict[str, List[Tuple[str, float]]],
                                start: str, target: str) -> Tuple[List[str], float]:
        """
        Dijkstra for conversion path optimization
        Time: O(E log V), Space: O(V)

        Pattern: Priority queue + path reconstruction
        Real use: Attribution modeling, conversion optimization
        """
        distances = {start: 0}
        previous = {}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == target:
                # Reconstruct path
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(start)
                return path[::-1], distances[target]

            for neighbor, weight in graph.get(current, []):
                if neighbor in visited:
                    continue

                new_dist = current_dist + weight

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        return [], float('inf')

    def find_influencer_users(self, follows: List[Tuple[int, int]]) -> List[int]:
        """
        Find users with high PageRank (influencers)
        Time: O(iterations * E), Space: O(V)

        Pattern: PageRank algorithm
        Real use: Influencer targeting, seed user selection
        """
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)

        for follower, followed in follows:
            graph[follower].append(followed)
            reverse_graph[followed].append(follower)

        # Get all users
        all_users = set()
        for u, v in follows:
            all_users.add(u)
            all_users.add(v)

        # Initialize PageRank
        n = len(all_users)
        pagerank = {user: 1.0 / n for user in all_users}
        damping = 0.85
        iterations = 10

        for _ in range(iterations):
            new_pagerank = {}

            for user in all_users:
                rank = (1 - damping) / n

                for follower in reverse_graph[user]:
                    out_degree = len(graph[follower])
                    if out_degree > 0:
                        rank += damping * pagerank[follower] / out_degree

                new_pagerank[user] = rank

            pagerank = new_pagerank

        # Return top influencers
        sorted_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return [user for user, _ in sorted_users[:10]]

    def bipartite_matching(self, users: List[int], ads: List[int],
                          preferences: List[Tuple[int, int]]) -> Dict[int, int]:
        """
        Maximum bipartite matching (user-ad assignment)
        Time: O(V * E), Space: O(V)

        Pattern: Augmenting paths (simplified Hungarian)
        Real use: Ad assignment, budget allocation
        """
        graph = defaultdict(list)
        for user, ad in preferences:
            graph[user].append(ad)

        match_user_to_ad = {}
        match_ad_to_user = {}

        def dfs(user: int, visited: Set[int]) -> bool:
            for ad in graph[user]:
                if ad in visited:
                    continue

                visited.add(ad)

                if ad not in match_ad_to_user or dfs(match_ad_to_user[ad], visited):
                    match_user_to_ad[user] = ad
                    match_ad_to_user[ad] = user
                    return True

            return False

        for user in users:
            visited = set()
            dfs(user, visited)

        return match_user_to_ad


class UnionFind:
    """Union-Find for clustering and component analysis"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        """Path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank"""
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.components -= 1
        return True

    def get_components(self) -> int:
        return self.components


def test_graph_solutions():
    """Test suite for graph algorithms"""

    solver = GraphSolutions()

    # Test 1: User similarity network
    print("=== User Similarity Network ===")
    edges = [(1, 2), (2, 3), (1, 3), (4, 5)]
    graph = solver.user_similarity_network(edges)
    cluster = solver.find_connected_users_bfs(graph, 1)
    print(f"User 1's cluster: {cluster}")

    # Test 2: Click fraud detection
    print("\n=== Click Fraud Detection ===")
    suspicious_clicks = [(1, 2), (2, 3), (3, 1)]  # Cycle
    has_fraud = solver.detect_click_fraud_cycle(suspicious_clicks)
    print(f"Fraud detected: {has_fraud}")

    # Test 3: Campaign scheduling
    print("\n=== Campaign Dependencies ===")
    dependencies = [(0, 1), (0, 2), (1, 3), (2, 3)]
    order = solver.campaign_dependency_order(4, dependencies)
    print(f"Campaign launch order: {order}")

    # Test 4: Conversion path
    print("\n=== Conversion Path Optimization ===")
    conversion_graph = {
        'homepage': [('product', 0.3), ('category', 0.5)],
        'category': [('product', 0.2), ('cart', 0.1)],
        'product': [('cart', 0.4)],
        'cart': [('checkout', 0.7)]
    }
    path, cost = solver.shortest_conversion_path(conversion_graph, 'homepage', 'checkout')
    print(f"Optimal path: {path}, Cost: {cost}")

    # Test 5: Influencer detection
    print("\n=== Influencer Detection ===")
    follows = [(1, 2), (1, 3), (2, 3), (4, 3), (5, 3)]
    influencers = solver.find_influencer_users(follows)
    print(f"Top influencers: {influencers[:3]}")

    # Test 6: User-Ad matching
    print("\n=== User-Ad Matching ===")
    users = [1, 2, 3]
    ads = [101, 102, 103]
    preferences = [(1, 101), (1, 102), (2, 102), (3, 103)]
    matching = solver.bipartite_matching(users, ads, preferences)
    print(f"User-Ad assignments: {matching}")

    # Test 7: Union-Find for clustering
    print("\n=== Union-Find Clustering ===")
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    print(f"Number of user clusters: {uf.get_components()}")


if __name__ == "__main__":
    test_graph_solutions()

    print("\n✅ Graph module ready!")
    print("BFS/DFS/Advanced patterns implemented")
    print("Real-world ad-tech use cases included")