"""
FLOOD FILL ALGORITHM - CRITEO INTERVIEW CRITICAL
Multiple implementations with optimizations
Time/Space complexity analysis included
"""

from collections import deque
from typing import List, Tuple, Set
import time
import numpy as np


class FloodFillSolutions:
    """Complete Flood Fill implementations for technical interviews"""
    
    @staticmethod
    def flood_fill_bfs(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        BFS Implementation (MOST COMMON IN INTERVIEWS)
        Time: O(m*n) where m,n are dimensions
        Space: O(m*n) for queue in worst case
        
        Used in: Image processing, game development, segmentation
        """
        if not image or not image[0]:
            return image
            
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        # Edge case: same color
        if original_color == new_color:
            return image
        
        # BFS with queue
        queue = deque([(sr, sc)])
        image[sr][sc] = new_color
        
        # 4-directional movement
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check boundaries and color match
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    image[nr][nc] == original_color):
                    
                    image[nr][nc] = new_color
                    queue.append((nr, nc))
        
        return image
    
    @staticmethod
    def flood_fill_dfs_recursive(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        DFS Recursive Implementation
        Time: O(m*n)
        Space: O(m*n) for recursion stack
        
        Simpler but can cause stack overflow for large images
        """
        if not image or not image[0]:
            return image
            
        original_color = image[sr][sc]
        
        if original_color == new_color:
            return image
        
        def dfs(r: int, c: int):
            # Out of bounds or wrong color
            if (r < 0 or r >= len(image) or 
                c < 0 or c >= len(image[0]) or 
                image[r][c] != original_color):
                return
            
            image[r][c] = new_color
            
            # Recurse in 4 directions
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        dfs(sr, sc)
        return image
    
    @staticmethod
    def flood_fill_dfs_iterative(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        DFS Iterative with Stack
        Time: O(m*n)
        Space: O(m*n)
        
        Avoids recursion limit issues
        """
        if not image or not image[0]:
            return image
            
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        if original_color == new_color:
            return image
        
        stack = [(sr, sc)]
        
        while stack:
            r, c = stack.pop()
            
            if (r < 0 or r >= rows or 
                c < 0 or c >= cols or 
                image[r][c] != original_color):
                continue
            
            image[r][c] = new_color
            
            # Add neighbors to stack
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        
        return image
    
    @staticmethod
    def flood_fill_optimized(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        OPTIMIZED VERSION - Production Ready
        - Uses visited set to avoid revisiting
        - Boundary checking optimization
        - Memory efficient
        
        Time: O(m*n)
        Space: O(min(m*n, P)) where P is pixels to fill
        """
        if not image or not image[0]:
            return image
            
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        if original_color == new_color:
            return image
        
        # Use set for O(1) lookups
        visited = set()
        queue = deque([(sr, sc)])
        visited.add((sr, sc))
        
        while queue:
            r, c = queue.popleft()
            image[r][c] = new_color
            
            # Check all 4 neighbors
            for nr, nc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    image[nr][nc] == original_color):
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return image
    
    @staticmethod
    def flood_fill_8_directional(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
        """
        8-Directional Flood Fill (includes diagonals)
        Used in: Advanced image processing, game AI
        
        Time: O(m*n)
        Space: O(m*n)
        """
        if not image or not image[0]:
            return image
            
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        if original_color == new_color:
            return image
        
        # 8 directions including diagonals
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        queue = deque([(sr, sc)])
        visited = {(sr, sc)}
        
        while queue:
            r, c = queue.popleft()
            image[r][c] = new_color
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    image[nr][nc] == original_color):
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return image
    
    @staticmethod
    def flood_fill_with_boundary(image: List[List[int]], sr: int, sc: int, 
                                new_color: int, boundary_color: int) -> List[List[int]]:
        """
        Flood Fill with Boundary Detection
        Stops at specified boundary color
        
        Used in: Paint bucket with boundaries, region segmentation
        """
        if not image or not image[0]:
            return image
            
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        if original_color == new_color or original_color == boundary_color:
            return image
        
        queue = deque([(sr, sc)])
        visited = {(sr, sc)}
        
        while queue:
            r, c = queue.popleft()
            image[r][c] = new_color
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    image[nr][nc] != boundary_color and 
                    image[nr][nc] == original_color):
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return image
    
    @staticmethod
    def count_islands(grid: List[List[int]]) -> int:
        """
        RELATED PROBLEM: Count Islands using Flood Fill
        Common follow-up question in interviews
        
        Time: O(m*n)
        Space: O(min(m*n, P)) where P is largest island
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        islands = 0
        
        def bfs(r: int, c: int):
            queue = deque([(r, c)])
            visited.add((r, c))
            
            while queue:
                curr_r, curr_c = queue.popleft()
                
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    
                    if (0 <= nr < rows and 
                        0 <= nc < cols and 
                        (nr, nc) not in visited and 
                        grid[nr][nc] == 1):
                        
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i, j) not in visited:
                    bfs(i, j)
                    islands += 1
        
        return islands
    
    @staticmethod
    def get_region_size(image: List[List[int]], sr: int, sc: int) -> int:
        """
        Get size of connected region
        Useful for: Image analysis, game mechanics
        
        Time: O(m*n)
        Space: O(m*n)
        """
        if not image or not image[0]:
            return 0
        
        rows, cols = len(image), len(image[0])
        target_color = image[sr][sc]
        visited = set()
        
        queue = deque([(sr, sc)])
        visited.add((sr, sc))
        size = 0
        
        while queue:
            r, c = queue.popleft()
            size += 1
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    image[nr][nc] == target_color):
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return size
    
    @staticmethod
    def flood_fill_multi_source(image: List[List[int]], sources: List[Tuple[int, int]], 
                               new_color: int) -> List[List[int]]:
        """
        Multi-source Flood Fill (starts from multiple points)
        Used in: Parallel processing, distance maps
        
        Time: O(m*n)
        Space: O(m*n)
        """
        if not image or not image[0] or not sources:
            return image
        
        rows, cols = len(image), len(image[0])
        
        # Get original colors from all sources
        original_colors = {image[sr][sc] for sr, sc in sources}
        
        # Initialize queue with all sources
        queue = deque(sources)
        visited = set(sources)
        
        # Mark all sources with new color
        for sr, sc in sources:
            image[sr][sc] = new_color
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 
                    0 <= nc < cols and 
                    (nr, nc) not in visited and 
                    image[nr][nc] in original_colors):
                    
                    visited.add((nr, nc))
                    image[nr][nc] = new_color
                    queue.append((nr, nc))
        
        return image


def run_performance_tests():
    """Performance comparison of different implementations"""
    print("=" * 60)
    print("FLOOD FILL PERFORMANCE TESTING")
    print("=" * 60 + "\n")
    
    # Test different grid sizes
    sizes = [(10, 10), (50, 50), (100, 100), (200, 200)]
    
    for rows, cols in sizes:
        print(f"\nGrid Size: {rows}x{cols}")
        print("-" * 40)
        
        # Create test image
        image = [[1 for _ in range(cols)] for _ in range(rows)]
        
        # Add some variation
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 3 == 0:
                    image[i][j] = 0
        
        solver = FloodFillSolutions()
        
        # Test BFS
        test_image = [row[:] for row in image]
        start = time.time()
        solver.flood_fill_bfs(test_image, 0, 0, 2)
        bfs_time = (time.time() - start) * 1000
        print(f"  BFS:        {bfs_time:.3f}ms")
        
        # Test DFS Iterative
        test_image = [row[:] for row in image]
        start = time.time()
        solver.flood_fill_dfs_iterative(test_image, 0, 0, 2)
        dfs_time = (time.time() - start) * 1000
        print(f"  DFS (iter): {dfs_time:.3f}ms")
        
        # Test Optimized
        test_image = [row[:] for row in image]
        start = time.time()
        solver.flood_fill_optimized(test_image, 0, 0, 2)
        opt_time = (time.time() - start) * 1000
        print(f"  Optimized:  {opt_time:.3f}ms")
        
        # For smaller grids, test recursive
        if rows <= 100:
            test_image = [row[:] for row in image]
            start = time.time()
            solver.flood_fill_dfs_recursive(test_image, 0, 0, 2)
            rec_time = (time.time() - start) * 1000
            print(f"  DFS (rec):  {rec_time:.3f}ms")


def interview_examples():
    """Common interview test cases"""
    print("\n" + "=" * 60)
    print("INTERVIEW TEST CASES")
    print("=" * 60 + "\n")
    
    solver = FloodFillSolutions()
    
    # Test Case 1: Basic flood fill
    print("Test 1: Basic Flood Fill")
    image1 = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1]
    ]
    print("Original:")
    for row in image1:
        print(row)
    
    result1 = solver.flood_fill_bfs([row[:] for row in image1], 1, 1, 2)
    print("\nAfter flood fill (1,1) -> 2:")
    for row in result1:
        print(row)
    
    # Test Case 2: Island counting
    print("\n" + "-" * 40)
    print("Test 2: Count Islands")
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    print("Grid:")
    for row in grid:
        print(row)
    print(f"Number of islands: {solver.count_islands(grid)}")
    
    # Test Case 3: Region size
    print("\n" + "-" * 40)
    print("Test 3: Get Region Size")
    image3 = [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    print("Image:")
    for row in image3:
        print(row)
    print(f"Size of region at (0,0): {solver.get_region_size(image3, 0, 0)}")
    print(f"Size of region at (2,2): {solver.get_region_size(image3, 2, 2)}")
    
    # Test Case 4: Edge cases
    print("\n" + "-" * 40)
    print("Test 4: Edge Cases")
    
    # Empty image
    print("Empty image:", solver.flood_fill_bfs([], 0, 0, 1))
    
    # Single cell
    single = [[5]]
    print("Single cell:", solver.flood_fill_bfs(single, 0, 0, 3))
    
    # Same color (no change)
    same_color = [[1, 1], [1, 1]]
    result = solver.flood_fill_bfs([row[:] for row in same_color], 0, 0, 1)
    print("Same color:", result)


def complexity_analysis():
    """Detailed complexity analysis"""
    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60 + "\n")
    
    analysis = """
    FLOOD FILL COMPLEXITY BREAKDOWN:
    
    1. BFS Implementation:
       - Time:  O(m × n) - visits each pixel at most once
       - Space: O(min(m × n, p)) where p = pixels to fill
       - Best for: Large connected regions, guaranteed shortest path
    
    2. DFS Recursive:
       - Time:  O(m × n)
       - Space: O(m × n) - recursion stack in worst case
       - Risk:  Stack overflow for large images (>1000×1000)
       - Best for: Small images, simple implementation
    
    3. DFS Iterative:
       - Time:  O(m × n)
       - Space: O(min(m × n, p))
       - Best for: Avoiding recursion limits, memory control
    
    4. Optimized with Visited Set:
       - Time:  O(m × n)
       - Space: O(min(m × n, p))
       - Optimization: O(1) visited check vs O(1) color check
       - Best for: Production systems, repeated operations
    
    REAL-WORLD APPLICATIONS AT CRITEO:
    
    1. User Segment Discovery:
       - Find connected user groups with similar behavior
       - Fill regions in feature space
    
    2. Ad Creative Analysis:
       - Identify dominant colors in advertisements
       - Segment image regions for A/B testing
    
    3. Fraud Detection:
       - Identify clusters of suspicious activity
       - Fill connected components in interaction graphs
    
    4. Geographic Targeting:
       - Define marketing regions based on performance
       - Expand successful campaign areas
    
    OPTIMIZATION TIPS FOR INTERVIEWS:
    
    1. Early termination: Check if new_color == old_color
    2. Boundary optimization: Check bounds before adding to queue
    3. Memory optimization: Use bit manipulation for visited tracking
    4. Parallel processing: Multi-source flood fill for large images
    5. Cache locality: Process in row-major order when possible
    """
    
    print(analysis)


if __name__ == "__main__":
    print("\n🎯 FLOOD FILL ALGORITHM - CRITEO INTERVIEW PREP\n")
    
    # Run performance tests
    run_performance_tests()
    
    # Show interview examples
    interview_examples()
    
    # Display complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ FLOOD FILL MODULE READY!")
    print("=" * 60)
    print("\nKey Takeaways for Criteo Interview:")
    print("  • BFS is preferred for production (more predictable)")
    print("  • Always handle edge cases (empty, same color)")
    print("  • Consider memory constraints for large datasets")
    print("  • Be ready to discuss real-world applications")
    print("  • Know the complexity analysis by heart")