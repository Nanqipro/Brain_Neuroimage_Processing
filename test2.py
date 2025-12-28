import sys
from collections import deque
def solve():
    input = sys.stdin.readline
    first = input().split()
    if not first:
        return
    n, m = map(int, first)
    grid = [input().strip() for _ in range(n)]
    # 2. 预处理：BFS标记连通块
    comp_id = [[0] * m for _ in range(n)] # 记录每个格子属于哪个连通块ID
    comp_size = {}                        # 记录每个ID的面积
    curr_id = 1
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 'o' and comp_id[r][c] == 0:
                # 发现新的水田块，BFS遍历
                q = deque([(r, c)])
                comp_id[r][c] = curr_id
                size = 0
                while q:
                    x, y = q.popleft()
                    size += 1
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < m:
                            if grid[nx][ny] == 'o' and comp_id[nx][ny] == 0:
                                comp_id[nx][ny] = curr_id
                                q.append((nx, ny))
                comp_size[curr_id] = size
                curr_id += 1
    # 3. 计算输出
    res = []
    for r in range(n):
        row = []
        for c in range(m):
            if grid[r][c] == 'o':
                row.append("0")
            else:
                # 旱田：自身1 + 四周不同连通块的面积
                seen = set()
                total = 1
                for dx, dy in dirs:
                    nx, ny = r + dx, c + dy
                    if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 'o':
                        gid = comp_id[nx][ny]
                        if gid not in seen:
                            total += comp_size[gid]
                            seen.add(gid)
                row.append(str(total))
        res.append(" ".join(row))
    print("\n".join(res))
if __name__ == '__main__':
    solve()
