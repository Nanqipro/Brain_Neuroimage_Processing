import sys

def solve():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    t = data[0]
    idx = 1
    out = []
    for _ in range(t):
        n, c, d = data[idx], data[idx + 1], data[idx + 2]
        idx += 3
        total = n * n
        b = data[idx:idx + total]
        idx += total
        m = min(b)
        target = []
        for i in range(n):
            base = m + i * c
            for j in range(n):
                target.append(base + j * d)
        b.sort()
        target.sort()
        out.append("YES" if b == target else "NO")
    sys.stdout.write("\n".join(out))

if __name__ == "__main__":
    solve()
