def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return primes


def main():
    t = int(input())
    test_cases = [int(input()) for _ in range(t)]
    max_n = max(test_cases)

    # Precompute primes up to max_n.
    primes = sieve(max_n)

    results = []
    for n in test_cases:
        total = 0
        for p in primes:
            if p > n:
                break
            total += n // p
        results.append(str(total))

    print("\n".join(results))


if __name__ == '__main__':
    main()