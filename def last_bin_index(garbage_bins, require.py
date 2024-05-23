def last_bin_index(garbage_bins, required_garbage, num_bins_needed):

    garbage_bins.sort()

    count = 0
    last_bin_index = -1

    for i, bin_garbage in enumerate(garbage_bins, start=1):
        if bin_garbage == required_garbage:
            count += 1
            last_bin_index = i
            if count == num_bins_needed:
                return last_bin_index

    if count < num_bins_needed:
        return -1

    return last_bin_index

NQ = input()


N, Q = map(int, NQ.split())

garbage_bins = list(map(int, input().split()))

for _ in range(Q):
    query = input()

    required_garbage, num_bins_needed = map(int, query.split())

    last_index = last_bin_index(garbage_bins, required_garbage, num_bins_needed)

    print(last_index)


