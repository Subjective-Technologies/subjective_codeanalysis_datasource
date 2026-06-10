def quicksort(values):
    if len(values) <= 1:
        return values
    pivot = values[len(values) // 2]
    smaller = [v for v in values if v < pivot]
    equal = [v for v in values if v == pivot]
    larger = [v for v in values if v > pivot]
    return quicksort(smaller) + equal + quicksort(larger)
