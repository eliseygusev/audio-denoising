Находим в массиве непрерывную подпоследовательность, имеющую наибольшую сумму.

```python

def max_subarray(numbers):
    if len(numbers) == 0:
        return []

    if max(numbers) <= 0:
        return [max(numbers)]

    best_sum = 0
    cur_sum = 0

    best_start = None
    best_end = None

    for cur_end, elem in enumerate(numbers):
        if cur_sum > 0:
            cur_sum += elem
        else:
            # if cur_sum < 0 then it doesn't make sense
            # to continue considering subsequence
            cur_start = cur_end
            cur_sum = elem

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start = cur_start
            best_end = cur_end

    return numbers[best_start: best_end + 1]

```
