Here is a Python 3 script. Work out exactly what each print statement marked Q1 to Q8 outputs. You cannot run it: reason through it.

```python
def add_job(job, queue=[]):
    queue.append(job)
    return queue


def make_limiters(limits):
    return [lambda used: used < limit for limit in limits]


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class Model:
    loaded = []

    def __init__(self, name):
        self.name = name
        self.loaded.append(name)


def pick_model(free_gb, sizes):
    try:
        for name, size in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True):
            if size <= free_gb:
                return name
        return "none"
    finally:
        if free_gb < 4:
            return "too-small"


def retry_delays(base, attempts):
    delays = [base * 2 ** n for n in range(attempts)]
    return delays[::-2], sum(delays) // 3, -sum(delays) // 3


def merge(defaults, overrides):
    merged = defaults
    merged.update(overrides)
    return merged


def fastest(samples):
    rates = {name: round(tokens / seconds) for name, tokens, seconds in samples}
    best = max(rates, key=rates.get)
    return best, rates[best]


add_job("a")
add_job("b", [])
print(add_job("c"))                                             # Q1

print([check(5) for check in make_limiters([3, 8])])            # Q2

chunks = batches(list(range(7)), 3)
first = list(chunks)
print(len(first), first[-1], list(chunks))                      # Q3

Model("qwen")
m = Model("gemma")
m.loaded = m.loaded + ["llama"]
Model("mistral")
print(Model.loaded, m.loaded)                                   # Q4

sizes = {"qwen": 20.5, "gemma": 16.0, "embed": 0.6}
print(pick_model(18, sizes), pick_model(3, sizes), pick_model(0.5, sizes))    # Q5

print(retry_delays(1, 5))                                       # Q6

defaults = {"num_ctx": 32768, "temperature": 0.0}
custom = merge(defaults, {"temperature": 0.7})
custom["num_ctx"] = 65536
print(defaults["num_ctx"], defaults["temperature"])             # Q7

print(fastest([("qwen", 205, 2), ("ornith", 206, 2), ("gemma", 45, 2)]))      # Q8
```

Reply with ONLY a fenced ```json block holding one object with the keys "q1" to "q8". Write each printed value as JSON: Python lists and tuples become arrays, True and False become true and false, and strings become JSON strings. When one print statement prints several values, give an array of them in order. For example, print(1, [2, 3], "a") is [1, [2, 3], "a"] and print((4, True)) is [4, true].
