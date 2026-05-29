import time

class TimeRegister():
    def __init__(self):
        self.time_log = {}

    def register(self, cls, name):
        if not cls in self.time_log:
            self.time_log[cls] = {}
        
        if not name in self.time_log[cls]:
            self.time_log[cls][name] = []
        st = time.time()
        self.time_log[cls][name].append(st)

    def deregister(self, cls, name):
        et = time.time()
        st = self.time_log[cls][name].pop()
        elapsed = et - st
        self.time_log[cls][name].append(elapsed)

    def report(self):
        print("=== Time Report ===")
        for cls, entries in self.time_log.items():
            print(f"Class: {cls}")
            for name, times in entries.items():
                times = times[10:]  # Skip the first 10 runs to allow for warm-up
                total_time = sum(times)
                avg_time = total_time / len(times)
                print(f"  {name}: Total={total_time:.4f}s, Avg={avg_time:.4f}s over {len(times)} runs")

    def reset(self):
        self.time_log = {}