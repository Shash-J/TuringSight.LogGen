from queue import PriorityQueue


class TaskPriorityQueue:
    def __init__(self):
        self.q = PriorityQueue()
        self.counter = 0

    def put(self, task: dict):
        self.counter += 1
        self.q.put((task["priority"], self.counter, task))

    def get(self):
        _, _, task = self.q.get()
        return task

    def empty(self):
        return self.q.empty()

    def qsize(self):
        return self.q.qsize()