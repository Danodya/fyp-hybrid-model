from queue import *


class DataQueue:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.queue = Queue()

    def enqueue(self, item):
        self.queue.put(item)

    def get(self):
        self.queue.get(block=False)
