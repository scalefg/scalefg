# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
"""
Implementation of a thread-safe queue with one producer and one consumer.
"""


class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def enqueue(self, task):
        self.cv.acquire()
        self.queue.append(task)
        self.cv.notify()
        self.cv.release()

    # TODO(epoll)
    def dequeue(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        task = self.queue.pop(0)
        self.cv.release()
        return task

    def clear(self):
        self.cv.acquire()
        self.queue.clear()
        self.cv.release()

    def len(self):
        return len(self.queue)