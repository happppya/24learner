import threading
import time

from learner.inference import BatchingQueue, resolve_batch


def test_full_batch_flushes_immediately():
    queue = BatchingQueue(max_size=4)
    futures = [queue.submit(item) for item in range(4)]
    batch = queue.drain(blocking=False)
    assert batch is not None
    assert batch.items == [0, 1, 2, 3]
    resolve_batch(batch, lambda x: x * 2)
    assert [f.result() for f in futures] == [0, 2, 4, 6]


def test_idle_drain_blocks_until_timeout():
    queue = BatchingQueue(max_size=8, timeout_s=0.02)
    start = time.perf_counter()
    batch = queue.drain(blocking=True)
    elapsed = time.perf_counter() - start
    assert batch is None
    assert elapsed >= 0.015

    queue.submit("instant")
    start = time.perf_counter()
    batch = queue.drain(blocking=True)
    assert time.perf_counter() - start < 0.015
    assert batch is not None and batch.items == ["instant"]


def test_nonblocking_drain_on_empty_returns_none():
    queue = BatchingQueue(max_size=4)
    assert queue.drain(blocking=False) is None


def test_close_rejects_new_submits_but_drains_remaining():
    queue = BatchingQueue(max_size=2)
    future = queue.submit("kept")
    queue.close()
    try:
        queue.submit("too-late")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    batch = queue.drain(blocking=False)
    assert batch is not None and batch.items == ["kept"]
    resolve_batch(batch, lambda x: x)
    assert future.result() == "kept"
    assert queue.drain(blocking=True) is None


def test_invalid_max_size_rejected():
    try:
        BatchingQueue(max_size=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_concurrent_producers_single_consumer():
    queue = BatchingQueue(max_size=32, timeout_s=0.001)
    producers = 4
    per_producer = 250
    total = producers * per_producer
    futures = []

    def produce(seed_offset):
        for k in range(per_producer):
            futures.append(queue.submit(seed_offset * per_producer + k))

    threads = [threading.Thread(target=produce, args=(p,)) for p in range(producers)]
    consumer_done = threading.Event()

    def consume():
        seen = 0
        while True:
            batch = queue.drain(blocking=True)
            if batch is None:
                if queue.is_closed():
                    break
                continue
            resolve_batch(batch, lambda x: x + 1)
            seen += len(batch.items)
        assert seen == total
        consumer_done.set()

    consumer = threading.Thread(target=consume)
    consumer.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    queue.close()
    consumer_done.wait(timeout=5)

    results = sorted(f.result() for f in futures)
    assert results == list(range(1, total + 1))
