import time

print("start", flush=True)
t0 = time.time()

import tensorflow as tf

print("tensorflow imported", flush=True)
print(tf.__version__, flush=True)
print("seconds:", time.time() - t0, flush=True)