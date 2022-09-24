# matrix-mul-test

To run javascript:

- `cd js`
- `npm install`
- `npm run start`

To run python:

- `cd py`
- `python -m venv venv`
- `source venv/bin/activate` (or `source venv/bin/activate.fish` if using fish)
- `pip install -r requirements.txt`
- `python3 main.py`


Results (JS):

```
default:                  avg: 470.10ms    std: 3.37ms      p0: 465.00ms     p5: 466.00ms     p25: 467.00ms    p50: 470.00ms    p75: 474.00ms    p95: 476.00ms    p100: 476.00ms  
GPU.js single:            avg: 6.30ms      std: 3.95ms      p0: 23.00ms      p5: 4.00ms       p25: 5.00ms      p50: 5.00ms      p75: 6.00ms      p95: 8.00ms      p100: 8.00ms      FAIL: greatest diff is 2.18e+0 on value 1.37e+6
GPU.js unsigned:          avg: 5.15ms      std: 3.93ms      p0: 22.00ms      p5: 3.00ms       p25: 4.00ms      p50: 4.00ms      p75: 4.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 2.43e+0 on value 1.37e+6
GPU.js single:            avg: 4.90ms      std: 0.77ms      p0: 4.00ms       p5: 4.00ms       p25: 4.00ms      p50: 5.00ms      p75: 5.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.18e+0 on value 1.37e+6
ml-matrix:                avg: 132.55ms    std: 7.16ms      p0: 129.00ms     p5: 129.00ms     p25: 129.00ms    p50: 130.00ms    p75: 131.00ms    p95: 156.00ms    p100: 156.00ms  
tfjs-node-f32:            avg: 1.35ms      std: 1.11ms      p0: 1.00ms       p5: 1.00ms       p25: 1.00ms      p50: 1.00ms      p75: 1.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 1.06e+0 on value 1.27e+6
mathjs:                   avg: 1881.00ms   std: 9.00ms      p0: 1872.00ms    p5: 1872.00ms    p25: 1872.00ms   p50: 1890.00ms   p75: 1890.00ms   p95: 1890.00ms   p100: 1890.00ms 
ndarray-same-out:         avg: 117.20ms    std: 1.21ms      p0: 116.00ms     p5: 116.00ms     p25: 116.00ms    p50: 117.00ms    p75: 118.00ms    p95: 121.00ms    p100: 121.00ms  
ndarray-diff-out:         avg: 117.30ms    std: 1.05ms      p0: 116.00ms     p5: 116.00ms     p25: 117.00ms    p50: 117.00ms    p75: 117.00ms    p95: 121.00ms    p100: 121.00ms  
matrix-js:                avg: 1245.00ms   std: 5.00ms      p0: 1240.00ms    p5: 1240.00ms    p25: 1240.00ms   p50: 1250.00ms   p75: 1250.00ms   p95: 1250.00ms   p100: 1250.00ms 
vanilla:                  avg: 441.25ms    std: 6.00ms      p0: 436.00ms     p5: 436.00ms     p25: 437.00ms    p50: 441.00ms    p75: 442.00ms    p95: 459.00ms    p100: 459.00ms  
vanilla (no loop):        avg: 572.15ms    std: 13.73ms     p0: 541.00ms     p5: 550.00ms     p25: 566.00ms    p50: 571.00ms    p75: 583.00ms    p95: 599.00ms    p100: 599.00ms  
vanilla (cache-line):     avg: 541.05ms    std: 10.63ms     p0: 533.00ms     p5: 534.00ms     p25: 535.00ms    p50: 538.00ms    p75: 542.00ms    p95: 573.00ms    p100: 573.00ms  
vanilla Float32Array:     avg: 256.20ms    std: 3.59ms      p0: 253.00ms     p5: 253.00ms     p25: 254.00ms    p50: 255.00ms    p75: 256.00ms    p95: 268.00ms    p100: 268.00ms    FAIL: greatest diff is 1.28e-2 on value 1.34e+6
vanilla typed array:      avg: 146.60ms    std: 22.48ms     p0: 138.00ms     p5: 138.00ms     p25: 139.00ms    p50: 139.00ms    p75: 140.00ms    p95: 216.00ms    p100: 216.00ms  
```

Results (JS, sorted)
```
tfjs-node-f32:            avg: 1.35ms      std: 1.11ms      p0: 1.00ms       p5: 1.00ms       p25: 1.00ms      p50: 1.00ms      p75: 1.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 1.06e+0 on value 1.27e+6
GPU.js single:            avg: 4.90ms      std: 0.77ms      p0: 4.00ms       p5: 4.00ms       p25: 4.00ms      p50: 5.00ms      p75: 5.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.18e+0 on value 1.37e+6
GPU.js unsigned:          avg: 5.15ms      std: 3.93ms      p0: 22.00ms      p5: 3.00ms       p25: 4.00ms      p50: 4.00ms      p75: 4.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 2.43e+0 on value 1.37e+6
GPU.js single:            avg: 6.30ms      std: 3.95ms      p0: 23.00ms      p5: 4.00ms       p25: 5.00ms      p50: 5.00ms      p75: 6.00ms      p95: 8.00ms      p100: 8.00ms      FAIL: greatest diff is 2.18e+0 on value 1.37e+6
ndarray-same-out:         avg: 117.20ms    std: 1.21ms      p0: 116.00ms     p5: 116.00ms     p25: 116.00ms    p50: 117.00ms    p75: 118.00ms    p95: 121.00ms    p100: 121.00ms  
ndarray-diff-out:         avg: 117.30ms    std: 1.05ms      p0: 116.00ms     p5: 116.00ms     p25: 117.00ms    p50: 117.00ms    p75: 117.00ms    p95: 121.00ms    p100: 121.00ms  
ml-matrix:                avg: 132.55ms    std: 7.16ms      p0: 129.00ms     p5: 129.00ms     p25: 129.00ms    p50: 130.00ms    p75: 131.00ms    p95: 156.00ms    p100: 156.00ms  
vanilla typed array:      avg: 146.60ms    std: 22.48ms     p0: 138.00ms     p5: 138.00ms     p25: 139.00ms    p50: 139.00ms    p75: 140.00ms    p95: 216.00ms    p100: 216.00ms  
vanilla Float32Array:     avg: 256.20ms    std: 3.59ms      p0: 253.00ms     p5: 253.00ms     p25: 254.00ms    p50: 255.00ms    p75: 256.00ms    p95: 268.00ms    p100: 268.00ms    FAIL: greatest diff is 1.28e-2 on value 1.34e+6
vanilla:                  avg: 441.25ms    std: 6.00ms      p0: 436.00ms     p5: 436.00ms     p25: 437.00ms    p50: 441.00ms    p75: 442.00ms    p95: 459.00ms    p100: 459.00ms  
default:                  avg: 470.10ms    std: 3.37ms      p0: 465.00ms     p5: 466.00ms     p25: 467.00ms    p50: 470.00ms    p75: 474.00ms    p95: 476.00ms    p100: 476.00ms  
vanilla (cache-line):     avg: 541.05ms    std: 10.63ms     p0: 533.00ms     p5: 534.00ms     p25: 535.00ms    p50: 538.00ms    p75: 542.00ms    p95: 573.00ms    p100: 573.00ms  
vanilla (no loop):        avg: 572.15ms    std: 13.73ms     p0: 541.00ms     p5: 550.00ms     p25: 566.00ms    p50: 571.00ms    p75: 583.00ms    p95: 599.00ms    p100: 599.00ms  
matrix-js:                avg: 1245.00ms   std: 5.00ms      p0: 1240.00ms    p5: 1240.00ms    p25: 1240.00ms   p50: 1250.00ms   p75: 1250.00ms   p95: 1250.00ms   p100: 1250.00ms 
mathjs:                   avg: 1881.00ms   std: 9.00ms      p0: 1872.00ms    p5: 1872.00ms    p25: 1872.00ms   p50: 1890.00ms   p75: 1890.00ms   p95: 1890.00ms   p100: 1890.00ms 
```

Specs:

```
Model: 2021 MacBook Pro 16-inch, Apple M1 Max, 32 GB RAM
OS: macOS Monterey 12.5
Node version: 16.17.0
```
