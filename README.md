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
GPU.js single:            avg: 6.00ms      std: 4.66ms      p0: 26.00ms      p5: 4.00ms       p25: 4.00ms      p50: 5.00ms      p75: 5.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.05e+0 on value 1.29e+6
GPU.js unsigned:          avg: 4.50ms      std: 2.25ms      p0: 14.00ms      p5: 3.00ms       p25: 4.00ms      p50: 4.00ms      p75: 4.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 2.20e+0 on value 1.32e+6
GPU.js single:            avg: 4.20ms      std: 0.51ms      p0: 3.00ms       p5: 4.00ms       p25: 4.00ms      p50: 4.00ms      p75: 5.00ms      p95: 5.00ms      p100: 5.00ms      FAIL: greatest diff is 2.05e+0 on value 1.29e+6
ml-matrix:                avg: 135.55ms    std: 8.13ms      p0: 131.00ms     p5: 131.00ms     p25: 132.00ms    p50: 132.00ms    p75: 134.00ms    p95: 157.00ms    p100: 157.00ms  
tfjs wasm:                avg: 6.25ms      std: 1.37ms      p0: 12.00ms      p5: 5.00ms       p25: 6.00ms      p50: 6.00ms      p75: 6.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.17e+0 on value 1.29e+6
tfjs tensorflow:          avg: 1.55ms      std: 1.96ms      p0: 1.00ms       p5: 1.00ms       p25: 1.00ms      p50: 1.00ms      p75: 1.00ms      p95: 2.00ms      p100: 2.00ms      FAIL: greatest diff is 1.06e+0 on value 1.28e+6
mathjs:                   avg: 1833.50ms   std: 42.50ms     p0: 1791.00ms    p5: 1791.00ms    p25: 1791.00ms   p50: 1876.00ms   p75: 1876.00ms   p95: 1876.00ms   p100: 1876.00ms 
ndarray-same-out:         avg: 118.45ms    std: 2.18ms      p0: 117.00ms     p5: 117.00ms     p25: 117.00ms    p50: 118.00ms    p75: 119.00ms    p95: 125.00ms    p100: 125.00ms  
ndarray-diff-out:         avg: 118.00ms    std: 2.00ms      p0: 116.00ms     p5: 116.00ms     p25: 117.00ms    p50: 118.00ms    p75: 118.00ms    p95: 125.00ms    p100: 125.00ms  
matrix-js:                avg: 1264.00ms   std: 13.00ms     p0: 1251.00ms    p5: 1251.00ms    p25: 1251.00ms   p50: 1277.00ms   p75: 1277.00ms   p95: 1277.00ms   p100: 1277.00ms 
vanilla:                  avg: 453.15ms    std: 8.71ms      p0: 440.00ms     p5: 442.00ms     p25: 449.00ms    p50: 451.00ms    p75: 458.00ms    p95: 473.00ms    p100: 473.00ms  
vanilla (no loop):        avg: 563.60ms    std: 31.42ms     p0: 542.00ms     p5: 542.00ms     p25: 545.00ms    p50: 553.00ms    p75: 576.00ms    p95: 678.00ms    p100: 678.00ms  
vanilla (cache-line):     avg: 454.70ms    std: 7.99ms      p0: 445.00ms     p5: 445.00ms     p25: 450.00ms    p50: 454.00ms    p75: 459.00ms    p95: 478.00ms    p100: 478.00ms  
vanilla Float32Array:     avg: 243.30ms    std: 3.76ms      p0: 241.00ms     p5: 241.00ms     p25: 241.00ms    p50: 242.00ms    p75: 243.00ms    p95: 256.00ms    p100: 256.00ms    FAIL: greatest diff is 1.29e-2 on value 1.34e+6
vanilla typed array:      avg: 145.40ms    std: 23.04ms     p0: 137.00ms     p5: 137.00ms     p25: 137.00ms    p50: 138.00ms    p75: 138.00ms    p95: 215.00ms    p100: 215.00ms  
```

Results (JS, sorted)
```
tfjs tensorflow:          avg: 1.55ms      std: 1.96ms      p0: 1.00ms       p5: 1.00ms       p25: 1.00ms      p50: 1.00ms      p75: 1.00ms      p95: 2.00ms      p100: 2.00ms      FAIL: greatest diff is 1.06e+0 on value 1.28e+6
GPU.js single:            avg: 4.20ms      std: 0.51ms      p0: 3.00ms       p5: 4.00ms       p25: 4.00ms      p50: 4.00ms      p75: 5.00ms      p95: 5.00ms      p100: 5.00ms      FAIL: greatest diff is 2.05e+0 on value 1.29e+6
GPU.js unsigned:          avg: 4.50ms      std: 2.25ms      p0: 14.00ms      p5: 3.00ms       p25: 4.00ms      p50: 4.00ms      p75: 4.00ms      p95: 6.00ms      p100: 6.00ms      FAIL: greatest diff is 2.20e+0 on value 1.32e+6
GPU.js single:            avg: 6.00ms      std: 4.66ms      p0: 26.00ms      p5: 4.00ms       p25: 4.00ms      p50: 5.00ms      p75: 5.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.05e+0 on value 1.29e+6
tfjs wasm:                avg: 6.25ms      std: 1.37ms      p0: 12.00ms      p5: 5.00ms       p25: 6.00ms      p50: 6.00ms      p75: 6.00ms      p95: 7.00ms      p100: 7.00ms      FAIL: greatest diff is 2.17e+0 on value 1.29e+6
ndarray-diff-out:         avg: 118.00ms    std: 2.00ms      p0: 116.00ms     p5: 116.00ms     p25: 117.00ms    p50: 118.00ms    p75: 118.00ms    p95: 125.00ms    p100: 125.00ms  
ndarray-same-out:         avg: 118.45ms    std: 2.18ms      p0: 117.00ms     p5: 117.00ms     p25: 117.00ms    p50: 118.00ms    p75: 119.00ms    p95: 125.00ms    p100: 125.00ms  
ml-matrix:                avg: 135.55ms    std: 8.13ms      p0: 131.00ms     p5: 131.00ms     p25: 132.00ms    p50: 132.00ms    p75: 134.00ms    p95: 157.00ms    p100: 157.00ms  
vanilla typed array:      avg: 145.40ms    std: 23.04ms     p0: 137.00ms     p5: 137.00ms     p25: 137.00ms    p50: 138.00ms    p75: 138.00ms    p95: 215.00ms    p100: 215.00ms  
vanilla Float32Array:     avg: 243.30ms    std: 3.76ms      p0: 241.00ms     p5: 241.00ms     p25: 241.00ms    p50: 242.00ms    p75: 243.00ms    p95: 256.00ms    p100: 256.00ms    FAIL: greatest diff is 1.29e-2 on value 1.34e+6
vanilla:                  avg: 453.15ms    std: 8.71ms      p0: 440.00ms     p5: 442.00ms     p25: 449.00ms    p50: 451.00ms    p75: 458.00ms    p95: 473.00ms    p100: 473.00ms  
vanilla (cache-line):     avg: 454.70ms    std: 7.99ms      p0: 445.00ms     p5: 445.00ms     p25: 450.00ms    p50: 454.00ms    p75: 459.00ms    p95: 478.00ms    p100: 478.00ms  
vanilla (no loop):        avg: 563.60ms    std: 31.42ms     p0: 542.00ms     p5: 542.00ms     p25: 545.00ms    p50: 553.00ms    p75: 576.00ms    p95: 678.00ms    p100: 678.00ms  
matrix-js:                avg: 1264.00ms   std: 13.00ms     p0: 1251.00ms    p5: 1251.00ms    p25: 1251.00ms   p50: 1277.00ms   p75: 1277.00ms   p95: 1277.00ms   p100: 1277.00ms 
mathjs:                   avg: 1833.50ms   std: 42.50ms     p0: 1791.00ms    p5: 1791.00ms    p25: 1791.00ms   p50: 1876.00ms   p75: 1876.00ms   p95: 1876.00ms   p100: 1876.00ms 
```

Specs:

```
Model: 2021 MacBook Pro 16-inch, Apple M1 Max, 32 GB RAM
OS: macOS Monterey 12.5
Node version: 16.17.0
```
