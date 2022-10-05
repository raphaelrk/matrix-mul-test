# matrix-mul-test

Testing python vs js (node) vs js (chrome) vs C implementations of multiplying two 512x512 matrices on an M1 Max. Very WIP, careful drawing conclusions from this.

Some things to note:
- The context of this is wanting to do deep learning in js but worrying all the libraries would be significantly slower than numpy -- but it turns out a number of them are quite faster.
- There's a sharp drop from libraries taking ~100ms to ~5ms in the benchmarks, presumably the difference between single-core and multicore/gpu usage.
- None of these use the neural engine which can apparently do ~5 teraflops
- I should rewrite the C code to output "ms per multiply" like everything else so its easier to compare against
- I haven't tried to improve the python much, I really didn't expect these js libraries to outperform numpy and numba + numpy. I should try comparing to more python libraries, particularly ones that use the GPU.

### Instructions

To run node tests:

- `cd node`
- `npm install`
- `npm run start`

To run browser tests:

- `cd js`
- `npm install`
- `npm run serve`
- open `https://localhost:8080/` in Chrome Canary/Dev with webgpu enabled (`chrome://flags/#enable-unsafe-webgpu`)
- press run, check console

To run python (pip -- numba and numpy):

- `cd py`
- `python -m venv venv`
- `source venv/bin/activate` (or `source venv/bin/activate.fish` if using fish)
- `pip install -r requirements.txt`
- `python3 main_pip.py`

To run python (conda -- numba, numpy, pytorch, tensorflow):

- `cd py`
- `mamba create --name matrix-mul-test python=3.10`
- `conda activate matrix-mul-test`
- `mamba install pytorch -c pytorch-nightly`
- `mamba install -c apple tensorflow-deps`
- `mamba install -c conda-forge numpy`
- `mamba install -c numba numba`
- `mamba install scipy`
- `pip install tensorflow-macos`
- `pip install tensorflow-metal`
- `mamba env export -n matrix-mul-test > ENV.yml`
- `python main_conda.py`

To run c:

- `cd c`
- `clang++ blas_test.cc -framework Accelerate -std=c++11 -O3 -o blas_test`
- `./blas_test 512 512 512 100 100`

### Results

Results (Node):

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

Results (Node, sorted)
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

Results (Browser)
```
GPU.js single:                        totalAvg: 8.22ms       avg: 8.15ms      std: 6.35ms      p0: 22.00ms      p5: 31.00ms      p25: 5.00ms      p50: 6.00ms      p75: 7.00ms      p95: 8.00ms      p100: 8.00ms         browser.ts:103   FAIL: greatest diff is 2.03e+0 on value 1.25e+6
browser.ts:86 GPU.js unsigned:        totalAvg: 7.40ms       avg: 7.40ms      std: 4.27ms      p0: 10.00ms      p5: 13.00ms      p25: 5.00ms      p50: 6.00ms      p75: 6.00ms      p95: 9.00ms      p100: 9.00ms         browser.ts:103   FAIL: greatest diff is 2.28e+0 on value 1.25e+6
browser.ts:86 GPU.js single:          totalAvg: 6.93ms       avg: 6.95ms      std: 5.33ms      p0: 30.00ms      p5: 5.00ms       p25: 5.00ms      p50: 6.00ms      p75: 6.00ms      p95: 7.00ms      p100: 7.00ms         browser.ts:103   FAIL: greatest diff is 2.03e+0 on value 1.25e+6
browser.ts:86 ml-matrix:              totalAvg: 128.25ms     avg: 128.25ms    std: 4.93ms      p0: 125.00ms     p5: 125.00ms     p25: 126.00ms    p50: 128.00ms    p75: 128.00ms    p95: 148.00ms    p100: 148.00ms  
browser.ts:86 tfjs cpu:               totalAvg: 244.65ms     avg: 244.65ms    std: 10.55ms     p0: 240.00ms     p5: 240.00ms     p25: 241.00ms    p50: 241.00ms    p75: 242.00ms    p95: 280.00ms    p100: 280.00ms       browser.ts:103   FAIL: greatest diff is 3.05e-1 on value 1.23e+6
browser.ts:86 tfjs webgl:             totalAvg: 0.76ms       avg: 0.75ms      std: 3.05ms      p0: 0.00ms       p5: 0.00ms       p25: 0.00ms      p50: 0.00ms      p75: 0.00ms      p95: 14.00ms     p100: 14.00ms        browser.ts:103   FAIL: greatest diff is 2.03e+0 on value 1.25e+6
browser.ts:86 tfjs wasm:              totalAvg: 2.51ms       avg: 2.50ms      std: 0.59ms      p0: 2.00ms       p5: 2.00ms       p25: 2.00ms      p50: 2.00ms      p75: 3.00ms      p95: 4.00ms      p100: 4.00ms         browser.ts:103   FAIL: greatest diff is 2.15e+0 on value 1.32e+6, threads: 5
browser.ts:86 tfjs webgpu:            totalAvg: 0.16ms       avg: 0.15ms      std: 0.48ms      p0: 0.00ms       p5: 0.00ms       p25: 0.00ms      p50: 0.00ms      p75: 0.00ms      p95: 2.00ms      p100: 2.00ms         browser.ts:103   FAIL: greatest diff is 2.03e+0 on value 1.25e+6
browser.ts:86 mathjs:                 totalAvg: 3085.92ms    avg: 3086.00ms   std: 383.00ms    p0: 2703.00ms    p5: 2703.00ms    p25: 2703.00ms   p50: 3469.00ms   p75: 3469.00ms   p95: 3469.00ms   p100: 3469.00ms 
browser.ts:86 ndarray-same-out:       totalAvg: 117.20ms     avg: 117.20ms    std: 1.81ms      p0: 116.00ms     p5: 116.00ms     p25: 116.00ms    p50: 117.00ms    p75: 118.00ms    p95: 124.00ms    p100: 124.00ms  
browser.ts:86 ndarray-diff-out:       totalAvg: 117.48ms     avg: 117.50ms    std: 0.87ms      p0: 116.00ms     p5: 116.00ms     p25: 117.00ms    p50: 117.00ms    p75: 118.00ms    p95: 119.00ms    p100: 119.00ms  
browser.ts:86 matrix-js:              totalAvg: 1208.26ms    avg: 1208.00ms   std: 7.00ms      p0: 1201.00ms    p5: 1201.00ms    p25: 1201.00ms   p50: 1215.00ms   p75: 1215.00ms   p95: 1215.00ms   p100: 1215.00ms 
browser.ts:86 vanilla:                totalAvg: 380.12ms     avg: 380.15ms    std: 86.63ms     p0: 350.00ms     p5: 350.00ms     p25: 350.00ms    p50: 351.00ms    p75: 352.00ms    p95: 656.00ms    p100: 656.00ms  
browser.ts:86 vanilla (no loop):      totalAvg: 469.40ms     avg: 469.40ms    std: 15.57ms     p0: 462.00ms     p5: 462.00ms     p25: 462.00ms    p50: 464.00ms    p75: 468.00ms    p95: 521.00ms    p100: 521.00ms  
browser.ts:86 vanilla (cache-line):   totalAvg: 385.11ms     avg: 385.10ms    std: 65.44ms     p0: 360.00ms     p5: 360.00ms     p25: 361.00ms    p50: 362.00ms    p75: 369.00ms    p95: 590.00ms    p100: 590.00ms  
browser.ts:86 vanilla Float32Array:   totalAvg: 291.79ms     avg: 291.80ms    std: 91.60ms     p0: 259.00ms     p5: 259.00ms     p25: 259.00ms    p50: 261.00ms    p75: 264.00ms    p95: 570.00ms    p100: 570.00ms       browser.ts:103   FAIL: greatest diff is 1.23e-2 on value 1.30e+6
browser.ts:86 vanilla typed array:    totalAvg: 154.96ms     avg: 154.95ms    std: 88.86ms     p0: 125.00ms     p5: 125.00ms     p25: 125.00ms    p50: 125.00ms    p75: 126.00ms    p95: 424.00ms    p100: 424.00ms  
```

Results (Python, Pip)
```
nil:                mean: 0.0ms     std: 0.0ms      p0: 0.0ms       p5: 0.0ms       p25: 0.0ms      p50: 0.0ms      p75: 0.0ms      p95: 0.0ms      p100: 0.0ms
numpy.matmul:       mean: 3.45ms    std: 0.69ms     p0: 2.62ms      p5: 2.63ms      p25: 2.93ms     p50: 3.3ms      p75: 3.71ms     p95: 4.59ms     p100: 5.27ms
numba three loop:   mean: 126.83ms  std: 0.69ms     p0: 125.87ms    p5: 125.96ms    p25: 126.42ms   p50: 126.68ms   p75: 127.07ms   p95: 128.45ms   p100: 128.46ms
numba two loop:     mean: 242.83ms  std: 1.6ms      p0: 240.16ms    p5: 240.41ms    p25: 242.08ms   p50: 242.75ms   p75: 243.39ms   p95: 245.83ms   p100: 246.51ms
numba one loop:     mean: 204.16ms  std: 3.93ms     p0: 196.99ms    p5: 197.24ms    p25: 202.28ms   p50: 204.54ms   p75: 206.41ms   p95: 209.61ms   p100: 211.88ms
numba no loop:      mean: 4.62ms    std: 3.98ms     p0: 2.36ms      p5: 2.37ms      p25: 2.47ms     p50: 2.93ms     p75: 3.74ms     p95: 11.32ms    p100: 17.98ms
```

Results (Python, Conda)
```
nil:                mean: 0.0ms     std: 0.0ms      p0: 0.0ms       p5: 0.0ms       p25: 0.0ms      p50: 0.0ms      p75: 0.0ms      p95: 0.0ms      p100: 0.0ms     
numpy.matmul:       mean: 3.77ms    std: 2.58ms     p0: 2.26ms      p5: 2.26ms      p25: 2.46ms     p50: 2.91ms     p75: 3.74ms     p95: 10.46ms    p100: 12.17ms   
numba three loop:   mean: 129.21ms  std: 2.13ms     p0: 126.05ms    p5: 126.22ms    p25: 127.32ms   p50: 129.44ms   p75: 129.96ms   p95: 133.77ms   p100: 134.13ms  
numba two loop:     mean: 244.9ms   std: 2.15ms     p0: 242.68ms    p5: 242.91ms    p25: 243.19ms   p50: 243.53ms   p75: 246.81ms   p95: 247.76ms   p100: 250.49ms  
numba one loop:     mean: 203.98ms  std: 3.45ms     p0: 198.24ms    p5: 198.76ms    p25: 202.11ms   p50: 203.4ms    p75: 206.93ms   p95: 209.34ms   p100: 209.98ms  
numba no loop:      mean: 2.77ms    std: 0.81ms     p0: 1.55ms      p5: 1.65ms      p25: 2.31ms     p50: 2.7ms      p75: 2.83ms     p95: 4.22ms     p100: 5.11ms    
pytorch.matmul:     mean: 0.96ms    std: 1.43ms     p0: 0.53ms      p5: 0.53ms      p25: 0.55ms     p50: 0.57ms     p75: 0.65ms     p95: 1.29ms     p100: 7.17ms    
pytorch @:          mean: 0.58ms    std: 0.07ms     p0: 0.47ms      p5: 0.48ms      p25: 0.54ms     p50: 0.56ms     p75: 0.61ms     p95: 0.68ms     p100: 0.8ms     
tf.matmul:          mean: 1.81ms    std: 0.76ms     p0: 1.43ms      p5: 1.47ms      p25: 1.52ms     p50: 1.64ms     p75: 1.76ms     p95: 2.05ms     p100: 5.08ms    
tf @:               mean: 1.61ms    std: 0.11ms     p0: 1.4ms       p5: 1.48ms      p25: 1.52ms     p50: 1.6ms      p75: 1.72ms     p95: 1.78ms     p100: 1.84ms    
tf.linalg.matmul:   mean: 1.55ms    std: 0.1ms      p0: 1.42ms      p5: 1.43ms      p25: 1.48ms     p50: 1.52ms     p75: 1.62ms     p95: 1.75ms     p100: 1.75ms    
```

Results (C)
```
./blas_test 512 512 512 300 300
about 2 tflops/s
```

### Specs:

```
Model: 2021 MacBook Pro 16-inch, Apple M1 Max, 32 GB RAM
OS: macOS Monterey 12.5
Node: 16.17.0
Python: 3.10.6
Chrome: 107.0.5304.10 (Official Build) dev (arm64)
```
