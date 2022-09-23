# js-matrix-mul-test

To run:

- `npm install`
- `npm run start`


Results:

```
default:                  min: 462ms      max: 473ms      avg: 465ms     
GPU.js single:            min: 4ms        max: 21ms       avg: 5.22ms      WARN: greatest diff is 1.81e-4
GPU.js unsigned:          min: 3ms        max: 24ms       avg: 4.42ms      FAIL: greatest diff is 9.96e-1
GPU.js single:            min: 4ms        max: 7ms        avg: 4.62ms      WARN: greatest diff is 1.81e-4
ml-matrix:                min: 128ms      max: 155ms      avg: 130.44ms  
tfjs-node:                min: 0ms        max: 10ms       avg: 1.18ms      WARN: greatest diff is 9.32e-5
mathjs:                   min: 1851ms     max: 1902ms     avg: 1876.5ms  
ndarray-same-out:         min: 116ms      max: 122ms      avg: 117.66ms  
ndarray-diff-out:         min: 116ms      max: 122ms      avg: 116.98ms  
matrix-js:                min: 1231ms     max: 1244ms     avg: 1237.5ms  
vanilla:                  min: 419ms      max: 472ms      avg: 427ms     
vanilla (no loop):        min: 527ms      max: 576ms      avg: 546ms     
vanilla (cache-line):     min: 508ms      max: 631ms      avg: 526.52ms  
vanilla Float32Array:     min: 249ms      max: 268ms      avg: 252.82ms    WARN: greatest diff is 1.19e-6
vanilla typed array:      min: 137ms      max: 221ms      avg: 142.68ms  
```

Specs:

```
Model: 2021 MacBook Pro 16-inch, Apple M1 Max, 32 GB RAM
OS: macOS Monterey 12.5
Node version: 16.17.0
```
