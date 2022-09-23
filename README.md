# js-matrix-mul-test

```
GPU.js:                   min: 4ms        max: 21ms       avg: 6.4ms
ml-matrix:                min: 128ms      max: 155ms      avg: 134.5ms
tfjs-node:                min: 1ms        max: 7ms        avg: 1.7ms
mathjs:                   min: 1843ms     max: 1919ms     avg: 1881ms
ndarray:                  min: 123ms      max: 128ms      avg: 124.3ms
matrix-js:                min: 911ms      max: 955ms      avg: 933ms
vanilla (no loop):        min: 313ms      max: 330ms      avg: 318.3ms
vanilla (cache-line):     min: 297ms      max: 312ms      avg: 301.6ms
vanilla Float32Array:     min: 246ms      max: 376ms      avg: 273.4ms
vanilla typed array:      min: 138ms      max: 156ms      avg: 142ms
```

Model: 2021 MacBook Pro 16-inch, Apple M1 Max, 32 GB RAM
OS: macOS Monterey 12.5
Node version: 16.17.0
