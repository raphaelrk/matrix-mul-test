// js-matrix-mul-test/index.js
// compare performance of matrix multiplication of various libraries

const sz = 512;
const A = new Array(sz).fill(0).map(() => new Array(sz).fill(0).map(() => 100 * Math.random()));
const B = new Array(sz).fill(0).map(() => new Array(sz).fill(0).map(() => 100 * Math.random()));
const A_flat = new Array(sz * sz).map((_, i) => A[Math.floor(i / sz)][i % sz]);
const B_flat = new Array(sz * sz).map((_, i) => B[Math.floor(i / sz)][i % sz]);
const A_typed = new Array(sz).fill(0).map((_, i) => new Float32Array(sz).fill(0).map((_, j) => A[i][j]));
const B_typed = new Array(sz).fill(0).map((_, i) => new Float32Array(sz).fill(0).map((_, j) => B[i][j]));
const A_flat_typed = new Float32Array(sz * sz).map((_, i) => A_flat[i]);
const B_flat_typed = new Float32Array(sz * sz).map((_, i) => B_flat[i]);

const AB = timeIt('default', () => {
  const out = new Array(sz).fill(0).map(() => new Array(sz).fill(0));
  for (let i = 0; i < sz; i++) {
    for (let j = 0; j < sz; j++) {
      // row i, col j of AB = (row i of A).dot(col j of B)
      for (let k = 0; k < sz; k++) {
        out[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return out;
});
process.stdout.write('\n');

function timeIt<T>(name: string, f: () => T, N_iters=20): T {
  let times = [];
  let res = null;
  for (let i = 0; i < N_iters; i++) {
    let time = Date.now();
    res = f();
    time = Date.now() - time;
    times.push(time);
  }
  times.sort();
  let avg = times.reduce((a, b) => a + b, 0) / times.length;
  let var_ = times.map(t => (t - avg) ** 2).reduce((a, b) => a + b, 0) / times.length;
  let std = Math.sqrt(var_);
  let p0 = times[0];
  let p5 = times[Math.floor(times.length / 20)];
  let p25 = times[Math.floor(times.length / 4)];
  let p50 = times[Math.floor(times.length / 2)];
  let p75 = times[Math.floor(times.length * 3 / 4)];
  let p95 = times[Math.floor(times.length * 19 / 20)];
  let p100 = times[times.length - 1];

  let str = [
    `${name}:`.padEnd(25),
    `avg: ${avg.toFixed(2)}ms`.padEnd(16),
    `std: ${std.toFixed(2)}ms`.padEnd(16),
    `p0: ${p0.toFixed(2)}ms`.padEnd(16),
    `p5: ${p5.toFixed(2)}ms`.padEnd(16),
    `p25: ${p25.toFixed(2)}ms`.padEnd(16),
    `p50: ${p50.toFixed(2)}ms`.padEnd(16),
    `p75: ${p75.toFixed(2)}ms`.padEnd(16),
    `p95: ${p95.toFixed(2)}ms`.padEnd(16),
    `p100: ${p100.toFixed(2)}ms`.padEnd(16),
  ].join(' ');
  process.stdout.write(str);
  return res!;
}

function check(getElem: (i: number, j: number) => number) : boolean {
  let greatestDiff = 0;
  let greatestDiffVal = 0;
  for (let i = 0; i < sz; i++) {
    for (let j = 0; j < sz; j++) {
      let diff = Math.abs(AB[i][j] - getElem(i, j));
      if (diff > greatestDiff) {
        greatestDiff = diff;
        greatestDiffVal = AB[i][j];
      }
    }
  }
  if (greatestDiff > 1e-3) {
    process.stdout.write(`  FAIL: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
  }
  else if (greatestDiff > 1e-6) {
    process.stdout.write(`  WARN: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
  }
  process.stdout.write('\n');
  return true;
}

// GPU.js
import { GPU } from 'gpu.js';
(function GPUJSTest() {
  // setup
  const matrices = [A, B];
  const gpu = new GPU();
  const gpuMatrixMultiplySinglePrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
    let sum = 0;
    for (let i = 0; i < 512; i++) {
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
  }, {
    precision: 'single',
  }).setOutput([512, 512]);
  const gpuMatrixMultiplyUnsignedPrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
    let sum = 0;
    for (let i = 0; i < 512; i++) {
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
  }, {
    precision: 'unsigned',
  }).setOutput([512, 512]);

  // loop
  const res1 = timeIt('GPU.js single', () => gpuMatrixMultiplySinglePrecision(matrices[0], matrices[1])) as number[][];

  // check
  check((i, j) => res1[i][j]);

  // loop
  const res2 = timeIt('GPU.js unsigned', () => gpuMatrixMultiplyUnsignedPrecision(matrices[0], matrices[1])) as number[][];

  // check
  check((i, j) => res2[i][j]);


  // loop
  const res3 = timeIt('GPU.js single', () => gpuMatrixMultiplySinglePrecision(matrices[0], matrices[1])) as number[][];

  // check
  check((i, j) => res3[i][j]);
})();

// ml-matrix
import { Matrix } from 'ml-matrix';
(function mlMatrixTest() {
  // setup
  // const matrices = [Matrix.rand(512, 512), Matrix.rand(512, 512)];
  const matrices = [new Matrix(A), new Matrix(B)];

  // loop
  const res = timeIt('ml-matrix', () => matrices[0].mmul(matrices[1]));

  // check
  check((i, j) => res.get(i, j));
}());

// tfjs-node
import * as tf from '@tensorflow/tfjs-node';
(function tfjsNodeTest() {
  // setup
  // const tensors = [tf.randomUniform([512, 512]), tf.randomUniform([512, 512])];
  const tensors = [tf.tensor2d(A), tf.tensor2d(B)];

  // loop
  const res : tf.Tensor = timeIt('tfjs-node-f32', () => tensors[0].matMul(tensors[1]));

  // check
  let data = res.dataSync();
  check((i, j) => data[i * sz + j]);
})();

// mathjs
import * as math from 'mathjs';
(function mathjsTest() {
  // setup
  // const matrices = [math.random([512, 512]), math.random([512, 512])];
  const matrices = [math.matrix(A), math.matrix(B)];

  // loop
  // only do 2 iterations because it's so slow
  const res = timeIt('mathjs', () => math.multiply(matrices[0], matrices[1]), 2);

  // check
  check((i, j) => res.get([i, j]));
})();

// ndarray
// @ts-ignore
import zeros from 'zeros';
import ndarray from 'ndarray';
import cwise from 'cwise';
import ops from 'ndarray-ops';
(function ndarrayTest() {
  // setup
  // const matrices = [zeros([512, 512]), zeros([512, 512])];
  // ops.random(matrices[0]);
  // ops.random(matrices[1]);
  const matrices = [ndarray(A_flat, [sz, sz]), ndarray(B_flat, [sz, sz])];
  const multiply = cwise({
    args: ['array', 'array', 'array'],
    body: function (a, b, c) {
      c = 0;
      for (let i = 0; i < 512; i++) {
        c += a * b;
      }
    },
  });

  // loop
  const out = zeros([512, 512]);
  timeIt('ndarray-same-out', () => multiply(matrices[0], matrices[1], out));
  process.stdout.write('\n');
  timeIt('ndarray-diff-out', () => multiply(matrices[0], matrices[1], zeros([512, 512])));

  // check
  check((i, j) => out.get(i, j));
})();

// matrix-js
// @ts-ignore
import matrix from 'matrix-js';
(function matrixJsTest() {
  // setup
  const matrices = [matrix(A), matrix(B)];

  // loop
  // only do 2 iterations because it's so slow
  const res = timeIt('matrix-js', () => matrices[0].prod(matrices[1]), 2);

  // check
  check((i, j) => res[i][j]);
})();

// vanilla
(function vanillaTest() {
  // setup
  const matrices = [A, B];

  // loop
  const res = timeIt('vanilla', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += matrices[0][i][k] * matrices[1][k][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  });

  // check
  check((i, j) => res[i][j]);
})();

// vanilla (reducer)
(function vanillaNoLoopTest() {
  // setup
  const matrices = [A, B];

  // loop
  const res = timeIt('vanilla (no loop)', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        out[i][j] = matrices[0][i].reduce((sum, a_val, k) => sum + a_val * matrices[1][k][j], 0);
      }
    }
    return out;
  });

  // check
  check((i, j) => res[i][j]);
})();

// vanilla (cache-line)
(function vanillaCacheLineTest() {
  // setup
  const matrices = [A, B];

  // loop
  const res = timeIt('vanilla (cache-line)', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k += 8) {
          sum += matrices[0][i][k] * matrices[1][k][j];
          sum += matrices[0][i][k + 1] * matrices[1][k + 1][j];
          sum += matrices[0][i][k + 2] * matrices[1][k + 2][j];
          sum += matrices[0][i][k + 3] * matrices[1][k + 3][j];
          sum += matrices[0][i][k + 4] * matrices[1][k + 4][j];
          sum += matrices[0][i][k + 5] * matrices[1][k + 5][j];
          sum += matrices[0][i][k + 6] * matrices[1][k + 6][j];
          sum += matrices[0][i][k + 7] * matrices[1][k + 7][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  });

  // check
  check((i, j) => res[i][j]);
})();

// vanilla Float32Array
(function vanillaFloat32ArrayTest() {
  // setup
  const matrices = [A_typed, B_typed];

  // loop
  const res = timeIt('vanilla Float32Array', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += matrices[0][i][k] * matrices[1][k][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  });

  // check
  check((i, j) => res[i][j]);
})();

// vanilla typed flat array
(function vanillaTypedArrayTest() {
  const matrices = [A_flat_typed, B_flat_typed];

  // loop
  const res = timeIt('vanilla typed array', () => {
    const out = new Float32Array(sz * sz);
    for (let i = 0; i < sz; i++) {
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += matrices[0][i * sz + k] * matrices[1][k * sz + j];
        }
        out[i * sz + j] = sum;
      }
    }
    return out;
  });

  // check
  check((i, j) => res[i * sz + j]);
})();

// ndarray-gemm just straight up gave wrong answers

// blasjs
// truly aggravatingly confusing
// types don't match up with implementation
// have no idea how to use / if it's possible to use
