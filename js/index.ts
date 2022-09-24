// matrix-mul-test/js/index.js
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

const AB = new Array(sz).fill(0).map(() => new Array(sz).fill(0));
for (let i = 0; i < sz; i++) {
  for (let j = 0; j < sz; j++) {
    for (let k = 0; k < sz; k++) {
      AB[i][j] += A[i][k] * B[k][j];
    }
  }
}

function runTest<T>(name: string, f: () => T, getElem: (res: T, i: number, j: number) => number, { N_iters=20, checkRes=true, transformRes } : { N_iters?: number, checkRes?: boolean, transformRes?: (res: T) => T } = {}): void {
  let res = timeIt(name, () => f(), N_iters);
  if (transformRes) res = transformRes(res);
  if (checkRes) check((i, j) => getElem(res, i, j));
  else process.stdout.write('\n');
}

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

function check(getElem: (i: number, j: number) => number) : [number, number] {
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
  return [greatestDiff, greatestDiffVal];
}

// GPU.js
import { GPU } from 'gpu.js';
(function GPUJSTest() {
  const gpu = new GPU();
  const gpuMatrixMultiplySinglePrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += a[this.thread.y][i] * b[i][this.thread.x];
    return sum;
  }, { precision: 'single' }).setOutput([512, 512]);
  const gpuMatrixMultiplyUnsignedPrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += a[this.thread.y][i] * b[i][this.thread.x];
    return sum;
  }, { precision: 'unsigned' }).setOutput([512, 512]);

  runTest('GPU.js single', () => gpuMatrixMultiplySinglePrecision(A, B) as number[][], (res, i, j) => res[i][j]);
  runTest('GPU.js unsigned', () => gpuMatrixMultiplyUnsignedPrecision(A, B) as number[][], (res, i, j) => res[i][j]);
  runTest('GPU.js single', () => gpuMatrixMultiplySinglePrecision(A, B) as number[][], (res, i, j) => res[i][j]);
})();

// ml-matrix
import { Matrix } from 'ml-matrix';
(function mlMatrixTest() {
  const matrices = [new Matrix(A), new Matrix(B)];
  runTest('ml-matrix', () => matrices[0].mmul(matrices[1]), (res, i, j) => res.get(i, j));
}());

// tfjs-node
import * as tfNode from '@tensorflow/tfjs-node';
(function tfjsNodeTest() {
  const tensors = [tfNode.tensor2d(A), tfNode.tensor2d(B)];
  let transformRes = (res: any) : any => res.dataSync();
  runTest('tfjs-node', () => tensors[0].matMul(tensors[1]), (res, i, j) => res[i * sz + j], { transformRes });
})();

// mathjs
import * as math from 'mathjs';
(function mathjsTest() {
  const matrices = [math.matrix(A), math.matrix(B)];
  runTest('mathjs', () => math.multiply(matrices[0], matrices[1]), (res, i, j) => res.get([i, j]), { N_iters: 2 }); // only do 2 iterations because it's slow
})();

// ndarray
// @ts-ignore
import zeros from 'zeros';
import ndarray from 'ndarray';
import cwise from 'cwise';
(function ndarrayTest() {
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
  const out = zeros([512, 512]);
  runTest('ndarray-same-out', () => multiply(matrices[0], matrices[1], out), (res, i, j) => out.get(i, j));
  runTest('ndarray-diff-out', () => multiply(matrices[0], matrices[1], zeros([512, 512])), (res, i, j) => out.get(i, j), { checkRes: false });
})();

// matrix-js
// @ts-ignore
import matrix from 'matrix-js';
(function matrixJsTest() {
  const matrices = [matrix(A), matrix(B)];
  runTest('matrix-js', () => matrices[0].prod(matrices[1]), (res, i, j) => res[i][j], { N_iters: 2 }); // only do 2 iterations because it's slow
})();

// vanilla
(function vanillaTest() {
  runTest('vanilla', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += A[i][k] * B[k][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  }, (res, i, j) => res[i][j]);
})();

// vanilla (reducer)
(function vanillaNoLoopTest() {
  runTest('vanilla (no loop)', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        out[i][j] = A[i].reduce((sum, a_val, k) => sum + a_val * B[k][j], 0);
      }
    }
    return out;
  }, (res, i, j) => res[i][j]);
})();

// vanilla (cache-line)
(function vanillaCacheLineTest() {
  runTest('vanilla (cache-line)', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k += 8) {
          sum += A[i][k]     * B[k][j];
          sum += A[i][k + 1] * B[k + 1][j];
          sum += A[i][k + 2] * B[k + 2][j];
          sum += A[i][k + 3] * B[k + 3][j];
          sum += A[i][k + 4] * B[k + 4][j];
          sum += A[i][k + 5] * B[k + 5][j];
          sum += A[i][k + 6] * B[k + 6][j];
          sum += A[i][k + 7] * B[k + 7][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  }, (res, i, j) => res[i][j]);
})();

// vanilla Float32Array
(function vanillaFloat32ArrayTest() {
  runTest('vanilla Float32Array', () => {
    const out = new Array(sz);
    for (let i = 0; i < sz; i++) {
      out[i] = new Array(sz);
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += A_typed[i][k] * B_typed[k][j];
        }
        out[i][j] = sum;
      }
    }
    return out;
  }, (res, i, j) => res[i][j]);
})();

// vanilla typed flat array
(function vanillaTypedArrayTest() {
  runTest('vanilla typed array', () => {
    const out = new Float32Array(sz * sz);
    for (let i = 0; i < sz; i++) {
      for (let j = 0; j < sz; j++) {
        let sum = 0;
        for (let k = 0; k < sz; k++) {
          sum += A_flat_typed[i * sz + k] * B_flat_typed[k * sz + j];
        }
        out[i * sz + j] = sum;
      }
    }
    return out;
  }, (res, i, j) => res[i * sz + j]);
})();

// ndarray-gemm gave incorrect result

// blasjs
// very confusing
// types don't match up with implementation
// have no idea how to use / if it's possible to use

// `npm install numjs` didn't work
