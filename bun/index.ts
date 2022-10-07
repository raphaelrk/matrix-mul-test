import * as sm from '@shumai/shumai/shumai';

const sz = 512;
const A = sm.randn([sz, sz]);
const B = sm.randn([sz, sz]);

const A_F32 = A.toFloat32Array();
const B_F32 = B.toFloat32Array();

// const start = Date.now();
// for (let i = 0; i < 5000; i++) {
//   const C = A.matmul(B);
// }
// const end = Date.now();
// console.log("shumai:  total average: ", (end - start) / 5000);

const AB = new Array(sz).fill(0).map(() => new Array(sz).fill(0));
for (let i = 0; i < sz; i++) {
  for (let j = 0; j < sz; j++) {
    for (let k = 0; k < sz; k++) {
      AB[i][j] += A_F32[i * sz + k] * B_F32[k * sz + j];
    }
  }
}

async function runTest<T>(name: string, f: () => T, getElem: (res: T, i: number, j: number) => number, { N_iters=20, checkRes=true, transformRes } : { N_iters?: number, checkRes?: boolean, transformRes?: (res: T) => Promise<T> } = {}): void {
  let res = await timeIt(name, () => f(), N_iters);
  if (transformRes) res = await transformRes(res);
  if (checkRes) check((i, j) => getElem(res, i, j));
  else console.log('\n');
}

async function timeIt<T>(name: string, f: () => T, N_iters=20): Promise<T> {
  let times = [];
  let res = null;
  let start = performance.now();
  for (let i = 0; i < N_iters; i++) {
    let time = Date.now();
    res = f();
    time = Date.now() - time;
    times.push(time);
  }
  times.sort();
  let end = performance.now();
  let totalAvg = (end - start) / N_iters;
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
    `totalAvg: ${totalAvg.toFixed(2)}ms`.padEnd(16),
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
  console.log(str);
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
    console.log(`  FAIL: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
  }
  else if (greatestDiff > 1e-6) {
    console.log(`  WARN: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
  }
  return [greatestDiff, greatestDiffVal];
}

// @ts-ignore
runTest('shumai', () => A.matmul(B), (res, i, j) => res[i*sz + j], { N_iters: 5000, transformRes: async (res) => await res.toFloat32Array() });


/** copying from node tests **/
/** these don't seem to work yet in bun but haven't dug too deep **/

// matrix-mul-test/js/index.js
// compare performance of matrix multiplication of various libraries
// import { GPU } from 'gpu.js';
// import * as tf from '@tensorflow/tfjs-node'; // for node
// import '@tensorflow/tfjs-backend-wasm'; // for either browser or node
// {
//   const sz = 512;
//   const A = new Array(sz).fill(0).map(() => new Array(sz).fill(0).map(() => 100 * Math.random()));
//   const B = new Array(sz).fill(0).map(() => new Array(sz).fill(0).map(() => 100 * Math.random()));
//   const A_flat = new Array(sz * sz).map((_, i) => A[Math.floor(i / sz)][i % sz]);
//   const B_flat = new Array(sz * sz).map((_, i) => B[Math.floor(i / sz)][i % sz]);
//   const A_typed = new Array(sz).fill(0).map((_, i) => new Float32Array(sz).fill(0).map((_, j) => A[i][j]));
//   const B_typed = new Array(sz).fill(0).map((_, i) => new Float32Array(sz).fill(0).map((_, j) => B[i][j]));
//   const A_flat_typed = new Float32Array(sz * sz).map((_, i) => A_flat[i]);
//   const B_flat_typed = new Float32Array(sz * sz).map((_, i) => B_flat[i]);
//
//   const AB = new Array(sz).fill(0).map(() => new Array(sz).fill(0));
//   for (let i = 0; i < sz; i++) {
//     for (let j = 0; j < sz; j++) {
//       for (let k = 0; k < sz; k++) {
//         AB[i][j] += A[i][k] * B[k][j];
//       }
//     }
//   }
//
//   function runTest<T>(name: string, f: () => T, getElem: (res: T, i: number, j: number) => number, { N_iters=20, checkRes=true, transformRes } : { N_iters?: number, checkRes?: boolean, transformRes?: (res: T) => T } = {}): void {
//     let res = timeIt(name, () => f(), N_iters);
//     if (transformRes) res = transformRes(res);
//     if (checkRes) check((i, j) => getElem(res, i, j));
//     else console.log('\n');
//   }
//
//   function timeIt<T>(name: string, f: () => T, N_iters=20): T {
//     let times = [];
//     let res = null;
//     for (let i = 0; i < N_iters; i++) {
//       let time = Date.now();
//       res = f();
//       time = Date.now() - time;
//       times.push(time);
//     }
//     times.sort();
//     let avg = times.reduce((a, b) => a + b, 0) / times.length;
//     let var_ = times.map(t => (t - avg) ** 2).reduce((a, b) => a + b, 0) / times.length;
//     let std = Math.sqrt(var_);
//     let p0 = times[0];
//     let p5 = times[Math.floor(times.length / 20)];
//     let p25 = times[Math.floor(times.length / 4)];
//     let p50 = times[Math.floor(times.length / 2)];
//     let p75 = times[Math.floor(times.length * 3 / 4)];
//     let p95 = times[Math.floor(times.length * 19 / 20)];
//     let p100 = times[times.length - 1];
//
//     let str = [
//       `${name}:`.padEnd(25),
//       `avg: ${avg.toFixed(2)}ms`.padEnd(16),
//       `std: ${std.toFixed(2)}ms`.padEnd(16),
//       `p0: ${p0.toFixed(2)}ms`.padEnd(16),
//       `p5: ${p5.toFixed(2)}ms`.padEnd(16),
//       `p25: ${p25.toFixed(2)}ms`.padEnd(16),
//       `p50: ${p50.toFixed(2)}ms`.padEnd(16),
//       `p75: ${p75.toFixed(2)}ms`.padEnd(16),
//       `p95: ${p95.toFixed(2)}ms`.padEnd(16),
//       `p100: ${p100.toFixed(2)}ms`.padEnd(16),
//     ].join(' ');
//     console.log(str);
//     return res!;
//   }
//
//   function check(getElem: (i: number, j: number) => number) : [number, number] {
//     let greatestDiff = 0;
//     let greatestDiffVal = 0;
//     for (let i = 0; i < sz; i++) {
//       for (let j = 0; j < sz; j++) {
//         let diff = Math.abs(AB[i][j] - getElem(i, j));
//         if (diff > greatestDiff) {
//           greatestDiff = diff;
//           greatestDiffVal = AB[i][j];
//         }
//       }
//     }
//     if (greatestDiff > 1e-3) {
//       console.log(`  FAIL: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
//     }
//     else if (greatestDiff > 1e-6) {
//       console.log(`  WARN: greatest diff is ${greatestDiff.toExponential(2)} on value ${greatestDiffVal.toExponential(2)}`);
//     }
//     console.log('\n');
//     return [greatestDiff, greatestDiffVal];
//   }
//
//   // GPU.js
//   (function GPUJSTest() {
//     const gpu = new GPU();
//     const gpuMatrixMultiplySinglePrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
//       let sum = 0;
//       for (let i = 0; i < 512; i++) sum += a[this.thread.y][i] * b[i][this.thread.x];
//       return sum;
//     }, { precision: 'single' }).setOutput([512, 512]);
//     const gpuMatrixMultiplyUnsignedPrecision = gpu.createKernel(function (a: number[][], b: number[][]) {
//       let sum = 0;
//       for (let i = 0; i < 512; i++) sum += a[this.thread.y][i] * b[i][this.thread.x];
//       return sum;
//     }, { precision: 'unsigned' }).setOutput([512, 512]);
//
//     runTest('GPU.js single', () => gpuMatrixMultiplySinglePrecision(A, B) as number[][], (res, i, j) => res[i][j]);
//     runTest('GPU.js unsigned', () => gpuMatrixMultiplyUnsignedPrecision(A, B) as number[][], (res, i, j) => res[i][j]);
//     runTest('GPU.js single', () => gpuMatrixMultiplySinglePrecision(A, B) as number[][], (res, i, j) => res[i][j]);
//   })();
//
//   let backends = ['wasm', 'tensorflow'];
//   for (let backend of backends) {
//     await tf.setBackend(backend);
//     const tensors = [tf.tensor2d(A), tf.tensor2d(B)];
//     let transformRes = (res: any) : any => res.dataSync();
//     runTest(`tfjs ${backend}`, () => tensors[0].matMul(tensors[1]), (res, i, j) => res[i * sz + j], { transformRes });
//   }
// }
