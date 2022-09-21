// js-matrix-mul-test/index.js
// compare performance of matrix multiplication of various libraries

// const SZ = 512;

function timeIt(name, f, N_iters=10) {
  let start = Date.now();
  let min = Infinity;
  let max = 0;
  for (let i = 0; i < N_iters; i++) {
    let time = Date.now();
    f();
    time = Date.now() - time;
    if (time < min) min = time;
    if (time > max) max = time;
  }
  let avg = (Date.now() - start) / N_iters;
  const label = `${name}:`.padEnd(25);
  const minStr = `${min}ms`.padEnd(10);
  const maxStr = `${max}ms`.padEnd(10);
  const avgStr = `${avg}ms`;
  console.log(`${label} min: ${minStr} max: ${maxStr} avg: ${avgStr}`);
}

// GPU.js
import { GPU } from 'gpu.js';
(function GPUJSTest() {
  // setup
  const generateMatrices = () => {
    const matrices : any[] = [[], []];
    for (let y = 0; y < 512; y++){
      matrices[0].push([]);
      matrices[1].push([]);
      for (let x = 0; x < 512; x++){
        matrices[0][y].push(Math.random());
        matrices[1][y].push(Math.random());
      }
    }
    return matrices;
  };
  const gpu = new GPU();
  const gpuMatrixMultiply = gpu.createKernel(function (a, b) {
    let sum = 0;
    for (let i = 0; i < 512; i++) {
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
  }).setOutput([512, 512]);
  const matrices = generateMatrices();

  // loop
  timeIt('GPU.js', () => gpuMatrixMultiply(matrices[0], matrices[1]));
})();

// ml-matrix
import { Matrix } from 'ml-matrix';
(function mlMatrixTest() {
  // setup
  const matrices = [Matrix.rand(512, 512), Matrix.rand(512, 512)];

  // loop
  timeIt('ml-matrix', () => matrices[0].mmul(matrices[1]));
}());

// tfjs-node
import * as tf from '@tensorflow/tfjs-node';
(function tfjsNodeTest() {
  // setup
  const tensors = [tf.randomUniform([512, 512]), tf.randomUniform([512, 512])];

  // loop
  timeIt('tfjs-node', () => tensors[0].matMul(tensors[1]));
})();

// mathjs
import * as math from 'mathjs';
(function mathjsTest() {
  // setup
  const matrices = [math.random([512, 512]), math.random([512, 512])];

  // loop
  // only do 2 iterations because it's so slow
  timeIt('mathjs', () => math.multiply(matrices[0], matrices[1]), 2);
})();

// vanilla
(function vanillaTest() {
  // setup
  const matrices = [
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
  ];

  // loop
  timeIt('vanilla', () => {
    const result = Array(512).fill(0).map(_ => Array(512).fill(0));
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        let sum = 0;
        for (let i = 0; i < 512; i++) {
          sum += matrices[0][y][i] * matrices[1][i][x];
        }
        result[y][x] = sum;
      }
    }
  });
})();

// vanilla (reducer)
(function vanillaNoLoopTest() {
  // setup
  const matrices = [
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
  ];

  // loop
  timeIt('vanilla (no loop)', () => {
    const result = Array(512).fill(0).map(_ => Array(512).fill(0));
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        result[y][x] = matrices[0][y].reduce((sum, val, i) => sum + val * matrices[1][i][x], 0);
      }
    }
  });
})();

// vanilla (cache-line)
(function vanillaCacheLineTest() {
  // setup
  const matrices = [
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
    Array(512).fill(0).map(_ => Array(512).fill(0).map(_ => Math.random())),
  ];

  // loop
  timeIt('vanilla (cache-line)', () => {
    const result = Array(512).fill(0).map(_ => Array(512).fill(0));
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        let sum = 0;
        for (let i = 0; i < 512; i += 8) {
          sum += matrices[0][y][i] * matrices[1][i][x];
          sum += matrices[0][y][i + 1] * matrices[1][i + 1][x];
          sum += matrices[0][y][i + 2] * matrices[1][i + 2][x];
          sum += matrices[0][y][i + 3] * matrices[1][i + 3][x];
          sum += matrices[0][y][i + 4] * matrices[1][i + 4][x];
          sum += matrices[0][y][i + 5] * matrices[1][i + 5][x];
          sum += matrices[0][y][i + 6] * matrices[1][i + 6][x];
          sum += matrices[0][y][i + 7] * matrices[1][i + 7][x];
        }
        result[y][x] = sum;
      }
    }
  });
})();

// vanilla Float32Array
(function vanillaFloat32ArrayTest() {
  // setup
  const matrices = [
    Array(512).fill(0).map(_ => new Float32Array(512).map(_ => Math.random())),
    Array(512).fill(0).map(_ => new Float32Array(512).map(_ => Math.random())),
  ];

  // loop
  timeIt('vanilla Float32Array', () => {
    const result = Array(512).fill(0).map(_ => new Float32Array(512));
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        let sum = 0;
        for (let i = 0; i < 512; i++) {
          sum += matrices[0][y][i] * matrices[1][i][x];
        }
        result[y][x] = sum;
      }
    }
  });
})();



// webassembly
// import { MatrixMultiply } from 'matrix-multiply-wasm';

// @stdlib/

// // implementing neural networks from scratch in javascript
//
// // import { GPU, IKernelFunctionThis, Texture } from "gpu.js";
// // const gpu = new GPU({
// //   mode: "gpu",
// // });
// //
// // function kernelFunction(this: IKernelFunctionThis): [number, number] {
// //   return [1, 1];
// // }
// //
// // function subKernel(): [number, number] {
// //   return [1, 1];
// // }
// //
// // const kernelMap = gpu.createKernelMap<typeof kernelFunction>({
// //   test: subKernel,
// // }, kernelFunction)
// // .setOutput([1])
// // .setPipeline(true);
// //
// // const result = kernelMap();
// //
// // console.log((result.test as Texture).toArray() as [number, number][]);
//
// // function kernelFunction(this: IKernelFunctionThis): number {
// //   // return 1 + this.thread.x;
// //   return this.thread.x + this.thread.y + this.thread.z;
// // }
// // const kernelMap = gpu.createKernel<typeof kernelFunction>(kernelFunction).setOutput([3,3,3]);
// // const result = kernelMap();
// // console.log(result as number[][][]);
//
// // const matrixMultiply = gpu.createKernel(function (a, b) {
// //   let sum = 0;
// //   for (let i = 0; i < 3; i++) {
// //     sum += a[this.thread.y][i] * b[i][this.thread.x];
// //   }
// //   return sum;
// // }, {
// //   output: [3, 3],
// // });
//
// // var kernel = gpu.createKernel(function() {
// //   return this.thread.x;
// // }).setOutput([100]);
// // console.log(kernel());
// //
// // var kernel = gpu.createKernel(function() {
// //   const i = 1;
// //   const j = 0.89;
// //   return i + j;
// // }).setOutput([100]);
// // console.log(kernel());
// //
// // var kernel = gpu.createKernel(function() {
// //   return [0.08, 2];
// // }).setOutput([100]);
// // console.log(kernel());
//
// // const kernel = gpu.createKernel<[number], {}>(function(x) {
// //   return x;
// // }).setOutput([100]);
// // console.log(kernel(42));
