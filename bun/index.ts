import * as sm from '@shumai/shumai/shumai';

const A = sm.randn([512, 512]);
const B = sm.randn([512, 512]);

const start = Date.now();
for (let i = 0; i < 5000; i++) {
  const C = A.matmul(B);
}
const end = Date.now();

console.log("shumai:  total average: ", (end - start) / 5000);
