import fs from 'fs';
import http from 'http';
import https from 'https';

const port = 8080;

// check if key and cert are available
// if not, execute ./ssl/make_certs.sh to generate key and cert
import { execSync } from 'child_process';
const exist = fs.existsSync('./ssl/key.pem') && fs.existsSync('./ssl/cert.pem');
if (!exist) {
  console.log('key.pem and cert.pem not found, generating...');
  execSync('cd ssl && ./make_certs.sh');
}

const server = https.createServer({
  key: fs.readFileSync('./ssl/key.pem'),
  cert: fs.readFileSync('./ssl/cert.pem')
}, (req, res) => {
  if (req.url === '/') {
    res.writeHead(200, {
      'Content-Type': 'text/html',
      // for tfjs-backend-wasm support
      // https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(fs.readFileSync('index.html'));
  }
  else if (req.url === '/dist/browser.js') {
    res.writeHead(200, {
      'Content-Type': 'text/javascript',
      // for tfjs-backend-wasm support
      // https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'same-origin',
    });
    res.end(fs.readFileSync('dist/browser.js'));
  }
  else if (
    req.url === '/dist/tfjs-backend-wasm.wasm' ||
    req.url === '/dist/tfjs-backend-wasm-simd.wasm' ||
    req.url === '/dist/tfjs-backend-wasm-threaded-simd.wasm'
  ) {
    res.writeHead(200, {
      'Content-Type': 'application/wasm',
      // for tfjs-backend-wasm support
      // https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'same-origin',
    });
    res.end(fs.readFileSync(`node_modules/@tensorflow/tfjs-backend-wasm/dist/${req.url.slice(6)}`));
  }
  else if (req.url === '/dist/browser.js.map') {
    res.writeHead(200, { 'Content-Type': 'text/javascript' });
    res.end(fs.readFileSync('dist/browser.js.map'));
  }
  else {
    console.log("Invalid request: " + req.url);
    res.writeHead(404);
    res.end();
  }
});

server.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
