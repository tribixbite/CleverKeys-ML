#!/usr/bin/env node
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 8787;

const mime = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.txt': 'text/plain',
  '.wasm': 'application/wasm'
};

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);
  let filePath = path.join(root, urlPath);
  if (urlPath.endsWith('/')) filePath = path.join(filePath, 'index.html');
  if (!path.extname(filePath)) filePath += '.html';
  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      res.statusCode = 404;
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.setHeader('Content-Type', mime[ext] || 'application/octet-stream');
    const stream = fs.createReadStream(filePath);
    stream.on('error', () => { res.statusCode = 500; res.end('Server error'); });
    stream.pipe(res);
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Dev server at http://127.0.0.1:${port}/`);
});

