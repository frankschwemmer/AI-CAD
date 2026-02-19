import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const port = Number.parseInt(process.env.PORT ?? '4173', 10);

const mimeTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png']
]);

createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', `http://${req.headers.host}`);
    let filePath = url.pathname === '/' ? '/index.html' : url.pathname;

    if (filePath.includes('..')) {
      res.writeHead(400).end('Bad request');
      return;
    }

    const absPath = path.join(__dirname, filePath);
    const body = await readFile(absPath);
    const contentType = mimeTypes.get(path.extname(absPath)) ?? 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': contentType }).end(body);
  } catch {
    res.writeHead(404).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Viewer running on http://127.0.0.1:${port}`);
});
