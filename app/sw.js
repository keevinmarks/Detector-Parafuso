// Guarda os arquivos do app no celular para ele abrir mesmo sem internet.
// Ao trocar o modelo, mude a versão abaixo para o celular buscar os arquivos novos.
const VERSAO = 'visao-2';
const ARQUIVOS = ['./', 'index.html', 'app.js', 'modelo.json', 'modelo.onnx', 'manifest.json', 'icone-192.png',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js'];
self.addEventListener('install', e => e.waitUntil(caches.open(VERSAO).then(c => c.addAll(ARQUIVOS)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSAO).map(k => caches.delete(k))))));
self.addEventListener('fetch', e => e.respondWith(caches.match(e.request).then(r => r || fetch(e.request).then(resp => {
  if (e.request.method === 'GET' && resp.ok && e.request.url.includes('onnxruntime-web')) { const cp = resp.clone(); caches.open(VERSAO).then(c => c.put(e.request, cp)); }
  return resp;
}))));
