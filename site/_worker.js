// Serve the byte ranges requested by native video players. Other requests use the static site.
const VIDEO = { 'content-type': 'video/mp4', 'accept-ranges': 'bytes', 'cache-control': 'public, max-age=3600' };
// the range a request asks for of n bytes: [first, last], null where it asks for none, false where it cannot be met
function rangeOf(header, n) {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m || (!m[1] && !m[2])) return false;
  let a = m[1] ? +m[1] : n - +m[2], b = m[1] && m[2] ? +m[2] : n - 1;
  a = Math.max(a, 0); b = Math.min(b, n - 1);
  return a > b ? false : [a, b];
}
export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (!url.pathname.endsWith('.mp4')) return env.ASSETS.fetch(req);
    const file = await env.ASSETS.fetch(new Request(url.origin + url.pathname));
    const range = req.headers.get('range');
    if (!file.ok || !range) { const h = new Headers(file.headers); h.set('accept-ranges', 'bytes'); return new Response(req.method === 'HEAD' ? null : file.body, { status: file.status, headers: h }); }
    const bytes = await file.arrayBuffer(), n = bytes.byteLength, r = rangeOf(range, n);
    if (!r) return new Response(null, { status: 416, headers: { 'content-range': `bytes */${n}` } });
    const headers = { ...VIDEO, 'content-range': `bytes ${r[0]}-${r[1]}/${n}`, 'content-length': String(r[1] - r[0] + 1) };
    return new Response(req.method === 'HEAD' ? null : bytes.slice(r[0], r[1] + 1), { status: 206, headers });
  },
};
