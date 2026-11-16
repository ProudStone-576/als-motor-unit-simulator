// Assemble the static site into _site/ (GitHub Pages, Vercel, or any web server).
//
//   node web/build_site.mjs && npx serve _site      (or: python -m http.server -d _site)
//
// web/index.html holds the page body (it is also published as a Claude artifact,
// which supplies its own document skeleton); this wraps it in a full HTML document.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ? process.argv[2] : join(here, "..", "_site");

const icon =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E" +
  "%3Ccircle cx='16' cy='16' r='14' fill='%233d36a8'/%3E%3Ccircle cx='11' cy='13' r='4' fill='%23d8c4a6'/%3E" +
  "%3Ccircle cx='20' cy='12' r='3' fill='%23d8c4a6'/%3E%3Ccircle cx='17' cy='21' r='4' fill='%23c2336b'/%3E%3C/svg%3E";

const head = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="icon" href="${icon}">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
`;

mkdirSync(out, { recursive: true });
const body = readFileSync(join(here, "index.html"), "utf8");
writeFileSync(join(out, "index.html"), `${head}${body}\n</body>\n</html>\n`);
for (const name of ["mndsim.js", "app.js", "trial-worker.js", "tokens.css", "base.css"]) copyFileSync(join(here, name), join(out, name));
writeFileSync(join(out, ".nojekyll"), "");
console.log(`site written to ${out}`);
