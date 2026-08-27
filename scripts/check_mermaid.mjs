#!/usr/bin/env node
/**
 * 校验 Markdown 里所有 ```mermaid 代码块能否正常渲染。
 *
 * 依赖（一次性）：
 *   npm i -g @mermaid-js/mermaid-cli
 * 会自动复用本机已安装的 Chrome / Edge，无需再下载一份 Chromium。
 *
 * 用法：
 *   node scripts/check_mermaid.mjs docs/week01/*.md README.md
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

const MMDC_CANDIDATES = [
  "/tmp/mmval/node_modules/.bin/mmdc",
  path.join(process.env.HOME ?? "", ".npm-global/bin/mmdc"),
  "/opt/homebrew/bin/mmdc",
  "/usr/local/bin/mmdc",
];

function resolveMmdc() {
  for (const p of MMDC_CANDIDATES) if (fs.existsSync(p)) return p;
  try {
    return execFileSync("which", ["mmdc"], { encoding: "utf8" }).trim();
  } catch {
    console.error("找不到 mmdc，请先执行: npm i -g @mermaid-js/mermaid-cli");
    process.exit(2);
  }
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error("usage: node check_mermaid.mjs <files...>");
  process.exit(2);
}

const mmdc = resolveMmdc();
const chrome = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
const env = { ...process.env };
if (chrome) env.PUPPETEER_EXECUTABLE_PATH = chrome;

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mermaid-check-"));
let total = 0;
let failed = 0;

for (const file of files) {
  const src = fs.readFileSync(file, "utf8");
  const blocks = [...src.matchAll(/```mermaid\n([\s\S]*?)```/g)].map((m) => m[1]);
  blocks.forEach((code, i) => {
    total += 1;
    const inFile = path.join(tmp, `b${total}.mmd`);
    fs.writeFileSync(inFile, code);
    try {
      execFileSync(mmdc, ["-i", inFile, "-o", path.join(tmp, `b${total}.svg`), "-q"], {
        stdio: ["ignore", "ignore", "pipe"],
        env,
      });
      console.log(`  ok   ${path.basename(file)} block #${i + 1}`);
    } catch (e) {
      failed += 1;
      console.log(`  FAIL ${path.basename(file)} block #${i + 1}`);
      console.log(String(e.stderr).split("\n").slice(0, 10).join("\n"));
    }
  });
}

console.log(`\n${total - failed}/${total} mermaid blocks OK`);
process.exit(failed ? 1 : 0);
