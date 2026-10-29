import { readdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
for (const dir of ["dist", "scripts", "tests"])
  for (const name of readdirSync(dir))
    if (/\.(mjs|js)$/.test(name)) {
      const result = spawnSync(
        process.execPath,
        ["--check", `${dir}/${name}`],
        { stdio: "inherit" },
      );
      if (result.status !== 0) process.exit(1);
    }
const html = readFileSync("dist/index.html", "utf8");
const app = readFileSync("dist/app.js", "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
if (new Set(ids).size !== ids.length) throw Error("Duplicate HTML IDs");
for (const [, id] of app.matchAll(/\$\(['"]([^'"]+)['"]\)/g)) {
  if (
    !ids.includes(id) &&
    !["finger", "fingerVal", "count", "cal"].includes(id)
  )
    throw Error(`Missing UI element: ${id}`);
}
console.log("Syntax and UI-reference checks passed.");

// Verify shipped assets as well as UI references.
import { existsSync } from "node:fs";
import { resolve } from "node:path";
for (const [, target] of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  if (!/^(?:https?:|data:|mailto:)/.test(target) && !existsSync(resolve("dist", target))) {
    throw Error(`Missing shipped asset: ${target}`);
  }
}
console.log("Shipped asset references passed.");
