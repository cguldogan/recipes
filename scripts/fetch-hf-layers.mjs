/**
 * Build-time: fetch the transformer layer count (num_hidden_layers) from each
 * recipe model's HuggingFace config.json, write to public/hf-layers.json so the
 * command builder can emit an exact VLLM_PP_LAYER_PARTITION for the combined
 * "All N cards" pipeline-parallel pool.
 *
 * Cached between builds (reads existing manifest, only fetches missing entries).
 */

import fs from "fs";
import path from "path";

const ROOT = process.cwd();
const MANIFEST = path.join(ROOT, "public", "hf-layers.json");

let cache = {};
if (fs.existsSync(MANIFEST)) {
  try { cache = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch {}
}

function findYamlFiles(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...findYamlFiles(full));
    else if (e.name.endsWith(".yaml") || e.name.endsWith(".yml")) out.push(full);
  }
  return out;
}

// num_hidden_layers lives at the top level for text models, or nested under
// text_config / llm_config / language_config for multimodal checkpoints.
function extractLayers(cfg) {
  if (!cfg || typeof cfg !== "object") return null;
  const keys = ["num_hidden_layers", "n_layers", "num_layers"];
  for (const k of keys) {
    if (typeof cfg[k] === "number" && cfg[k] > 0) return cfg[k];
  }
  for (const nest of ["text_config", "llm_config", "language_config", "thinker_config"]) {
    const n = extractLayers(cfg[nest]);
    if (n) return n;
  }
  return null;
}

async function fetchLayers(modelId) {
  try {
    const res = await fetch(`https://huggingface.co/${modelId}/resolve/main/config.json`, {
      headers: { "User-Agent": "vllm-recipes-build/1.0" },
    });
    if (!res.ok) return null;
    return extractLayers(await res.json());
  } catch {
    return null;
  }
}

const files = findYamlFiles(path.join(ROOT, "models"));
const hfIds = files.map((f) => {
  const rel = path.relative(path.join(ROOT, "models"), f);
  const parts = rel.split(path.sep);
  const repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
  return `${parts[0]}/${repo}`;
});

let fetched = 0, cached = 0, failed = 0;
for (const id of hfIds) {
  if (typeof cache[id] === "number") { cached++; continue; }
  const n = await fetchLayers(id);
  if (n) {
    cache[id] = n;
    fetched++;
  } else {
    failed++;
  }
  await new Promise((r) => setTimeout(r, 100));
}

fs.mkdirSync(path.dirname(MANIFEST), { recursive: true });
fs.writeFileSync(MANIFEST, JSON.stringify(cache, null, 2));

console.log(`✓ HF layers: ${fetched} fetched, ${cached} cached, ${failed} failed (${hfIds.length} total)`);
