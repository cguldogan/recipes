// "My rig" — a user's local GPU workstation, shared between the Browse filter
// and the per-recipe "Your rig" fit panel. The rig is a map of catalog card id
// → quantity. Total VRAM (Σ count × per-card capacity) is the fit budget.
//
// `profiles` maps a GPU count to the taxonomy hardware-profile id that the
// command builder can actually emit a `vllm serve` command for. Counts without
// a mapped profile still get a fit verdict in the panel, just no "use" button.

export const CARD_CATALOG = [
  { id: "rtx5090", label: "RTX 5090", per_gpu_gb: 32, profiles: { 1: "rtx_5090", 2: "rtx_5090_2x" } },
  { id: "rtx4090", label: "RTX 4090", per_gpu_gb: 24, profiles: { 2: "rtx_4090_2x" } },
  { id: "rtx3090", label: "RTX 3090", per_gpu_gb: 24, profiles: {} },
  { id: "rtxpro6000", label: "RTX PRO 6000", per_gpu_gb: 96, profiles: { 1: "rtx_pro_6000", 2: "rtx_pro_6000_2x", 8: "rtx_pro_6000_8x" } },
  // Note: taxonomy's `rtx_pro_5000_4x` is the 72 GB Blackwell part, not this
  // 48 GB card — deliberately left unmapped.
  { id: "rtxpro5000", label: "RTX PRO 5000", per_gpu_gb: 48, profiles: {} },
  { id: "rtxpro4500", label: "RTX PRO 4500", per_gpu_gb: 32, profiles: {} },
];

export const CARD_BY_ID = Object.fromEntries(CARD_CATALOG.map((c) => [c.id, c]));
export const MAX_PER_CARD = 16; // sane upper bound on the quantity stepper
const STORAGE_KEY = "vllm-recipes:my-rig";

// Rig config <-> URL/storage string: `rtx5090:2,rtxpro6000:1` (count > 0 only,
// in catalog order) so the value stays stable and shareable.
export function parseRig(str) {
  const counts = {};
  for (const part of (str || "").split(",").filter(Boolean)) {
    const [id, n] = part.split(":");
    const c = parseInt(n, 10);
    if (CARD_BY_ID[id] && c > 0) counts[id] = Math.min(c, MAX_PER_CARD);
  }
  return counts;
}

export function encodeRig(counts) {
  return CARD_CATALOG
    .filter((c) => counts[c.id] > 0)
    .map((c) => `${c.id}:${counts[c.id]}`)
    .join(",");
}

export function rigVramOf(counts) {
  return CARD_CATALOG.reduce((sum, c) => sum + (counts[c.id] || 0) * c.per_gpu_gb, 0);
}

// "2× RTX 5090 + 1× RTX PRO 6000"
export function rigLabel(counts) {
  return CARD_CATALOG
    .filter((c) => counts[c.id] > 0)
    .map((c) => `${counts[c.id]}× ${c.label}`)
    .join(" + ");
}

export function isRigEmpty(counts) {
  return rigVramOf(counts) <= 0;
}

// localStorage so the rig set on Browse is remembered on every recipe page.
// SSR-safe (guards `window`); never throws on private-mode / quota errors.
export function loadRig() {
  if (typeof window === "undefined") return {};
  try {
    return parseRig(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return {};
  }
}

export function saveRig(counts) {
  if (typeof window === "undefined") return;
  try {
    const s = encodeRig(counts);
    if (s) window.localStorage.setItem(STORAGE_KEY, s);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

// Synthetic hardware-profile id for "use every card in the rig at once". The
// command builder injects a profile under this id (rigCombinedProfile) so the
// pool can render a command even though it isn't a real taxonomy entry.
export const RIG_COMBINED_ID = "__rig_combined__";

export function rigGpuCount(counts) {
  return CARD_CATALOG.reduce((s, c) => s + (counts[c.id] || 0), 0);
}
function rigCardTypes(counts) {
  return CARD_CATALOG.filter((c) => (counts[c.id] || 0) > 0).length;
}
// Smallest per-card VRAM among owned cards — the binding constraint for
// even-split pipeline parallelism (every stage holds ~model/N).
function rigMinPerCard(counts) {
  const owned = CARD_CATALOG.filter((c) => (counts[c.id] || 0) > 0);
  return owned.length ? Math.min(...owned.map((c) => c.per_gpu_gb)) : 0;
}

// Deployable pool options for a rig:
//  - per owned card type: a 1× pool and (when you own ≥2) an N× pool, each
//    tensor-parallel and mapped to a taxonomy profile id when one exists.
//  - when the rig mixes ≥2 card types: an "All N cards" pool that pools every
//    card via pipeline parallelism (the only way to combine uneven VRAM —
//    tensor-parallel needs identical cards). profileId = RIG_COMBINED_ID.
export function rigPools(counts) {
  const pools = [];
  for (const c of CARD_CATALOG) {
    const n = counts[c.id] || 0;
    if (n < 1) continue;
    pools.push({ key: `${c.id}-1`, label: c.label, gpus: 1, vramGb: c.per_gpu_gb, profileId: c.profiles[1] || null });
    if (n >= 2) {
      pools.push({ key: `${c.id}-${n}`, label: `${c.label} ×${n}`, gpus: n, vramGb: c.per_gpu_gb * n, profileId: c.profiles[n] || null });
    }
  }
  const totalGpus = rigGpuCount(counts);
  if (rigCardTypes(counts) >= 2 && totalGpus >= 2) {
    // "All N cards" via UNEVEN pipeline parallelism: with VLLM_PP_LAYER_PARTITION
    // weighting layers by each card's VRAM (e.g. 32:32:96), the full VRAM sum is
    // usable — most layers land on the biggest card. (Plain even-split PP would
    // cap at smallest×N and OOM, which is why the command ships the partition.)
    pools.push({
      key: "combined",
      label: `All ${totalGpus} cards`,
      gpus: totalGpus,
      vramGb: rigVramOf(counts),
      vramList: rigGpuVramList(counts),
      profileId: RIG_COMBINED_ID,
      pipeline: true,
    });
  }
  return pools;
}

// Per-GPU VRAM, one entry per physical card (e.g. 2×5090 + 6000 → [32,32,96]).
// The pipeline-partition weights are derived from this.
export function rigGpuVramList(counts) {
  const list = [];
  for (const c of CARD_CATALOG) {
    for (let i = 0; i < (counts[c.id] || 0); i++) list.push(c.per_gpu_gb);
  }
  return list;
}

// Largest model this rig can serve — the biggest deployable pool (single card,
// TP within a card type, or uneven-PP across all). Used as the Browse fit budget.
export function rigUsableVram(counts) {
  const pools = rigPools(counts);
  return pools.length ? Math.max(...pools.map((p) => p.vramGb)) : 0;
}

// The synthetic taxonomy profile the command builder registers for the combined
// pool. Uneven pipeline-parallel across every card (layers weighted by VRAM via
// VLLM_PP_LAYER_PARTITION), so the full VRAM sum is usable. `pp_vram` is the
// per-GPU VRAM list used to build the partition. null when fewer than 2 GPUs.
export function rigCombinedProfile(counts) {
  const totalGpus = rigGpuCount(counts);
  if (totalGpus < 2) return null;
  return {
    brand: "NVIDIA",
    generation: "blackwell",
    display_name: "Your rig",
    description: `${rigLabel(counts)} · uneven pipeline-parallel across all cards`,
    gpu_count: totalGpus,
    vram_gb: rigVramOf(counts),
    scalable: false,
    parallel_mode: "pipeline",
    pp_vram: rigGpuVramList(counts),
  };
}
