# vLLM 0.13 Upgrade Guide

**Branch:** `vllm-0.13-upgrade`
**Current Version:** `vllm>=0.9.2,<=0.10.0`
**Target Version:** `vllm>=0.13.0`

---

## Upgrade Path Summary

```
v0.10.0 → v0.11.0 → v0.12.0 → v0.13.0
```

Each version has breaking changes that must be addressed sequentially.

---

## Breaking Changes by Version

### v0.11.0 (Critical)

| Change | Impact | Action Required |
|--------|--------|-----------------|
| **V0 Engine Removal** | Complete removal of AsyncLLMEngine, LLMEngine, MQLLMEngine | Verify ART uses V1 engine exclusively |
| **V0 LoRA interface removed** | V0-only LoRA methods gone | Verify LoRA API usage |
| **CUDA Graph default changed** | `FULL_AND_PIECEWISE` now default | May affect performance characteristics |
| **CUDA 13 requirement** | Build requirement | Update Docker/CI environments |
| **PyTorch 2.8+ for CPU** | Dependency update | Update requirements |

### v0.12.0 (Moderate)

| Change | Impact | Action Required |
|--------|--------|-----------------|
| **PyTorch 2.9.0 + CUDA 12.9** | Major dependency bump | Update all environments |
| **`num_lookahead_slots` removed** | Parameter eliminated | Remove from any init calls |
| **`best_of` removed from SamplingParams** | Sampling API change | ⚠️ Check if ART uses this |
| **LoRA extra vocab support removed** | LoRA feature removal | Verify not using extra vocab |
| **xformers backend deprecated** | Attention backend | Migrate to flash-attn or other |
| **`seed=None` deprecated** | Must be explicit | Add explicit seeds |
| **`--fully-sharded-loras` flag added** | New LoRA option | Optional: evaluate for perf |

### v0.13.0 (Moderate)

| Change | Impact | Action Required |
|--------|--------|-----------------|
| **`VLLM_ATTENTION_BACKEND` env var removed** | Must use `--attention-backend` CLI | Update all launch scripts |
| **PassConfig flags renamed** | CLI argument names changed | Update any CLI usage |
| **Removed `-O.xx` flag** | Optimization flag gone | Remove from scripts |
| **Removed tokenizer setter** | API change | Use init-time config |
| **`embed_input_ids`/`embed_multimodal` removed** | Fallback methods gone | Use new multimodal API |

---

## ART Codebase Impact Analysis

### Files That Import vLLM Directly

Based on codebase analysis, these files need review:

```
ART/
├── art/
│   ├── dev/
│   │   ├── backend.py          # vLLM backend integration
│   │   ├── engine_args.py      # EngineArgs with max_lora_rank
│   │   └── ...
│   └── ...
└── pyproject.toml              # Version constraint
```

### Key vLLM APIs Used in ART

| API | Usage | Risk |
|-----|-------|------|
| `LLM` class | Model loading | Low - stable API |
| `SamplingParams` | Generation params | **Medium** - `best_of` removed |
| `LoRARequest` | LoRA adapter loading | Low - path unchanged |
| `enable_lora` param | LoRA support | Low - still supported |
| `max_lora_rank` param | LoRA rank config | Low - constraints unchanged |
| `trust_remote_code` | Qwen3 support | Low - still supported |

### Critical Checks - AUDIT RESULTS

| Check | Result | Location | Action |
|-------|--------|----------|--------|
| `best_of` parameter | ❌ Not used | - | No action |
| `num_lookahead_slots` | ⚠️ **USED** | `src/art/dev/engine.py:80` | Must remove/migrate |
| V0 engine methods | ⚠️ **USED** | `src/art/unsloth/state.py:19,53` | **CRITICAL: V0 removed in 0.11** |
| `VLLM_ATTENTION_BACKEND` env | ❌ Not used | - | No action |
| xformers backend | ⚠️ In dependencies | `uv.lock`, `pyproject.toml` | Remove from deps |
| LoRA extra vocab | ⚠️ **USED** | `src/art/dev/engine.py:71` | Must remove |
| `seed=None` | ⚠️ **USED** | `src/art/torchtune/config.py:91` | Add explicit default |

### Critical Issues Found

**1. V0 Engine Dependency (BLOCKER)**

`src/art/unsloth/state.py:52-53`:
```python
# Sticking with V0 engine for now
os.environ["VLLM_USE_V1"] = "0"
```

This explicitly forces V0 engine which is **completely removed in v0.11**. The Unsloth integration path must be migrated to V1.

Also imports V0-specific classes:
```python
from vllm.engine.async_llm_engine import AsyncLLMEngine  # V0
```

**2. num_lookahead_slots (REMOVED in v0.12)**

`src/art/dev/engine.py:80`:
```python
num_lookahead_slots: int
```

This parameter no longer exists in vLLM 0.12+.

**3. lora_extra_vocab_size (REMOVED in v0.12)**

`src/art/dev/engine.py:71`:
```python
lora_extra_vocab_size: int
```

This feature was removed in v0.12.

**4. Dual Engine Usage**

The codebase uses BOTH V0 and V1 engines:
- `src/art/unsloth/state.py` → Forces V0 (`VLLM_USE_V1=0`)
- `src/art/vllm/engine.py` → Forces V1 (`envs.VLLM_USE_V1 = True`)
- `src/art/torchtune/service.py` → Uses V1 (`vllm.v1.engine.async_llm`)

This dual-engine approach will fail since V0 is removed.

---

## Upgrade Steps

### Phase 1: Audit (This Document)

- [x] Document breaking changes v0.11-v0.13
- [x] Audit ART codebase for affected APIs
- [x] Identify all vLLM usage patterns
- [x] Create test plan
- [x] **Validate vLLM 0.13 on Blackwell hardware (msd6079.mjhst.com)**

### Phase 2: Code Changes (Priority Order)

**P0 - Blockers:**
- [x] Migrate `src/art/unsloth/state.py` from V0 to V1 engine
  - Removed `os.environ["VLLM_USE_V1"] = "0"`
  - Replaced `AsyncLLMEngine` with `AsyncLLM` (V1)
  - Updated all V0-specific imports
  - Simplified vLLMState.train_mode() for V1 semantics
- [x] Remove `num_lookahead_slots` from `src/art/dev/engine.py`
- [x] Remove `lora_extra_vocab_size` from `src/art/dev/engine.py`

**P1 - Required:**
- [x] Update `pyproject.toml` version constraint to `vllm>=0.13.0`
- [x] Update `pyproject.toml` torch constraint to `torch>=2.9.0`
- [x] Add explicit default for `seed` in `src/art/torchtune/config.py`
- [x] Update Unsloth to `>=2025.12.9` (supports vLLM 0.12+)
- [x] Update unsloth-zoo to `>=2025.12.7`
- [x] Update transformers to `>=4.57.1` (required by Unsloth 2025.12+)
- [x] xformers is transitive dep, not directly controlled (vLLM 0.13 uses flash-attn/flashinfer)

**P2 - Cleanup:**
- [x] Unify engine usage (everything on V1)
- [x] Deprecated `create_engine_pause_and_resume_functions` (V0 specific)
- [ ] Update any CLI scripts for renamed PassConfig flags
- [x] Updated CUDA/PyTorch requirements in pyproject.toml

### Phase 3: Testing

- [x] Test basic model loading/inference with vLLM 0.13 (Qwen3-4B ✓)
- [ ] Run existing tests with vLLM 0.13
- [ ] Test LoRA loading/inference (both Unsloth and Torchtune paths)
- [ ] Test multi-GPU (tensor_parallel)
- [ ] Test all supported models (Qwen3-4B ✓, Qwen3-8B)
- [ ] Benchmark performance comparison vs v0.10

### Phase 4: Documentation

- [ ] Update README with new requirements
- [ ] Update Docker images (CUDA 12.9, PyTorch 2.9)
- [ ] Document V1 engine migration

---

## Compatibility Matrix

| Component | v0.10 | v0.13 | Notes |
|-----------|-------|-------|-------|
| Python | 3.9+ | 3.9+ | No change |
| PyTorch | 2.x | 2.9.0 | **Upgrade required** |
| CUDA | 11.8+ | 12.9 | **Upgrade required** |
| LoRA API | ✓ | ✓ | Stable |
| `max_lora_rank` | ✓ | ✓ | Same constraints |
| `SamplingParams` | ✓ | Modified | `best_of` removed |
| V0 Engine | ✓ | ✗ | **Removed** |

---

## New Features in v0.13 (Optional Adoption)

| Feature | Description | Benefit |
|---------|-------------|---------|
| Anthropic API | `/v1/messages` endpoint | Anthropic client compat |
| Whisper speedup | ~3x faster than v0.12 | Audio model perf |
| Blackwell support | SM103 (GB300), SM120 (RTX PRO 6000) | Future hardware |
| Async scheduling fixes | Correctness improvements | Reliability |
| Binary embeddings | `encoding_format=bytes_only` | Embedding efficiency |

---

## Blackwell SM120 Validation (2025-12-28)

Successfully validated vLLM 0.13.0 on NVIDIA RTX PRO 6000 Blackwell (SM120, 98GB VRAM).

### Test Configuration

| Component | Version |
|-----------|---------|
| vLLM | 0.13.0 |
| PyTorch | 2.9.0+cu128 |
| CUDA Toolkit | 12.8 |
| Driver | 570.195.03 |
| GPU | RTX PRO 6000 Blackwell Max-Q (98GB) |

### Key Requirements for Blackwell

1. **CUDA Toolkit Required**: FlashInfer backend needs nvcc for JIT compilation
   ```bash
   sudo apt-get install cuda-toolkit-12-8
   export CUDA_HOME=/usr/local/cuda-12.8
   ```

2. **Use FLASHINFER Backend**: Default flash-attn backend fails on SM120
   ```bash
   # Deprecated in v0.14 - use --attention-config.backend instead
   export VLLM_ATTENTION_BACKEND=FLASHINFER
   ```

3. **Fix HuggingFace Cache Permissions**: If previously run as root
   ```bash
   sudo chown -R $USER:$USER ~/.cache/huggingface/
   ```

### Performance Results

```
Model: Qwen/Qwen3-4B
Memory: 7.5 GiB model, 72 GiB KV cache available
Speed: ~125 tokens/sec output
Max concurrency: 256x for 2048 token requests
```

### Verified Working

- [x] vLLM 0.13.0 installation via pip
- [x] V1 engine (only option, V0 removed)
- [x] FLASHINFER attention backend on SM120
- [x] Qwen3-4B model loading and inference
- [x] CUDA graph capture (FULL_AND_PIECEWISE mode)

---

## Rollback Plan

If upgrade fails:

```bash
# Revert to v0.10
git checkout main
pip install "vllm>=0.9.2,<=0.10.0"
```

---

## References

- [vLLM v0.13.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.13.0)
- [vLLM v0.12.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.12.0)
- [vLLM v0.11.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
