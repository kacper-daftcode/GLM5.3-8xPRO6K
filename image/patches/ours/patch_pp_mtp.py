"""Patch vLLM (obraz nvfp4-fi618): MTP/EAGLE draft z pipeline parallelism (plan A4: TP4 x PP2).

Runtime juz to wspiera: gpu_model_runner tworzy drafter WYLACZNIE na ostatnim stage'u PP ("we put the entire draft model
on the last PP rank"). Blokuje tylko check konfiguracji: SpeculativeConfig._verify_args ->
draft_model_config.verify_with_parallel_config(pp>1) -> "Supported models implement the SupportsPP interface"
(DeepSeekMTPModel nie jest PP-dzielony, bo nie musi). Tu: dla method in (mtp, eagle, eagle3) przy pp>1 sprawdzamy
tylko podzielnosc glow przez TP i pomijamy wymog SupportsPP. Idempotentny, twarde anchory.
"""
import pathlib

S = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py")


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


patch(
    S,
    """        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )
""",
    """        if self.draft_model_config:
            _pc = self.draft_parallel_config
            if _pc.pipeline_parallel_size > 1 and self.method in ("mtp", "eagle", "eagle3"):
                # TQ_PP: draft w calosci na ostatnim stage'u PP (gpu_model_runner: drafter tylko na
                # is_last_rank), wiec nie potrzebuje SupportsPP; zostaje check podzielnosci glow przez TP.
                # 2026-09-17: dziala ze --async-scheduling (broadcast stanu spekulacji ze stage'u last, embed
                # drafta z checkpointu, wyrownanie output_token_ids na PP0). Sciezka sync (batch queue) jest
                # blokowana w GPUModelRunner (drafty aplikowane poza kolejnoscia -> smieci).
                _heads = self.draft_model_config.model_arch_config.total_num_attention_heads
                if _heads % _pc.tensor_parallel_size != 0:
                    raise ValueError(
                        f"Total number of attention heads ({_heads}) must be divisible by "
                        f"tensor parallel size ({_pc.tensor_parallel_size})."
                    )
                logger.warning(
                    "TQ_PP: draft %r runs entirely on the last PP stage (pp=%d); "
                    "skipping the SupportsPP requirement for the draft model.",
                    self.method,
                    _pc.pipeline_parallel_size,
                )
            else:
                self.draft_model_config.verify_with_parallel_config(_pc)
""",
)

# ---- gpu_model_runner: na stage'ach != last nie ma self.drafter -> guard hasattr w miejscach wspolnych dla rankow
R = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")

# 00) PP>1 + spekulacja wymaga async scheduling (sciezka sync/batch-queue aplikuje drafty poza kolejnoscia)
patch(
    R,
    """        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )
""",
    """        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )
        if (  # TQ_PP
            self.num_spec_tokens > 0
            and not self.use_async_scheduling
            and self.parallel_config.pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "TQ_PP: speculative decoding with pipeline_parallel_size>1 requires async "
                "scheduling (--async-scheduling); the synchronous batch-queue path applies draft "
                "tokens out of order and corrupts outputs."
            )
""",
)

# 0) helper debug (VLLM_TQ_PP_DEBUG=1 -> log kolektywow PP na stderr)
patch(
    R,
    "logger = init_logger(__name__)\n",
    """logger = init_logger(__name__)


def _tq_pp_debug():
    \"\"\"TQ_PP: zwraca funkcje logujaca (stderr, z rankiem i licznikiem) albo None, gdy VLLM_TQ_PP_DEBUG != 1.\"\"\"
    import os as _os

    if _os.environ.get("VLLM_TQ_PP_DEBUG", "0") != "1":
        return None
    import sys as _sys
    import time as _time

    g = globals()
    g["_tq_pp_seq"] = g.get("_tq_pp_seq", 0) + 1
    seq = g["_tq_pp_seq"]
    try:
        r = get_pp_group().rank_in_group
    except Exception:
        r = -1

    def _log(msg):
        _sys.stderr.write(f"[TQ_PP pp_rank={r} #{seq} t={_time.time():.3f}] {msg}\\n")
        _sys.stderr.flush()

    return _log
""",
)

# a) budowa attn metadata (wszystkie ranki)
patch(
    R,
    """            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(
                    self.drafter,""",
    """            if (
                self.speculative_config
                and hasattr(self, "drafter")  # TQ_PP
                and spec_decode_common_attn_metadata is None
            ):
                if isinstance(
                    self.drafter,""",
)
patch(
    R,
    """            if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
                )
            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):""",
    """            if (
                self.speculative_config
                and hasattr(self, "drafter")  # TQ_PP
                and isinstance(self.drafter, Step3p5MTPProposer)
            ):
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
                )
            elif (
                self.speculative_config
                and hasattr(self, "drafter")  # TQ_PP
                and isinstance(self.drafter, Gemma4Proposer)
            ):""",
)
# b) _dummy_run
patch(
    R,
    """            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.""",
    """            if self.speculative_config and hasattr(self, "drafter") and (  # TQ_PP
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.""",
)
# c) initialize_attn_backend drafta
patch(
    R,
    """        # Initialize drafter attention backend
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):""",
    """        # Initialize drafter attention backend
        if self.speculative_config and hasattr(self, "drafter") and (  # TQ_PP
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):""",
)
# d) initialize_cudagraph_keys drafta
patch(
    R,
    """        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)""",
    """        if self.speculative_config and hasattr(self, "drafter") and (  # TQ_PP
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)""",
)
# e) validate_same_kv_cache_group (extract_hidden_states)
patch(
    R,
    """        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)""",
    """        if (
            self.speculative_config
            and hasattr(self, "drafter")  # TQ_PP
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)""",
)

# ---- PP + async scheduling + spec decode: stan spekulacji z ostatniego stage'u do pozostalych ------------------
# Upstream: ostatni stage broadcastuje sampled_token_ids [num_reqs,1] zaraz po samplerze (asercja shape[-1]==1).
# Ze spekulacja potrzebne sa (po kroku drafta): next_token_id (ostatni zaakceptowany), liczba zaakceptowanych
# (korekty num_computed_tokens na GPU/CPU) i drafty na kolejny krok. Pakujemy w int32 [num_reqs, 2+k].
# f) sample_tokens: przy spekulacji odloz broadcast do po drafterze
patch(
    R,
    """        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )
""",
    """        _pp_spec_broadcast_pending = False  # TQ_PP
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                if self.num_spec_tokens > 0:
                    # TQ_PP: ze spekulacja stan (next token, accepted, drafts) znany dopiero po drafterze
                    _pp_spec_broadcast_pending = True
                else:
                    self._pp_broadcast_prev_sampled_token_ids(
                        sampler_output.sampled_token_ids
                    )
""",
)
# g) po bloku spekulacji, przed bookkeepingiem
patch(
    R,
    """        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
""",
    """        if _pp_spec_broadcast_pending:  # TQ_PP
            self._pp_broadcast_spec_state()

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
""",
)
# h) odbior na stage'ach != last (spec-aware) + i) nowa metoda broadcastu
patch(
    R,
    """    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        \"\"\"Receive sampled token ids broadcast from last PP stage\"\"\"
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        # skip for chunked prefill.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv
""",
    """    def _pp_broadcast_spec_state(self) -> None:
        \"\"\"TQ_PP: PP + async + spec decode. Broadcast from the last stage, after the drafter, one int32
        tensor [num_reqs, 2 + num_spec_tokens] = [next_token_id, valid_sampled_count, draft_0..draft_{k-1}].
        Mirrors what _copy_valid_sampled_token_count()/propose_draft_token_ids() left on this rank.\"\"\"
        pp = get_pp_group()
        assert pp.is_last_rank
        _tq_dbg = _tq_pp_debug()
        if self._is_all_reqs_chunked_prefill():
            if _tq_dbg:
                _tq_dbg(f"BCAST-SKIP(all chunked) num_reqs={self.input_batch.num_reqs}")
            return
        num_reqs = self.input_batch.num_reqs
        k = self.num_spec_tokens
        if _tq_dbg:
            _tq_dbg(f"BCAST num_reqs={num_reqs} k={k} reqs={list(self.input_batch.req_ids)}")
        packed = torch.zeros((num_reqs, 2 + k), dtype=torch.int32, device=self.device)
        prev = self.input_batch.prev_sampled_token_ids
        if prev is not None:
            packed[:, 0] = prev[:num_reqs, 0].to(torch.int32)
        counts = self.valid_sampled_token_count_gpu
        if counts is not None:
            packed[:, 1] = counts[:num_reqs].to(torch.int32)
        else:
            packed[:, 1] = 1
        drafts = self._draft_token_ids
        if torch.is_tensor(drafts) and drafts.numel() > 0 and drafts.dim() == 2:
            kk = min(k, drafts.shape[1])
            packed[:, 2 : 2 + kk] = drafts[:num_reqs, :kk].to(torch.int32)
        torch.distributed.broadcast(packed, src=pp.rank, group=pp.device_group)

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        \"\"\"Receive sampled token ids broadcast from last PP stage\"\"\"
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        k = self.num_spec_tokens
        if k > 0:
            # TQ_PP: spec decode - packed [next_token, valid_count, drafts...] (see _pp_broadcast_spec_state)
            recv = torch.zeros((num_reqs, 2 + k), dtype=torch.int32, device=self.device)
            recv[:, 1] = 1  # neutralnie: 1 sampled, 0 zaakceptowanych draftow (gdy broadcast pominiety)
            _tq_dbg = _tq_pp_debug()
            if not self._is_all_reqs_chunked_prefill():
                if _tq_dbg:
                    _tq_dbg(f"RECV num_reqs={num_reqs} k={k} reqs={list(self.input_batch.req_ids)}")
                torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
            elif _tq_dbg:
                _tq_dbg(f"RECV-SKIP(all chunked) num_reqs={num_reqs}")
            next_token_ids = recv[:, 0].contiguous()
            valid_counts = recv[:, 1].contiguous()
            # sets valid_sampled_token_count_gpu (+ async CPU copy/event) and prev_sampled_token_ids [num_reqs,1]
            self._copy_valid_sampled_token_count(next_token_ids, valid_counts)
            if self.input_batch.prev_sampled_token_ids is None:
                self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)
            self._draft_token_ids = recv[:, 2:].contiguous()
            self.prev_num_spec_tokens = k
        else:
            # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
            recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
            # skip for chunked prefill.
            if not self._is_all_reqs_chunked_prefill():
                torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
            self.input_batch.prev_sampled_token_ids = recv
""",
)

# j) _update_states: wyrownanie output_token_ids do scheduler-owego num_output_tokens (placeholdery + optymistyczne
#    drafty) dzialalo tylko na ostatnim stage'u (elif). Na PP0 lista rosla bez konca -> req.num_tokens > seq_len ->
#    discard_request_mask=True -> PP0 pomijal odbior broadcastu, PP1 nadawal -> deadlock GPU po kilku krokach.
patch(
    R,
    """            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure, or output_token_ids was inflated by the optimistic
                # extend above (async spec decode). Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]""",
    """            if num_output_tokens < len(req_state.output_token_ids):  # TQ_PP: takze na stage'ach != last
                # Some output tokens were discarded due to a sync-KV-load
                # failure, or output_token_ids was inflated by the optimistic
                # extend above (async spec decode). Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]""",
)

# k) debug: po zlozeniu input_ids (async) zaloguj tokeny kroku (male batche) - porownanie PP0 vs PP1
patch(
    R,
    """        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )
""",
    """        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )
        if (_d := _tq_pp_debug()) and total_num_scheduled_tokens <= 16:  # TQ_PP debug
            torch.cuda.synchronize()
            _d(
                f"INPUT_IDS step tokens={total_num_scheduled_tokens} ids={self.input_ids.gpu[:total_num_scheduled_tokens].tolist()} "
                f"prev_sampled={self.input_batch.prev_sampled_token_ids[:, 0].tolist()} "
                f"counts={self.valid_sampled_token_count_gpu.tolist() if self.valid_sampled_token_count_gpu is not None else None} "
                f"ncomp_cpu={self.input_batch.num_computed_tokens_cpu[:num_reqs].tolist()}"
            )
""",
)

# ---- llm_base_proposer: przy PP>1 draft MTP nie moze wspoldzielic embed_tokens targetu (sa na stage'u 0),
#      a checkpoint GLM-5.3 nie ma model.layers.78.embed_tokens -> embed drafta zostawal LOSOWY -> drafty-smieci
#      (akceptacja 1.3 tok/krok). Ladujemy model.embed_tokens.weight z checkpointu do embed_tokens drafta (TP-shard
#      przez weight_loader VocabParallelEmbedding).
P = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py")
patch(
    P,
    """        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:""",
    """        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )
            if not hasattr(self.model, "has_own_embed_tokens"):
                # TQ_PP: MTP - embed_tokens targetu zyje na pierwszym stage'u PP; checkpoint zwykle nie ma
                # kopii pod warstwa MTP, wiec zaladuj model.embed_tokens.weight z checkpointu do drafta.
                self._tq_load_target_embed_tokens()

    def _tq_load_target_embed_tokens(self) -> None:
        \"\"\"TQ_PP: zaladuj `model.embed_tokens.weight` z checkpointu targetu do embed_tokens drafta (ostatni stage PP).\"\"\"
        import glob
        import json
        import os

        from safetensors import safe_open

        inner = getattr(self.model, "model", None)
        draft_embed = getattr(inner, "embed_tokens", None) if inner is not None else None
        if draft_embed is None or not isinstance(getattr(draft_embed, "weight", None), torch.Tensor):
            logger.warning("TQ_PP: draft model has no embed_tokens.weight to load into")
            return
        model_path = self.vllm_config.model_config.model
        key = "model.embed_tokens.weight"
        files: list[str] = []
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            with open(idx_path) as fh:
                wm = json.load(fh).get("weight_map", {})
            if key in wm:
                files = [os.path.join(model_path, wm[key])]
        if not files:
            files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                if key not in sf.keys():
                    continue
                w = sf.get_tensor(key)
                loader = getattr(draft_embed.weight, "weight_loader", None)
                if loader is not None:
                    loader(draft_embed.weight, w)
                else:
                    draft_embed.weight.data.copy_(w.to(draft_embed.weight.dtype))
                logger.info(
                    "TQ_PP: loaded %s %s from %s into the draft embed_tokens (last PP stage)",
                    key, tuple(w.shape), os.path.basename(f),
                )
                return
        logger.warning("TQ_PP: %s not found in %s - draft embeddings stay uninitialised!", key, model_path)

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:""",
)

# ---- indexer.py: bufor expanded_block_table ma szerokosc cdiv(max_model_len, spec.block_size*cp) (=11719 @750k/64),
#      a block table = cdiv(max_model_len, group_block*cp) * (group_block/kernel_block) (=5860*2=11720, gdy grupa KV
#      ma blok 128 a kernel 64). Sciezka "variable decode lengths" (mieszane dlugosci decode, np. PP2+MTP @16 strumieni)
#      kopiuje cala szerokosc -> RuntimeError. Dopasowujemy bufor do szerokosci block table (raz, przy pierwszym buildzie,
#      przed capture cudagraphow; stary bufor trzymamy przy zyciu na wypadek juz zlapanych grafow).
I = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py")
patch(
    I,
    """        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
""",
    """        min_decode_len = int(decode_lens_cpu.min().item())
        if block_table.shape[1] != self.expanded_block_table_buffer.shape[1]:  # TQ_PP
            logger.warning(
                "indexer: expanded_block_table_buffer width %d != block_table width %d; "
                "reallocating to match (kv group block != kernel block).",
                self.expanded_block_table_buffer.shape[1],
                block_table.shape[1],
            )
            self._tq_old_buffers = getattr(self, "_tq_old_buffers", []) + [
                self.expanded_block_table_buffer
            ]
            self.expanded_block_table_buffer = torch.zeros(
                (self.expanded_block_table_buffer.shape[0], block_table.shape[1]),
                dtype=self.expanded_block_table_buffer.dtype,
                device=self.expanded_block_table_buffer.device,
            )
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
""",
)
assert "logger = init_logger(__name__)" in I.read_text(), "indexer.py bez loggera"

# ---- gpu_worker: debug send/recv tensorow posrednich (VLLM_TQ_PP_DEBUG=1)
W = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
patch(
    W,
    """        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(""",
    """        if forward_pass and not get_pp_group().is_first_rank:
            from vllm.v1.worker.gpu_model_runner import _tq_pp_debug as _tqd  # TQ_PP

            if (_d := _tqd()):
                _d(f"IRECV intermediates tokens={scheduler_output.total_num_scheduled_tokens} reqs={len(scheduler_output.num_scheduled_tokens)}")
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(""",
)
patch(
    W,
    """        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(""",
    """        # launch non-blocking send of intermediate tensors
        from vllm.v1.worker.gpu_model_runner import _tq_pp_debug as _tqd  # TQ_PP

        if (_d := _tqd()):
            _d(f"ISEND intermediates tokens={scheduler_output.total_num_scheduled_tokens} reqs={len(scheduler_output.num_scheduled_tokens)}")
        self._pp_send_work = get_pp_group().isend_tensor_dict(""",
)

import ast

src = S.read_text()
ast.parse(src)
ast.parse(R.read_text())
ast.parse(W.read_text())
ast.parse(P.read_text())
ast.parse(I.read_text())
assert "logger = init_logger(__name__)" in src or "logger = " in src, "brak loggera w speculative.py"
print("SYNTAX-OK")
print("PATCH-PP-MTP-DONE")
