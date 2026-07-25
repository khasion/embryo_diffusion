"""Shared model architectures for diffusion training and evaluation.

xThis module is the single source of truth for the model classes used by the
training and evaluation notebooks: VideoContextUNet (primary), TemporalAttnBlock,
SinusoidalTimeEmbedding, and EfficientNetV2MGray.  Notebooks import from here
instead of copy-pasting class definitions.

VideoContextUNet — conditional next-frame diffusion model.

  Borrows two components from the video-diffusion literature and specialises them
  to a *fixed conditional next-frame task* (predict frame t+1 from the previous k):

    * Backbone (VDM; Ho et al., 2022, §3): a factorized space-time UNet.  All
      K = k+1 frames (k context + 1 noisy target) are batch-expanded through the
      SAME UNet weights; spatial layers act per-frame and a TemporalAttnBlock after
      every spatial attention block mixes information across the K frames.

    * Observed/latent mask channel (FDM; Harvey et al., NeurIPS 2022, §4): each
      frame carries a binary channel (1 = observed/context, 0 = latent/target).

  Conditioning is *explicit* (CSDI-style; Tashiro et al., 2021): the k context
  frames are supplied clean as model inputs and only the target frame is noised.
  This is NOT VDM's joint reconstruction-guided conditioning, and the flexible
  task distribution of FDM (arbitrary observed/latent subsets) is not used — the
  task here is a single fixed autoregressive one.  DiffusionCond is an alias for
  VideoContextUNet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from torchvision.models import (
    EfficientNet_V2_M_Weights,
    efficientnet_v2_m,
    efficientnet_v2_s,
)


# ---------------------------------------------------------------------------
# TemporalAttnBlock – factorized temporal self-attention (VDM §3)
# ---------------------------------------------------------------------------

class TemporalAttnBlock(nn.Module):
    """Temporal self-attention for factorized space-time attention (Ho et al., 2022, §3 — VDM).

    Placed after every spatial attention block inside the UNet, allowing each spatial
    position to attend across all K video frames (context + noisy target) simultaneously.
    This is exactly the architecture described in VDM §3: spatial attention operates within
    each frame independently, temporal attention operates across all K frames at each
    spatial position, giving factorized space-time attention.

    Input/output: (B*K, C, H, W) — frames are batch-expanded by the caller.
    Internally reshapes to (B*H*W, K, C) for multi-head self-attention over the K-frame axis.

    Relative-distance positional embeddings distinguish frame ordering without requiring
    absolute video timestamps (VDM §3: "We use relative position embeddings in each
    temporal attention block so that the network can distinguish ordering of frames
    in a way that does not require an absolute notion of video time").

    Zero-initialised output projection → identity transform at initialisation; the spatial
    UNet training starts undisturbed and temporal mixing is learned gradually.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        max_rel_idx: int = 64,
    ):
        super().__init__()
        # Ensure n_heads divides channels (reduce if necessary)
        nh = n_heads
        while nh > 1 and channels % nh != 0:
            nh -= 1
        self.n_heads     = nh
        self.max_rel_idx = max_rel_idx

        self.norm        = nn.GroupNorm(min(8, channels), channels)
        self.attn        = nn.MultiheadAttention(channels, nh, batch_first=True, dropout=0.0)
        self.attn_norm   = nn.LayerNorm(channels)
        self.rel_pos_emb = nn.Embedding(max_rel_idx + 1, channels)
        nn.init.normal_(self.rel_pos_emb.weight, std=0.02)
        self.out_proj    = nn.Linear(channels, channels, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        K: int,
        rel_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B*K, C, H, W)  batch-expanded frame features
            K:       number of frames in the block
            rel_idx: (B, K) long  distance of each frame from the target frame
                     (0 for the target; ascending for context, closest context = 1)
        Returns:
            (B*K, C, H, W) with temporal context mixed in via residual.
        """
        BK, C, H, W = x.shape
        B = BK // K

        x_n = self.norm(x)  # (B*K, C, H, W)

        # Reshape for temporal attention: (B*K, C, H, W) → (B*H*W, K, C)
        x_n = x_n.view(B, K, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)

        # Relative-distance positional embeddings → add before attention
        if rel_idx is None:
            rel = torch.arange(K - 1, -1, -1, device=x.device).unsqueeze(0).expand(B, K)
        else:
            rel = rel_idx.clamp(0, self.max_rel_idx).long()
        pos_emb = self.rel_pos_emb(rel)                                            # (B, K, C)
        pos_emb = (pos_emb.unsqueeze(1).unsqueeze(1)
                   .expand(B, H, W, K, C).reshape(B * H * W, K, C))
        x_n = x_n + pos_emb

        # Self-attention across K frames at each spatial position
        res = x_n
        x_n = self.attn_norm(x_n)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x_n = res + self.out_proj(attn_out)                                        # (B*H*W, K, C)

        # Reshape back: (B*H*W, K, C) → (B*K, C, H, W), then residual
        x_n = x_n.view(B, H, W, K, C).permute(0, 3, 4, 1, 2).reshape(B * K, C, H, W)
        return x + x_n


# ---------------------------------------------------------------------------
# VideoContextUNet – conditional next-frame diffusion (VDM backbone + FDM mask)
# ---------------------------------------------------------------------------

class VideoContextUNet(nn.Module):
    """Conditional next-frame diffusion model with factorized space-time attention.

    Borrows the VDM factorized space-time UNet *backbone* (Ho et al., 2022, §3) and the
    FDM observed/latent *mask channel* (Harvey et al., NeurIPS 2022, §4), and specialises
    them to a fixed conditional next-frame task.  Conditioning is explicit / CSDI-style
    (Tashiro et al., 2021): the k context frames are supplied clean as inputs and only the
    target is noised.  This is NOT VDM's joint reconstruction-guided conditioning, and the
    flexible task distribution of FDM is not used (the task is a single fixed autoregressive
    one) — so the model is "VDM/FDM-inspired", not VDM-faithful.

    Architecture — VDM §3:
        All K = k+1 frames (k context + 1 noisy target) pass through the SAME
        UNet weights simultaneously by batch-expanding along the frame axis:
            (B, K, in_ch, H, W) → reshape → (B*K, in_ch, H, W) → UNet
        Each 2D spatial convolution operates independently on each frame
        (equivalent to VDM's 1×3×3 space-only convolutions).  After EACH spatial
        attention block a TemporalAttnBlock allows each spatial position to attend
        across all K frames — factorized space-time attention as in VDM §3.
        Only the target frame's output (index K-1 in each group) is returned.

    Training objective — VDM §3 / DDPM:
        Standard conditional DDPM / v-prediction loss.  k context frames are always
        provided explicitly (video prediction setting, VDM §4.2) — this is a fixed
        autoregressive task, not the flexible task distribution of Harvey et al.

    Per-frame input channels:
        [pixel (1ch)] + [mask (1ch, 1=observed / 0=latent)] + [time_map (1ch, optional)]
        The binary mask channel is adopted from Harvey et al. §4 to explicitly signal
        observed vs. latent frames.  Relative positional embeddings in each
        TemporalAttnBlock follow VDM §3 (no absolute frame timestamps required).

    CFG: context pixels and mask channel are zeroed for the unconditional branch
    (Ho & Salimans, 2022; VDM §2, classifier-free guidance).

    Temporal attention injection:
        TemporalAttnBlocks are wired via forward hooks on every Transformer2DModel /
        AttentionBlock inside the UNet.  Channel widths are discovered by a one-shot dry-run
        in __init__ so no manual channel bookkeeping is required.
        Zero-initialised output projections → identity at start.
    """

    def __init__(
        self,
        context_k: int,
        use_time_map: bool = True,
        img_size: int = 256,
        output_channels: int = 1,
        max_frame_index: int = 800,
        time_emb_dim: int = 8,
        block_out_channels: tuple[int, ...] = (32, 64, 128, 256, 256),
        layers_per_block: int = 1,
        n_attn_stages: int = 3,       # deepest N stages get spatial + temporal attention
        temporal_attn_heads: int = 4,
        max_ctx_dist: int = 64,
        cfg_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.context_k          = context_k
        self.output_channels    = output_channels
        self.use_time_map       = use_time_map
        self.max_frame_index    = max_frame_index
        self.block_out_channels = tuple(int(c) for c in block_out_channels)
        self.layers_per_block   = int(layers_per_block)
        self.cfg_dropout_prob   = float(cfg_dropout_prob)
        self.max_ctx_dist       = max_ctx_dist
        self._n_attn_stages     = int(n_attn_stages)
        # FDM §4: mask channel always active; attribute kept for API compatibility
        self.use_context_mask   = True

        # Per-frame input: pixel + mask + (time_map, optional)
        self._in_ch = output_channels + 1 + (1 if use_time_map else 0)

        if use_time_map:
            self.frame_emb  = nn.Embedding(max_frame_index + 1, time_emb_dim)
            self.frame_proj = nn.Linear(time_emb_dim, 1)
        else:
            self.frame_emb  = None
            self.frame_proj = None

        n_stages  = len(self.block_out_channels)
        n_plain   = max(0, n_stages - n_attn_stages)
        down_blocks = ["DownBlock2D"] * n_plain + ["AttnDownBlock2D"] * n_attn_stages
        up_blocks   = ["AttnUpBlock2D"] * n_attn_stages + ["UpBlock2D"] * n_plain

        attn_ch       = self.block_out_channels[n_plain]
        attn_head_dim = 8
        while attn_head_dim > 1 and attn_ch % attn_head_dim != 0:
            attn_head_dim //= 2

        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=self._in_ch,
            out_channels=output_channels,
            layers_per_block=self.layers_per_block,
            block_out_channels=self.block_out_channels,
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
            norm_num_groups=min(8, self.block_out_channels[0]),
            attention_head_dim=attn_head_dim,
        )

        # State set before each forward pass so hooks see the right K and rel_idx
        self._K: int = context_k + 1
        self._rel_idx: torch.Tensor | None = None

        # Discover attention-block channel counts, create TemporalAttnBlocks, wire hooks
        self._temporal_attns: nn.ModuleList = nn.ModuleList()
        self._hook_handles: list = []
        self._inject_temporal_attention(temporal_attn_heads, max_ctx_dist, img_size)

    # ------------------------------------------------------------------
    # Temporal attention injection
    # ------------------------------------------------------------------

    def _inject_temporal_attention(
        self, n_heads: int, max_rel_idx: int, img_size: int
    ) -> None:
        """Discover attention-block channel widths via a dry-run, then inject TemporalAttnBlocks.

        A single forward pass with a dummy (1, in_ch, H, W) tensor collects the output
        channel count of every Transformer2DModel / AttentionBlock in the UNet.  One
        TemporalAttnBlock per spatial-attention module is then created and wired in via a
        permanent forward hook that fires immediately after the spatial attention step.
        Module traversal order is deterministic in PyTorch (DFS), so the dry-run order and
        the permanent-hook order are guaranteed to match.
        """
        # Diffusers ≥0.15 uses 'Attention' in UNet2DModel; older versions used
        # 'AttentionBlock'.  UNet2DConditionModel uses 'Transformer2DModel'.
        # All three are matched so the injection works across diffusers versions.
        _ATTN_CLASS_NAMES = frozenset(('Attention', 'AttentionBlock', 'Transformer2DModel'))

        # Dry-run: collect (module_id, channels) in FORWARD EXECUTION ORDER.
        # IMPORTANT: use module IDs (not DFS traversal order) for permanent hook wiring
        # because diffusers registers up_blocks before mid_block in UNet2DModel.__init__,
        # so DFS order diverges from execution order at mid_block → wrong block assignment.
        exec_order: list[tuple[int, int]] = []   # [(id(module), channels), ...]

        def _ch_hook(module, inp, out):
            h = out.sample if hasattr(out, 'sample') else out
            if isinstance(h, torch.Tensor):
                exec_order.append((id(module), h.shape[1]))

        tmp_hooks = []
        for m in self.unet.modules():
            if m.__class__.__name__ in _ATTN_CLASS_NAMES:
                tmp_hooks.append(m.register_forward_hook(_ch_hook))

        with torch.no_grad():
            dummy = torch.zeros(1, self._in_ch, img_size, img_size)
            self.unet(dummy, torch.zeros(1, dtype=torch.long))

        for h in tmp_hooks:
            h.remove()

        assert exec_order, (
            "VideoContextUNet._inject_temporal_attention: no attention modules found in "
            f"UNet2DModel (searched for {sorted(_ATTN_CLASS_NAMES)}).  "
            "Check diffusers version and UNet block types."
        )

        # Create one TemporalAttnBlock per discovered spatial attention module (execution order)
        for _, channels in exec_order:
            self._temporal_attns.append(
                TemporalAttnBlock(channels, n_heads=n_heads, max_rel_idx=max_rel_idx)
            )

        # Fail loudly if the discovered count does not match the count implied by the
        # UNet2DModel block layout.  For n_attn_stages attention stages and
        # layers_per_block residual layers per stage, diffusers builds:
        #   down: n_attn_stages * layers_per_block          attention blocks
        #   mid : 1                                          attention block
        #   up  : n_attn_stages * (layers_per_block + 1)     attention blocks
        # A mismatch means a diffusers version changed the block structure (or the
        # attention class names), which previously caused temporal attention to be
        # silently dropped — leaving the model context-blind across frames.
        expected_ta = (
            self._n_attn_stages * self.layers_per_block
            + 1
            + self._n_attn_stages * (self.layers_per_block + 1)
        )
        assert len(self._temporal_attns) == expected_ta, (
            f"VideoContextUNet: injected {len(self._temporal_attns)} TemporalAttnBlocks "
            f"but expected {expected_ta} for n_attn_stages={self._n_attn_stages}, "
            f"layers_per_block={self.layers_per_block}.  The diffusers UNet2DModel block "
            "layout or attention class names may have changed — verify temporal attention "
            "is still being injected after every spatial attention block."
        )

        # Wire permanent forward hooks using module IDs captured during dry-run.
        # This guarantees each attention module gets its matching TemporalAttnBlock
        # regardless of DFS vs. execution order discrepancies.
        id_to_ta = {mod_id: ta for (mod_id, _), ta in zip(exec_order, self._temporal_attns)}
        for m in self.unet.modules():
            if id(m) in id_to_ta:
                ta = id_to_ta[id(m)]

                def make_hook(temporal_attn: TemporalAttnBlock):
                    def hook(module, inp, out):
                        h = out.sample if hasattr(out, 'sample') else out
                        h = temporal_attn(h, K=self._K, rel_idx=self._rel_idx)
                        if hasattr(out, 'sample'):
                            return out.__class__(sample=h)
                        return h
                    return hook

                self._hook_handles.append(m.register_forward_hook(make_hook(ta)))

    # ------------------------------------------------------------------
    # Time-map helper
    # ------------------------------------------------------------------

    def build_time_map(
        self, frame_idx: torch.Tensor | int, h: int, w: int
    ) -> torch.Tensor:
        if not self.use_time_map:
            raise ValueError("build_time_map called but use_time_map=False")
        if not torch.is_tensor(frame_idx):
            frame_idx = torch.tensor(
                frame_idx, device=next(self.parameters()).device, dtype=torch.long
            )
        if frame_idx.dim() == 0:
            frame_idx = frame_idx.view(1)
        frame_idx = frame_idx.clamp(0, self.max_frame_index).long()
        emb    = self.frame_emb(frame_idx)
        scalar = self.frame_proj(emb)
        return scalar.view(-1, 1, 1, 1).expand(-1, 1, h, w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        x_ctx: torch.Tensor,
        time_map: torch.Tensor | None = None,
        frame_idx: torch.Tensor | None = None,
        uncond: bool = False,
        ctx_rel_idx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,   # (B,) float: 1=cond, 0=uncond; per-sample CFG dropout
    ) -> torch.Tensor:
        """
        Args:
            x_noisy:     (B, 1, H, W)  noisy target frame
            t:           (B,)           diffusion timestep (shared across all K frames)
            x_ctx:       (B, k, H, W)  context frames
            time_map:    (B, 1, H, W)  prebuilt absolute frame-index map for target
            frame_idx:   (B,)           target frame absolute index (for per-frame time maps)
            uncond:      bool           zero context + mask for CFG unconditional branch
            ctx_rel_idx: (B, k) long    distance of each context frame from the target
        Returns:
            (B, 1, H, W)  predicted noise or velocity for the target frame only.
        """
        if x_ctx.dim() == 3:
            x_ctx = x_ctx.unsqueeze(0)

        B, _, H, W = x_noisy.shape
        k = self.context_k
        K = k + 1

        # Binary mask channel: 1 = observed (context), 0 = latent (target)  [FDM §4]
        # ctx_mask (B,) float: per-sample CFG dropout (1=cond, 0=uncond) from p_losses.
        # uncond bool: batch-wide unconditional pass (inference-time CFG).
        if ctx_mask is not None:
            m = ctx_mask.float().view(B, 1, 1, 1)
            x_ctx     = x_ctx * m                                                          # zero uncond pixels
            ctx_masks = m.view(B, 1, 1, 1, 1).expand(B, k, 1, H, W).contiguous()         # 0=latent per FDM §4
        elif uncond:
            x_ctx     = torch.zeros_like(x_ctx)
            ctx_masks = torch.zeros(B, k, 1, H, W, device=x_noisy.device, dtype=x_noisy.dtype)
        else:
            ctx_masks = torch.ones(B, k, 1, H, W, device=x_noisy.device, dtype=x_noisy.dtype)
        tgt_mask = torch.zeros(B, 1, 1, H, W, device=x_noisy.device, dtype=x_noisy.dtype)

        # Expand pixel tensors to 5-D: (B, frames, 1, H, W)
        x_ctx_5d   = x_ctx.unsqueeze(2)     # (B, k, 1, H, W)
        x_noisy_5d = x_noisy.unsqueeze(1)   # (B, 1, 1, H, W)

        ctx_parts = [x_ctx_5d, ctx_masks]
        tgt_parts = [x_noisy_5d, tgt_mask]

        if self.use_time_map:
            # Target frame time map
            if time_map is None:
                if frame_idx is None:
                    raise ValueError("frame_idx or time_map required when use_time_map=True")
                time_map = self.build_time_map(frame_idx, H, W)          # (B, 1, H, W)
            tgt_parts.append(time_map.unsqueeze(1))                      # (B, 1, 1, H, W)

            # Per-context-frame time maps: derived from target index minus relative distance.
            # Gives the model absolute temporal position for every frame in the window.
            if frame_idx is not None and ctx_rel_idx is not None:
                ctx_fi      = (frame_idx.unsqueeze(1) - ctx_rel_idx).clamp(0, self.max_frame_index)
                ctx_emb     = self.frame_emb(ctx_fi.view(B * k))
                ctx_scalars = self.frame_proj(ctx_emb)                   # (B*k, 1)
                ctx_time    = ctx_scalars.view(B, k, 1, 1, 1).expand(B, k, 1, H, W)
            else:
                ctx_time = torch.zeros(
                    B, k, 1, H, W, device=x_noisy.device, dtype=x_noisy.dtype
                )
            ctx_parts.append(ctx_time)

        # Assemble per-frame input and batch-expand: (B, K, in_ch, H, W) → (B*K, in_ch, H, W)
        ctx_input = torch.cat(ctx_parts, dim=2)   # (B, k, in_ch, H, W)
        tgt_input = torch.cat(tgt_parts, dim=2)   # (B, 1, in_ch, H, W)
        all_input = (
            torch.cat([ctx_input, tgt_input], dim=1)
            .view(B * K, self._in_ch, H, W)
        )

        # Full relative-distance index for all K frames: [ctx_rel_idx | 0 (target)]
        if ctx_rel_idx is not None:
            tgt_rel      = torch.zeros(B, 1, device=x_noisy.device, dtype=torch.long)
            full_rel_idx = torch.cat([ctx_rel_idx, tgt_rel], dim=1)     # (B, K)
        else:
            full_rel_idx = (
                torch.arange(K - 1, -1, -1, device=x_noisy.device)
                .unsqueeze(0).expand(B, K).contiguous()
            )

        # Update hook state, then run UNet with all K frames sharing timestep t (FDM design)
        self._K       = K
        self._rel_idx = full_rel_idx
        t_exp = t.repeat_interleave(K)                              # (B*K,)
        out   = self.unet(all_input, t_exp).sample                  # (B*K, 1, H, W)

        # Extract target frame output — last frame in each group of K
        return out.view(B, K, self.output_channels, H, W)[:, -1]   # (B, 1, H, W)


# ---------------------------------------------------------------------------
# DiffusionCond alias – points to the current recommended model
# ---------------------------------------------------------------------------

DiffusionCond = VideoContextUNet  # canonical alias; overwrites the None placeholder above


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding – used by the phase classifier
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_timesteps: int = 1000, max_period: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps
        self.max_period = max_period

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        t = t.float() / max(1, self.max_timesteps - 1)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, device=t.device).float()
            / max(1, half)
        )
        args = t[:, None] * freqs[None] * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


# ---------------------------------------------------------------------------
# EfficientNetV2MGray – noise-aware grayscale phase classifier
# ---------------------------------------------------------------------------

class EfficientNetV2MGray(nn.Module):
    def __init__(
        self,
        n_classes: int,
        time_dim: int = 32,
        max_timesteps: int = 1000,
        weights=None,
        backbone: str = "efficientnet_v2_m",
        dropout: float = 0.5,
    ):
        super().__init__()
        if backbone == "efficientnet_v2_s":
            self.base = efficientnet_v2_s(weights=weights)
        else:
            self.base = efficientnet_v2_m(weights=weights)
        old_conv = self.base.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        if weights is not None:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True) / 3.0
        self.base.features[0][0] = new_conv

        in_feats = self.base.classifier[1].in_features
        self.base.classifier = nn.Identity()

        self.time_embed = SinusoidalTimeEmbedding(time_dim, max_timesteps=max_timesteps)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats + time_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        feats = self.base.features(x)
        feats = self.base.avgpool(feats)
        feats = torch.flatten(feats, 1)
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        t_emb = self.time_mlp(self.time_embed(t.long()))
        feats = torch.cat([feats, t_emb], dim=1)
        return self.classifier(feats)
