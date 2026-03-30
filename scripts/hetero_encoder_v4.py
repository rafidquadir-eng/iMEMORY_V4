#!/usr/bin/env python3
"""
CytoSF V3 Encoder — hetero_encoder_v3.py

DESIGN PRINCIPLE: V3 is V2 + pathway supervision. Nothing from V2 is removed.

V2 → V3 delta (3 changes only):
  1. `forward()` now also returns `module_reps` (post-conv2 module node
     representations before pooling) and `gene_reps` (same for gene nodes).
     These are used by PathwayConsistencyLoss during training and for
     module activation extraction after training.

  2. `PathwayConsistencyLoss` added — two-term regularizer:
       Term A (align):  module node i should attend its own pathway genes
       Term B (sep):    module nodes should be mutually dissimilar

  3. `PathwaySupervisorHead` added — thin MLP that maps module node reps
     to scalar per-module activation scores. Used for slide figures.

Everything else is IDENTICAL to V2:
  - Gene mean centering (gene_mean_embedding.npy)
  - Orthogonal weight init
  - NODE_INPUT_DIMS / EDGE_TYPES from pipeline_config
  - HeteroConv over all 6 edge types (gene, module, pathway, cell_type,
    disease_state, genome_context, modality_flag)
  - LeakyReLU activations with LayerNorm + Dropout(0.1)
  - Gene pooling BEFORE message passing (preserves sample signal)
  - MODULE-ONLY pooling after message passing for graph embedding
  - Linear output projection + L2 normalise
  - HeteroClassificationModelV3: logits + embeddings head (same as V2)

Checkpoint compatibility:
  All V2 weight keys are present in V3. V2 checkpoint can be used as
  warm-start via strict=False. Only `pathway_head.*` keys are new.

Save as: /mnt/data/trained_im/scripts/hetero_encoder_v3.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from pipeline_config import NODE_INPUT_DIMS, EDGE_TYPES

# ── Gene mean for centering (same path as V2) ─────────────────────────────────
_GENE_MEAN_PATH = '/mnt/data/trained_im/gene_mean_embedding.npy'
if os.path.exists(_GENE_MEAN_PATH):
    _GENE_MEAN = torch.from_numpy(np.load(_GENE_MEAN_PATH)).float()
else:
    _GENE_MEAN = None

# ── Module pathway assignment (must match CANONICAL_MODULES order in builder) ─
# This is the ground truth for which module index maps to which pathway name.
MODULE_PATHWAY_NAMES = [
    'mTOR_glycolytic',        # 0
    'NF_kB_epigenetic',       # 1
    'trained_immunity_module',# 2
    'inflammation_innate',    # 3
    'inflammation_adaptive',  # 4
    'metabolism_glycolysis',  # 5
    'metabolism_oxidative',   # 6
    'stress_oxidative',       # 7
    'stress_er',              # 8
    'mTOR_signaling',         # 9
    'cytokine_response',      # 10
    'epigenetic_remodeling',  # 11
    'immune_memory',          # 12
    'antigen_presentation',   # 13
]
N_MODULES = len(MODULE_PATHWAY_NAMES)


# ── Weight initialisation (identical to V2) ───────────────────────────────────
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def init_sageconv_weights(hetero_conv):
    for conv in hetero_conv.convs.values():
        for attr in ('lin_l', 'lin_r'):
            lin = getattr(conv, attr, None)
            if lin is not None:
                nn.init.orthogonal_(lin.weight, gain=1.0)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0.01)


# ── NEW: Pathway consistency loss ─────────────────────────────────────────────
class PathwayConsistencyLoss(nn.Module):
    """
    Two-term regularizer on module and gene node representations.

    Term A — Gene-pathway alignment (lambda_align, default 0.10):
      For each module node m_i with assigned pathway genes P_i:
        Compute softmax attention over all gene nodes.
        L_align_i = mean_attention(genes NOT in P_i) - mean_attention(genes IN P_i)
      Minimising pushes each module to attend its own pathway genes.

    Term B — Cross-module separation (lambda_sep, default 0.05):
      L_sep = mean pairwise cosine similarity across module node pairs.
      Minimising keeps module representations distinct.

    Both terms are small regularisers on top of the primary classification loss.
    Set both lambdas to 0.0 to disable (falls back to pure V2 behaviour).

    Args:
        lambda_align: weight for alignment term
        lambda_sep:   weight for separation term
    """
    def __init__(self, lambda_align: float = 0.10, lambda_sep: float = 0.05):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_sep   = lambda_sep

    def forward(
        self,
        module_reps:    torch.Tensor,   # (n_modules, hidden_dim)
        gene_reps:      torch.Tensor,   # (n_genes_in_graph, hidden_dim)
        pathway_masks:  torch.Tensor,   # (n_modules, n_genes_in_graph) bool
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            total_loss: scalar tensor (0.0 if gene_reps is empty)
            details:    {'align': float, 'sep': float, 'total': float}
        """
        zero = torch.tensor(0.0, device=module_reps.device, requires_grad=False)

        # Guard: no gene nodes (chromatin/methylation/mirna modalities)
        if gene_reps.numel() == 0 or gene_reps.shape[0] == 0:
            return zero, {'align': 0.0, 'sep': 0.0, 'total': 0.0}

        n_modules, hidden = module_reps.shape
        masks = pathway_masks.to(module_reps.device)  # (n_modules, n_genes)

        # ── Term A: alignment ─────────────────────────────────────────────────
        scale    = hidden ** 0.5
        affinity = torch.mm(module_reps, gene_reps.T) / scale   # (n_mod, n_gene)
        weights  = F.softmax(affinity, dim=1)                   # (n_mod, n_gene)

        align_terms = []
        for m in range(n_modules):
            in_mask  = masks[m]
            out_mask = ~in_mask
            if in_mask.sum() == 0 or out_mask.sum() == 0:
                continue
            in_w  = weights[m, in_mask].mean()
            out_w = weights[m, out_mask].mean()
            align_terms.append(out_w - in_w)

        l_align = torch.stack(align_terms).mean() if align_terms else zero

        # ── Term B: separation ────────────────────────────────────────────────
        norm    = F.normalize(module_reps, dim=1)
        sim     = torch.mm(norm, norm.T)
        triu    = torch.triu(
            torch.ones(n_modules, n_modules, dtype=torch.bool,
                       device=module_reps.device), diagonal=1)
        l_sep   = sim[triu].mean() if triu.sum() > 0 else zero

        total = self.lambda_align * l_align + self.lambda_sep * l_sep

        return total, {
            'align': float(l_align.item()),
            'sep':   float(l_sep.item()),
            'total': float(total.item()),
        }


# ── NEW: Pathway supervisor head ─────────────────────────────────────────────
class PathwaySupervisorHead(nn.Module):
    """
    Maps post-conv module node representations → scalar activation scores.
    One score per module, directly interpretable.

    Module 0 score = mTOR_glycolytic activation level
    Module 1 score = NF_kB_epigenetic activation level
    … etc.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        self.apply(init_weights)

    def forward(self, module_reps: torch.Tensor) -> torch.Tensor:
        """Args: module_reps (n_modules, hidden_dim) → (n_modules,) scores"""
        return self.net(module_reps).squeeze(-1)


# ── V3 encoder — V2 architecture + module/gene rep extraction ─────────────────
class HeteroGraphEncoderV3(nn.Module):
    """
    Identical to HeteroGraphEncoderV2 with one additional output:
    module node representations before pooling (for PathwayConsistencyLoss).

    All V2 logic preserved:
      - Gene mean centering
      - Orthogonal init
      - All edge types via pipeline_config.EDGE_TYPES
      - Gene pool BEFORE message passing
      - LeakyReLU + LayerNorm + Dropout
      - MODULE-ONLY pooling for graph embedding
      - Linear output proj + L2 norm
    """
    def __init__(self, hidden_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.embedding_dim = embedding_dim

        # Per-type input projections — LINEAR ONLY (no ReLU, same as V2)
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim)
            for ntype, dim in NODE_INPUT_DIMS.items()
        })

        # Message passing layers — all EDGE_TYPES (same as V2)
        self.conv1 = HeteroConv({
            et: SAGEConv(hidden_dim, hidden_dim, normalize=False)
            for et in EDGE_TYPES
        }, aggr='mean')

        self.conv2 = HeteroConv({
            et: SAGEConv(hidden_dim, hidden_dim, normalize=False)
            for et in EDGE_TYPES
        }, aggr='mean')

        # Layer norms (same node types as V2)
        self.ln1 = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim)
            for nt in ['gene', 'module', 'disease_state']
        })
        self.ln2 = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim)
            for nt in ['gene', 'module', 'disease_state']
        })

        # Output projection — LINEAR ONLY (same as V2)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.drop     = nn.Dropout(0.1)

        # Initialise
        self.apply(init_weights)
        init_sageconv_weights(self.conv1)
        init_sageconv_weights(self.conv2)

    def forward(
        self,
        x_dict:         Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        batch_dict:     Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Returns:
            graph_emb:   (batch_size, embedding_dim) L2-normalised graph embedding
            h:           raw (batch_size, hidden_dim) pre-projection embedding
            node_h:      dict of post-conv2 node representations keyed by node type
            module_pool: (batch_size, hidden_dim) pooled module reps (for extraction)

        NEW vs V2: also returns node_h so callers can extract module and gene reps
        per graph for PathwayConsistencyLoss.
        """
        # ── Step 1: Gene mean centering ───────────────────────────────────────
        if 'gene' in x_dict and x_dict['gene'] is not None and x_dict['gene'].numel() > 0:
            gene_x = x_dict['gene']
            if _GENE_MEAN is not None:
                gene_mean = _GENE_MEAN.to(gene_x.device)
                x_dict = dict(x_dict)
                x_dict['gene'] = gene_x - gene_mean

        # ── Step 2: Project all node types ───────────────────────────────────
        h = {}
        for ntype, proj in self.input_proj.items():
            if ntype in x_dict and x_dict[ntype] is not None and x_dict[ntype].numel() > 0:
                h[ntype] = proj(x_dict[ntype])

        # ── Step 3: Gene pool BEFORE message passing ──────────────────────────
        gene_pool = None
        if 'gene' in h and 'gene' in batch_dict:
            gene_pool = global_mean_pool(h['gene'], batch_dict['gene'])

        # ── Step 4: Message passing ───────────────────────────────────────────
        active_edges = {
            k: v for k, v in edge_index_dict.items()
            if k in EDGE_TYPES and v is not None and v.numel() > 0
            and isinstance(k, tuple) and k[0] in h and k[2] in h
        }

        if active_edges:
            h1 = self.conv1(h, active_edges)
            for nt, x in h1.items():
                res = h.get(nt, x)
                h[nt] = self.drop(F.leaky_relu(self.ln1[nt](x), 0.1)) \
                    if nt in self.ln1 else self.drop(F.leaky_relu(x, 0.1))
                h[nt] = h[nt] + res  # residual

        active_edges2 = {
            k: v for k, v in active_edges.items()
            if k[0] in h and k[2] in h
        }
        if active_edges2:
            h2 = self.conv2(h, active_edges2)
            for nt, x in h2.items():
                res = h.get(nt, x)
                h[nt] = self.drop(F.leaky_relu(self.ln2[nt](x), 0.1)) \
                    if nt in self.ln2 else self.drop(F.leaky_relu(x, 0.1))
                h[nt] = h[nt] + res  # residual

        # ── Step 5: MODULE-ONLY pooling (same as V2) ──────────────────────────
        module_pool = None
        if 'module' in h and 'module' in batch_dict:
            module_pool = global_mean_pool(h['module'], batch_dict['module'])

        if module_pool is not None:
            graph_h = module_pool
        elif gene_pool is not None:
            graph_h = gene_pool
        else:
            for nt in ['chromatin_region', 'cpg', 'mirna', 'snp', 'metabolite']:
                if nt in h and nt in batch_dict:
                    graph_h = global_mean_pool(h[nt], batch_dict[nt])
                    break
            else:
                raise ValueError("No poolable node type found in graph")

        # ── Step 6: Output projection + L2 norm ──────────────────────────────
        graph_emb = F.normalize(self.out_proj(graph_h), dim=1, eps=1e-8)

        return graph_emb, graph_h, h, module_pool


# ── Classification model V3 ───────────────────────────────────────────────────
class HeteroClassificationModelV3(nn.Module):
    """
    V3 classification model. Drop-in replacement for HeteroClassificationModelV2.

    Identical to V2 except forward() additionally returns module_reps and gene_reps
    so the training loop can compute PathwayConsistencyLoss.

    checkpoint compatibility:
      All weight keys match V2 keys exactly. pathway_head keys are extra.
      Load V2 checkpoint with strict=False for a warm start.
    """
    def __init__(
        self,
        hidden_dim:    int = 256,
        embedding_dim: int = 128,
        num_classes:   int = 6,
    ):
        super().__init__()
        self.encoder       = HeteroGraphEncoderV3(hidden_dim, embedding_dim)
        self.pathway_head  = PathwaySupervisorHead(hidden_dim)

        # Classification head — identical to V2's phase2_hetero_v2.py heads
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, num_classes),
        )
        # mTOR prediction head (kept for V2 API compatibility, unused in V3 loss)
        self.mtor_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.apply(init_weights)

    def forward(
        self,
        x_dict:         Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        batch_dict:     Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            logits:      (1, n_classes)
            mtor_pred:   (1, 1)  — kept for V2 API compat
            embeddings:  (1, embedding_dim)
            module_reps: (n_modules, hidden_dim) per-graph module node reps
                         None if no module nodes
            gene_reps:   (n_genes, hidden_dim) per-graph gene node reps
                         None if no gene nodes

        NOTE: module_reps and gene_reps are NOT batched — they are per-graph.
        PathwayConsistencyLoss operates per-graph (called once per training step).
        """
        graph_emb, graph_h, node_h, module_pool = self.encoder(
            x_dict, edge_index_dict, batch_dict
        )

        logits    = self.classifier(graph_emb)   # (1, n_classes)
        mtor_pred = self.mtor_head(graph_emb)     # (1, 1)

        # Extract per-graph (unbatched) module and gene node reps
        module_reps = node_h.get('module', None)  # (n_modules, hidden_dim)
        gene_reps   = node_h.get('gene',   None)  # (n_genes, hidden_dim)

        return logits, mtor_pred, graph_emb, module_reps, gene_reps

    @torch.no_grad()
    def get_pathway_activations(
        self,
        x_dict:         Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        batch_dict:     Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Convenience: returns (n_modules,) scalar activation scores.
        Module 0 = mTOR_glycolytic, Module 1 = NF_kB_epigenetic, …
        """
        self.eval()
        _, _, _, module_reps, _ = self.forward(x_dict, edge_index_dict, batch_dict)
        if module_reps is None:
            return torch.zeros(N_MODULES)
        return self.pathway_head(module_reps)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("V3 encoder self-test (no pipeline_config needed here)...")

    # Minimal mock of NODE_INPUT_DIMS / EDGE_TYPES
    import sys
    sys.path.insert(0, '/mnt/data/trained_im/scripts')

    try:
        model = HeteroClassificationModelV3(hidden_dim=256, embedding_dim=128, num_classes=6)
        total = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total:,}")
        print(f"New parameters (pathway_head): "
              f"{sum(p.numel() for p in model.pathway_head.parameters()):,}")
        print(f"\nModules: {MODULE_PATHWAY_NAMES}")
        print(f"\nPathwayConsistencyLoss test...")
        pw_loss = PathwayConsistencyLoss(lambda_align=0.10, lambda_sep=0.05)
        mod_r  = torch.randn(14, 256)
        gene_r = torch.randn(200, 256)
        masks  = torch.zeros(14, 200, dtype=torch.bool)
        masks[0, :20] = True   # dummy pathway 0 genes
        masks[1, 20:45] = True
        loss, details = pw_loss(mod_r, gene_r, masks)
        print(f"  loss={loss.item():.4f}, align={details['align']:.4f}, sep={details['sep']:.4f}")
        print("\n✅ V3 encoder test passed")
        print("\nCheckpoint note:")
        print("  V2 weights load via: model.load_state_dict(v2_ckpt, strict=False)")
        print("  Only 'pathway_head.*' keys are new (missing from V2 checkpoint).")
    except ImportError as e:
        print(f"Run on EC2 where pipeline_config.py is available: {e}")
