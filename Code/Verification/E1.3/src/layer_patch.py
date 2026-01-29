"""
E1.3: PriceMoE Integration Patch for vLLM layer.py

This file documents the changes needed to integrate PriceMoE into vLLM's
FusedMoE layer. Apply these changes to:
    external/vllm/vllm/model_executor/layers/fused_moe/layer.py

The integration is minimal and non-invasive - it modifies router_logits
before the existing routing logic, preserving all existing functionality.

HOW TO APPLY:
1. Copy the integration code block below
2. Insert it at the beginning of the _select_experts method, right after
   the docstring (around line 1570)

INTEGRATION CODE:
-----------------
"""

# Add this import at the top of layer.py (around line 30):
IMPORT_BLOCK = '''
import os  # if not already imported

# PriceMoE integration (E1.3)
def _get_price_moe_router():
    """Get PriceMoE router if enabled."""
    if os.environ.get("VLLM_PRICE_MOE_ENABLED", "0") == "1":
        try:
            from vllm.model_executor.layers.fused_moe.price_moe_router import (
                get_or_create_router, PriceMoERouter
            )
            return get_or_create_router, PriceMoERouter
        except ImportError:
            pass
    return None, None
'''

# Add this code at the beginning of _select_experts method, after the docstring:
INTEGRATION_BLOCK = '''
        # === PriceMoE Integration (E1.3) ===
        # Modify router_logits with congestion penalty before routing
        _get_router, _PriceMoERouter = _get_price_moe_router()
        if _get_router is not None:
            try:
                router = _get_router(
                    layer_name=getattr(self, 'layer_name', f'layer_{id(self)}'),
                    num_experts=router_logits.shape[-1],
                )
                if isinstance(router, _PriceMoERouter):
                    # Apply congestion penalty: s_modified = s - alpha * q_e
                    router_logits = router.modify_logits(router_logits)
            except Exception as e:
                # Fail silently to not break inference
                import logging
                logging.getLogger(__name__).debug(f"PriceMoE failed: {e}")
        # === End PriceMoE Integration ===
'''

# Full modified _select_experts signature for reference:
MODIFIED_METHOD_START = '''
    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids)
                (tuple[torch.Tensor, torch.Tensor]):
                The weights and expert ids.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        # === PriceMoE Integration (E1.3) ===
        _get_router, _PriceMoERouter = _get_price_moe_router()
        if _get_router is not None:
            try:
                router = _get_router(
                    layer_name=getattr(self, 'layer_name', f'layer_{id(self)}'),
                    num_experts=router_logits.shape[-1],
                )
                if isinstance(router, _PriceMoERouter):
                    router_logits = router.modify_logits(router_logits)
            except Exception:
                pass
        # === End PriceMoE Integration ===

        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )
        # ... rest of the method unchanged ...
'''

# After routing is complete, we also want to update the queue.
# Add this after topk_ids is computed (around line 1650):
POST_ROUTING_UPDATE = '''
        # === PriceMoE Queue Update (E1.3) ===
        if _get_router is not None:
            try:
                router = _get_router(
                    layer_name=getattr(self, 'layer_name', f'layer_{id(self)}'),
                    num_experts=router_logits.shape[-1],
                )
                if isinstance(router, _PriceMoERouter):
                    router.update_queue(topk_ids)
            except Exception:
                pass
        # === End PriceMoE Queue Update ===
'''

# ============================================================================
# ALTERNATIVE: Minimal patch using environment variable check only
# ============================================================================
# If you prefer not to modify layer.py, you can also enable PriceMoE by:
# 1. Setting custom_routing_function in model config
# 2. Using the VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var
#
# However, the layer.py patch provides better integration and allows
# the PriceMoE router to maintain per-layer state.
# ============================================================================
