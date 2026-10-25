"""Shorten temporary tensor lifetimes in the pinned upstream pipeline call."""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from collections.abc import Callable


def release_working_tensors(call: Callable, garment_cached: bool = False) -> Callable:
    """Adapt two checked anchors without modifying or duplicating upstream files.

    The denoiser outputs and guidance math are untouched. Only temporary local
    references whose final use has passed are removed. The runner installs a
    small recorder on the pipeline to verify actual allocation changes.
    """
    original = inspect.unwrap(call)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    counts = {"step": 0, "decode": 0}
    cache_counts = {"posterior": 0, "image": 0}

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (
                garment_cached
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "cloth"
                and ast.unparse(node.value)
                == "self.vae.encode(cloth).latent_dist.sample()"
            ):
                cache_counts["posterior"] += 1
                node.value = ast.parse(
                    "self._garment_posterior.sample()", mode="eval"
                ).body
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "latents"
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Call)
                and ast.unparse(node.value.value.func) == "self.scheduler.step"
            ):
                counts["step"] += 1
                additions = ast.parse(
                    'self._record_working_tensors("step_before_release", i)\n'
                    "del latent_model_input, garment_features, noise_pred, _\n"
                    "del added_cond_kwargs, timestep, timestep_garment\n"
                    "if self.do_classifier_free_guidance:\n"
                    "    del noise_pred_uncond, noise_pred_text\n"
                    'self._record_working_tensors("step_after_release", i)\n'
                ).body
                return [node, *additions]
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "self.vae_decoder"
            ):
                counts["decode"] += 1
                additions = ast.parse(
                    'self._record_working_tensors("decode_before_release", None)\n'
                    "del gen_condition, cloth, image_embeds\n"
                    "del prompt_embeds, pooled_prompt_embeds\n"
                    'self._record_working_tensors("decode_after_release", None)\n'
                ).body
                return [*additions, node]
            return node

        def visit_If(self, node):
            if (
                garment_cached
                and ast.unparse(node.test) == "ip_adapter_image is not None"
            ):
                cache_counts["image"] += 1
                return ast.parse("image_embeds = self._garment_image_embeds").body
            return self.generic_visit(node)

    adapted = Rewrite().visit(copy.deepcopy(tree))
    if counts != {"step": 1, "decode": 1}:
        raise ValueError(f"Unexpected upstream pipeline anchors: {counts}")
    if garment_cached and cache_counts != {"posterior": 1, "image": 1}:
        raise ValueError(f"Unexpected garment cache anchors: {cache_counts}")
    ast.fix_missing_locations(adapted)
    namespace = {}
    exec(  # noqa: S102 - transforms trusted, pinned upstream source only
        compile(adapted, original.__code__.co_filename + " [tensor release]", "exec"),
        original.__globals__,
        namespace,
    )
    return namespace[original.__name__]
