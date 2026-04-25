import torch

# Standard ImageNet-1k class indices for common training subsets.
# Teacher checkpoints are trained on full ImageNet (1000 classes; embedding table shape [1001, D]).
# When the student uses N classes we must remap the teacher's y_embedder to [N+1, D]
# by selecting the N relevant rows plus the shared null token (last row).

# 10 diverse classes used in early JiT experiments.
IMAGENET10_INDICES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

# First 50 classes in alphabetical WNID order (the standard ImageNet-1k class ordering).
# WNIDs: n01440764 (tench, idx 0) … n01697457 (African crocodile, idx 49).
IMAGENET50_INDICES = list(range(50))

_SUBSET_INDICES: dict[int, list[int]] = {
    10: IMAGENET10_INDICES,
    50: IMAGENET50_INDICES,
}

_CLS_KEY = "y_embedder.embedding_table.weight"


def remap_teacher_y_embedder(state_dict: dict, class_num: int) -> dict:
    """
    Remap a full-ImageNet teacher y_embedder to a smaller class subset.

    Selects the rows corresponding to the subset classes plus the null token
    and writes the result back into state_dict in-place.

    Raises RuntimeError for unrecognised teacher/student class-count pairs.
    """
    if _CLS_KEY not in state_dict:
        return state_dict

    tw = state_dict[_CLS_KEY]
    expected = class_num + 1  # N classes + null token

    if tw.shape[0] == expected:
        return state_dict

    if tw.shape[0] != 1001:
        raise RuntimeError(
            f"Teacher y_embedder has {tw.shape[0]} rows; expected 1001 (full ImageNet). "
            "No remapping is available for this checkpoint."
        )

    if class_num not in _SUBSET_INDICES:
        raise RuntimeError(
            f"Teacher y_embedder shape mismatch ({tw.shape[0]} vs {expected}) "
            f"and no hardcoded mapping exists for {class_num} classes. "
            f"Supported subset sizes: {sorted(_SUBSET_INDICES)}."
        )

    indices = _SUBSET_INDICES[class_num]
    null_row = tw[-1].unsqueeze(0)  # last row is always the null token
    rows = torch.cat([tw[indices], null_row], dim=0)
    state_dict[_CLS_KEY] = rows
    print(
        f"[INFO] Remapped teacher y_embedder: {tw.shape[0]} → {rows.shape[0]} rows "
        f"using {class_num}-class ImageNet subset."
    )
    return state_dict
