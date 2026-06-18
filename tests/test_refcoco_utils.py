from tools.refcoco_utils import (
    canonical_image_name,
    compute_iou,
    deterministic_sample,
    expression_tags,
    flatten_sentences,
    normalize_expression,
    resize_dimensions,
    xywh_to_xyxy,
)


def test_normalize_expression():
    assert normalize_expression("  red   car  ") == "red car ."
    assert normalize_expression("red car.") == "red car ."
    assert normalize_expression("red car", append_period=False) == "red car"


def test_box_helpers():
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert compute_iou([0, 0, 5, 5], [5, 5, 10, 10]) == 0.0


def test_resize_dimensions():
    assert resize_dimensions(640, 480) == (1066, 800)
    assert max(resize_dimensions(2000, 500)) <= 1333


def test_sampling_is_stable():
    samples = [{"sent_id": index, "ref_id": index} for index in range(20)]
    first = deterministic_sample(samples, 5, 2026, "refcoco_unc_val")
    second = deterministic_sample(list(reversed(samples)), 5, 2026, "refcoco_unc_val")
    assert first == second
    assert len(first) == 5


def test_canonical_image_name():
    assert canonical_image_name(581857) == "COCO_train2014_000000581857.jpg"


def test_flatten_sentences_and_tags():
    refs = [{
        "split": "val",
        "ref_id": 4,
        "ann_id": 9,
        "image_id": 10,
        "category_id": 1,
        "sentences": [
            {"sent_id": 1, "sent": "the woman in red"},
            {"sent_id": 2, "sent": "person on the left"},
        ],
    }]
    annotations = {9: {"bbox": [1, 2, 3, 4]}}
    samples = flatten_sentences(refs, annotations, "val")
    assert len(samples) == 2
    assert samples[0]["gt_box_xywh"] == [1.0, 2.0, 3.0, 4.0]
    tags = expression_tags("the partially hidden woman in red", [0, 0, 5, 5], 100, 100)
    assert {"attribute", "occlusion", "person", "small_target"} <= set(tags)
