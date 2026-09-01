"""Safety contract for every field exposed to the V2 research planner."""
from __future__ import annotations


FIELD_REGISTRY = {
    "user_id": ("exposure_feature", "可作输入", "用户身份；仅由训练期行为构造其历史特征。"),
    "video_id": ("exposure_feature", "可作输入", "视频身份；不可用未来交互统计直接增强。"),
    "author_id": ("exposure_feature", "可作输入", "曝光前已知的视频作者。"),
    "tab": ("exposure_feature", "可作输入", "曝光前页面上下文。"),
    "duration_ms": ("exposure_feature", "可作输入", "曝光前已知内容长度。"),
    "date": ("exposure_feature", "可作输入", "可派生星期/相对日期；不能让测试参与开发。"),
    "hourmin": ("exposure_feature", "可作输入", "可派生小时桶。"),
    "long_view": ("primary_label", "仅作主标签", "官方固定相关性标签。"),
    "is_click": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_like": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_follow": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_comment": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_forward": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_hate": ("post_exposure_label", "仅作负反馈辅助标签", "曝光后行为，禁止作推理输入。"),
    "is_profile_enter": ("post_exposure_label", "仅作辅助标签", "曝光后行为，禁止作推理输入。"),
    "play_time_ms": ("post_exposure_label", "仅作连续辅助标签", "观看结果，禁止作推理输入。"),
    "profile_stay_time": ("post_exposure_label", "仅作连续辅助标签", "曝光后行为，禁止作推理输入。"),
    "comment_stay_time": ("post_exposure_label", "仅作连续辅助标签", "曝光后行为，禁止作推理输入。"),
    "video_features_statistic_pure.csv": ("leakage_review", "暂不使用", "全局统计的时间窗口未审计，可能包含未来信息。"),
    "log_random_4_22_to_5_08_pure.csv": ("unbiased_validation", "仅作随机曝光验证", "开发时仅可使用 validation 日期；不用于读取 hidden-test 标签。"),
}


def manifest():
    return [{"字段": name, "类别": kind, "允许操作": role, "原因": reason}
            for name, (kind, role, reason) in FIELD_REGISTRY.items()]
