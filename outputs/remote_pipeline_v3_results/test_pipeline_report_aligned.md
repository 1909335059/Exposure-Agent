# ExposureAgent 实验报告

## 场景 0007_001

- 数据划分：`test`
- 最终停止原因：`quality_satisfactory`
- 实际轮数：`1`

### 原始输入

![原始输入图像](data/sidd_srgb_subset/Data/0007_001_S6_00100_00100_5500_L/GT_SRGB_010.PNG)

- 初始参数：`{"image_id": "GT_SRGB_010", "iso": 100, "shutter_speed_s": 0.01, "ev": 6.643856189774724, "aperture": null}`
- 亮度直方图 bins：`32`

### 第 1 轮

- 上一轮反馈：`无，使用原始输入`
- 第一次 VLM 建议：`{"quality": {"brightness": 0.2086813896894455, "noise": 0.0, "motion_blur": 0.04325330705193142, "highlight": 0.0, "shadow": 0.015173931870669746, "overall_quality": 0.9883145522154797}, "action": {"ISO": 100, "Shutter": 0.01}, "continue": false, "reason": "objective_quality_accepted_current_target"}`
- RAG 检索：`{"retrieval_score": 0.154703417631864, "distance_components": {"visual": 0.0011011756648378057, "histogram": 0.468283585331649, "quality": 0.06335235862180377, "exposure": 0.011624588253153122, "initial_action": 0.0}, "scene_id": "0001_001", "run_id": "pseudo-0001_001--1.00", "initial_metadata": {"image_id": "0001_001", "iso": 100, "shutter_speed_s": 0.008333333333333333, "ev": 6.906890595608519, "aperture": null}, "final_metadata": {"image_id": "0001_001", "iso": 100, "shutter_speed_s": 0.013333333333333334, "ev": 6.22881869049588, "aperture": null}, "initial_vlm_action": {"ISO": 100, "Shutter": 0.008333333333333333}, "integrated_vlm_action": {"ISO": 100, "Shutter": 0.008333333333333333}, "final_action": {"ISO": 100, "Shutter": 0.013333333333333334}, "quality_before": {"quality": {"brightness": 0.1506422609090805, "noise": 0.0, "motion_blur": 0.08595637599273959, "highlight": 0.0, "shadow": 0.116128993456505, "overall_quality": 0.9547915389395978}, "dynamic_range": 0.1952086128294468, "midtone_ratio": 0.705629931678214, "sharpness_confidence": 0.976043064147234, "exposure_score": 0.9419355032717475, "contrast_score": 0.976043064147234, "noise_score": 1.0, "sharpness_score": 0.9140436240072605, "acceptable": false, "calibration_version": "no_reference_v2_sidd_gt_srgb"}, "quality_after": {"quality": {"brightness": 0.19498997926712036, "noise": 0.0, "motion_blur": 0.0788862298899684, "highlight": 0.0, "shadow": 0.03144546285604311, "overall_quality": 0.9779336614507976}, "dynamic_range": 0.2380627691745758, "midtone_ratio": 0.7721835306004619, "sharpness_confidence": 1.0, "exposure_score": 0.9842772685719784, "contrast_score": 1.0, "noise_score": 1.0, "sharpness_score": 0.9211137701100316, "acceptable": true, "calibration_version": "no_reference_v2_sidd_gt_srgb"}, "quality_gain": 0.023142122511199847, "successful": true, "label_source": "train_search_pseudo_label"}`
- 第二次 VLM 综合建议：`{"quality": {"brightness": 0.2086813896894455, "noise": 0.0, "motion_blur": 0.04325330705193142, "highlight": 0.0, "shadow": 0.015173931870669746, "overall_quality": 0.9883145522154797}, "action": {"ISO": 100, "Shutter": 0.01}, "continue": false, "reason": "objective_quality_accepted_current_target"}`
- 网格搜索最终目标：`{"ISO": 100, "Shutter": 0.0125}`
- 最终参数：`{"image_id": "GT_SRGB_010", "iso": 100, "shutter_speed_s": 0.0125, "ev": 6.321928094887363, "aperture": null}`
- 相对原图质量增益：`0.002441`
- 曝光是否满意：`True`

![第 1 轮输出图像](artifacts/remote_pipeline_v3_test_aligned/0007_001-5fb1bfff/round_01.png)

### 最终最佳结果

![最终最佳图像](artifacts/remote_pipeline_v3_test_aligned/0007_001-5fb1bfff/round_01.png)

- 最终参数：`{"image_id": "GT_SRGB_010", "iso": 100, "shutter_speed_s": 0.0125, "ev": 6.321928094887363, "aperture": null}`
- 最终客观质量：`{"quality": {"brightness": 0.23437349498271942, "noise": 0.0, "motion_blur": 0.04404082050544411, "highlight": 0.0, "shadow": 0.002180162625096228, "overall_quality": 0.990755803373892}, "dynamic_range": 0.274175688624382, "midtone_ratio": 0.8558536614703618, "sharpness_confidence": 1.0, "exposure_score": 0.9989099186874518, "contrast_score": 1.0, "noise_score": 1.0, "sharpness_score": 0.9559591794945559, "acceptable": true, "calibration_version": "no_reference_v2_sidd_gt_srgb"}`
