import deeplabcut

deeplabcut.video_inference_superanimal(
    [r"C:\Users\sam20\OneDrive\Desktop\petbehavior-dlc\input\14877397_1080_2320_30fps.mp4"],
    "superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    video_adapt=False,
)

print("DONE")
