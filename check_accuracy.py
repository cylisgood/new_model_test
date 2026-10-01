import deeplabcut

deeplabcut.video_inference_superanimal(
    [r"C:\Users\sam20\OneDrive\Desktop\petbehavior-dlc\input\test.mp4"],
    "superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    video_adapt=False,
)

print("DONE")
