from Library.yoloTrainingProcessing import processImagesYolo

inputD = "/Users/lorenzovigorelli/Library/CloudStorage/OneDrive-UniversitàdegliStudidiPadova/MAGISTRALE UNIPD/Third semester/Computer vision/OBJECT-RECOGNITION/data"
bestweights = "/Users/lorenzovigorelli/Library/CloudStorage/OneDrive-UniversitàdegliStudidiPadova/MAGISTRALE UNIPD/Third semester/Computer vision/OBJECT-RECOGNITION/TRAFFIC SIGNALS/runs/yolov9c_automatic_20250119_102802/train/weights/best.pt"
input_directory = inputD  
output_directory = "results/100epochsautov9"  
trained_model_path = bestweights  
base_model = "yolov5s.pt"  

inference = True  

processImagesYolo(
    inference=inference,
    inputDirectory=input_directory,
    outputDirectory=output_directory,
    trainedModelPath=trained_model_path,
    base_model=base_model,
)