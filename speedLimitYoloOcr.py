from Library.yoloTrainingProcessing import processSpeedSignals

input_directory =  "/Users/lorenzovigorelli/Library/CloudStorage/OneDrive-UniversitàdegliStudidiPadova/MAGISTRALE UNIPD/Third semester/Computer vision/OBJECT-RECOGNITION/data"
output_directory = "results/OCR100epochsv9"  
trained_model_path = "/Users/lorenzovigorelli/Library/CloudStorage/OneDrive-UniversitàdegliStudidiPadova/MAGISTRALE UNIPD/Third semester/Computer vision/OBJECT-RECOGNITION/TRAFFIC SIGNALS/runs/yolov9c_automatic_20250119_102802/train/weights/best.pt"
base_model = "yolov5s.pt" 


inference = True 

# Esegui la funzione
processSpeedSignals(
    inference=inference,
    inputDirectory=input_directory,
    outputDirectory=output_directory,
    trainedModelPath=trained_model_path,
    base_model=base_model,
    confidence_threshold=0.1
)