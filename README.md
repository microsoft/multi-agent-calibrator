# Semantic Kernel Auto Calibrator
Automatically Calibrate Semantic Kernel Project Code

# Local Development Setup 

python -m venv .venv
.venv\Scripts\Activate.ps1

# Conceptual overview of the Semi-Auto Calibrator
This research breaks down the Multi-Agent RAG system into modifiable or re-assemble components, making it eligible for generating component variants, and topology variants. The algorithm to do semi-auto calibration is to first (semi) auto generate N component variant or topology variant for a given use case. N variants make an experiment. To run the experiment, the calibrator updates multi-agent architecture object for each variant. Evaluate with a calibrator test set which includes the fields to do evaluation, i.e., expected answers, etc. By gathering all the evaluation metrics, the calibrator suggests the best variant. Then the developers could check in the code with suggested modifications and update the service automatically.