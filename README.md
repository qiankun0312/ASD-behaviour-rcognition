# Environment：

Ubuntu20.04
Pytorch：1.13.1
Cuda：11.7.1


# Data process:

python data_process_STGCN.py

python data_process_STGCN_2.py


# Train：

cd mmaction2

python tools/train.py configs/skeleton/stgcnpp/stgcnpp_mmad_binary.py --seed 0 --deterministic # binary classification

python tools/train.py configs/skeleton/stgcnpp/stgcnpp_mmad_22.py --seed 0 --deterministic # 22-Class Classification


# Test:

python tools/test.py configs/skeleton/stgcnpp/stgcnpp_mmad_binary.py checkpoints/SOME_CHECKPOINT.pth --dump result_binary.pkl # binary classification

python tools/test.py configs/skeleton/stgcnpp/stgcnpp_mmad_binary.py checkpoints/SOME_CHECKPOINT.pth --dump result_22.pkl # 22-Class Classification


# Result analysis(binary classification):

python analyze_binary.py


# Result analysis(22-Class Classification):

python analyze_multi.py

python confusion_matrix.py
