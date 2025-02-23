Dependency
torch==2.1.2+cu121
transformers==4.43.1
diffusers==0.30.3
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
CUDA 12.1

To run the experiment, need to modify the path first.

cd src
bash conda create -n my_code_env python=3.10
bash conda activate my_code_env
bash pip install -r requirements.txt
bash run_main.sh

