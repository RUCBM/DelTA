


## Installation
Our code is mainly based on [verl](https://github.com/volcengine/verl). To prepare the environment, please follow these steps:

```bash
conda create -n delta python==3.12
conda activate delta
pip install torch==2.9.1
pip install flash_attn==2.8.3
pip install sglang==0.5.6
cd verl-DelTA
pip install -e.
pip install math-verify
```


## Train

We provide an example for DelTA training in the script `verl-DelTA/recipe/dapo/srcs/run_DelTA.sh`.

