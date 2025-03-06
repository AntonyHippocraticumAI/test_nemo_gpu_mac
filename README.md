# Testing NeMo with Whisper by Antony :)

Testing the repository to understand the future implementation of NeMo and Whisper collaboration in the transcription server.

- CPU support and eazy implement GPU support (need to add choice in future version v2).


## Step by step for local working

See [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for your OS 

I use MAC so run the following code:
```bash
cd test_nemo_cpu_mac/

brew install mecab

conda create --name <env_name> --file requirements/conda_requirements.txt

conda activate <env_name>

pip install requirements/requirements.txt
```
