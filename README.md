# ODEWorld

[[📄 Paper]](https://arxiv.org/abs/2607.27924) &nbsp; [[🌐 Website]](https://dstate.github.io/odeworld_website/) &nbsp; [[🤗 Hugging Face]](https://huggingface.co/ldxxx?search=ODEWorld)

The Official Implementation of "ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow"

## Quick Start

### Installation

1. Clone this repository and create the environment.

```bash
git clone https://github.com/Dstate/ODEWorld.git
cd ODEWorld
conda create -n odeworld python=3.10 -y
conda activate odeworld
```

2. Install the remaining dependencies.

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. Download the pretrained checkpoints into `assets/pretrained`.

```bash
mkdir -p assets/pretrained

for model in \
  ODEWorld-PT-Flow-LIBERO \
  ODEWorld-PT-Flow-AgiBot \
  ODEWorld-Goal-Predictor-LIBERO \
  ODEWorld-RAE-LIBERO \
  ODEWorld-RAE-AgiBot
do
  hf download "ldxxx/${model}" --local-dir "assets/pretrained/${model}"
done
```

## Demo Inference

Run the LIBERO examples:

```bash
python demo_infer.py --dataset libero
```

Run the AgiBot examples:

```bash
python demo_infer.py --dataset agibot
```

Use `--case-ids case_00` to run a single case. Results are written to `outputs/<dataset>/<case_id>`.
