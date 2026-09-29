# ODEWorld

[[📄 Paper]](https://arxiv.org/abs/2607.27924) &nbsp; [[🌐 Website]](https://dstate.github.io/odeworld_website/) &nbsp; [[🤗 Hugging Face]](https://huggingface.co/collections/ldxxx/odeworld)

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

Example images and manifests are stored in `assets/examples/<dataset>/`.

Run the LIBERO examples:

```bash
python demo_infer.py --dataset libero
```

Run the AgiBot examples:

```bash
python demo_infer.py --dataset agibot
```

Use `--case-ids case_00` to run a single case. Results are written to `outputs/<dataset>/<case_id>`.

## Training

### Data Preparation

Run the following commands from the repository root to link your prepared HDF5 datasets into `assets/data`. Replace the source paths with your dataset locations:

```bash
mkdir -p assets/data
ln -s /path/to/libero assets/data/libero
ln -s /path/to/agibot_hdf5 assets/data/agibot_hdf5
```

The linked directories must match the trajectory paths in `assets/metas/*.json`:

```text
assets/data/
├── libero/
│   └── libero_90/<task>/demo_0.hdf5
└── agibot_hdf5/
    └── demo_0.hdf5
```

The loaders expect per-trajectory HDF5 files with encoded image frames and the fields specified by the metadata. Use datasets prepared in this format. Each `.hdf5` file contains one trajectory of `T` frames. The current dataset layouts are:

```text
# LIBERO: demo_*.hdf5
observation/
    third_image          # (T,), variable-length uint8 arrays of JPEG bytes
    wrist_image          # (T,), variable-length uint8 arrays of JPEG bytes
language_instruction     # scalar UTF-8 string describing the trajectory
action                   # (T, 7), float32
proprio                  # (T, 9), float32

# AgiBot: demo_*.hdf5
observation/
    head_image           # (T,), variable-length uint8 arrays of JPEG bytes
language_instruction     # scalar UTF-8 string describing the trajectory
action                   # (T, 16), float32
proprio                  # (T, 16), float32
```

Image datasets have HDF5 dtype `h5py.vlen_dtype(np.dtype("uint8"))`. Each element is a complete encoded image, decoded by `cv2.imdecode(frame, cv2.IMREAD_COLOR)`.


### Training Commands

Run from the repository root and choose the commands for your dataset.

1. Train the RAE image decoder.

   ```bash
   bash scripts/train_dinov2rae_libero.sh
   bash scripts/train_dinov2rae_agibot.sh
   ```

2. Train the PT-Flow dynamics model.

   ```bash
   bash scripts/train_dinov2ptflow_libero.sh
   bash scripts/train_dinov2ptflow_agibot.sh
   ```

3. Train the goal predictor for language-conditioned LIBERO inference.

   ```bash
   bash scripts/train_dinov2goalpred_libero.sh
   ```

## Reference
```bash
@article{liu-niu2026odeworld,
  title={ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow},
  author={Liu, Dongxiu and Niu, Haoyi and Cheng, Peng and Gao, Yuan and Kang, Xirui and Teng, Sangli and Sreenath, Koushil and Zhan, Xianyuan},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```
