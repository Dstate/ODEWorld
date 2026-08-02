from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from torchdiffeq import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import Dinov2GoalPred, Dinov2PTflowImgoal, Dinov2RAE


PRETRAINED_ROOT = ROOT / "assets" / "pretrained"
MODEL_DIRS = {
    "libero": {
        "pt_flow": PRETRAINED_ROOT / "ODEWorld-PT-Flow-LIBERO",
        "rae": PRETRAINED_ROOT / "ODEWorld-RAE-LIBERO",
        "goal_predictor": PRETRAINED_ROOT / "ODEWorld-Goal-Predictor-LIBERO",
    },
    "agibot": {
        "pt_flow": PRETRAINED_ROOT / "ODEWorld-PT-Flow-AgiBot",
        "rae": PRETRAINED_ROOT / "ODEWorld-RAE-AgiBot",
        "goal_predictor": None,
    },
}


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).to(device)


def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).round().byte().numpy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@torch.inference_mode()
def decode_latents(rae: Dinov2RAE, latents: torch.Tensor, chunk_size: int) -> list[np.ndarray]:
    frames = []
    for start in range(0, latents.shape[0], chunk_size):
        decoded = rae.decode(latents[start : start + chunk_size]).clamp(0, 1)
        frames.extend(tensor_to_bgr(frame) for frame in decoded)
    return frames


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_goal_comparison(
    start_image: torch.Tensor,
    predicted_goal: torch.Tensor,
    ground_truth_goal: torch.Tensor,
    path: Path,
) -> None:
    labels = ("Start", "Predicted Goal", "Ground Truth")
    panels = []
    for image, label in zip(
        (start_image[0], predicted_goal[0], ground_truth_goal[0]),
        labels,
    ):
        panel = tensor_to_bgr(image).copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise RuntimeError(f"Failed to write image: {path}")


@torch.inference_mode()
def save_pca_field(
    model: Dinov2PTflowImgoal,
    start_image: torch.Tensor,
    goal_image: torch.Tensor,
    output_dir: Path,
    *,
    horizon: float,
    steps: int,
    fps: int,
    grid_density: int = 20,
    arrow_dt: float = 0.02,
    arrow_scale: float = 2.0,
) -> None:
    """Project the PT-Flow rollout and time-conditioned velocity field into 2D PCA space."""
    if model.num_delta_tokens != 1:
        raise ValueError("PCA field visualization currently requires one delta token")

    batch_size = start_image.shape[0]
    start_latent = model.latent_encode(start_image)
    goal_latent = model.latent_encode(goal_image)
    z0 = model.delta_decouple(start_latent, start_latent)
    zg = model.delta_decouple(start_latent, goal_latent)

    def ode_func(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tau = time.reshape(1, 1).expand(batch_size, 1)
        return model.forward_vmodel(z0, state, zg, tau) * model.max_time_length

    time_grid = torch.linspace(0, horizon, steps + 1, device=start_image.device)
    trajectory = odeint(ode_func, z0, time_grid, method="rk4")

    traj_np = trajectory.squeeze(1).mean(dim=1).cpu().numpy()
    goal_np = zg.mean(dim=1).cpu().numpy()
    pca = PCA(n_components=2).fit(np.concatenate([traj_np, goal_np], axis=0))
    traj_2d = pca.transform(traj_np)
    goal_2d = pca.transform(goal_np)

    axis_range = max(3.0, float(np.abs(np.concatenate([traj_2d, goal_2d], axis=0)).max()) * 1.15)
    xx, yy = np.meshgrid(
        np.linspace(-axis_range, axis_range, grid_density),
        np.linspace(-axis_range, axis_range, grid_density),
    )
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_high = torch.tensor(
        pca.inverse_transform(grid_2d),
        dtype=z0.dtype,
        device=z0.device,
    ).unsqueeze(1)
    count = grid_high.shape[0]

    fields = []
    for step in range(1, steps + 1):
        tau = torch.full((count, 1), step * horizon / steps, device=z0.device)
        velocity = model.forward_vmodel(
            z0.expand(count, -1, -1),
            grid_high,
            zg.expand(count, -1, -1),
            tau,
        ) * model.max_time_length
        next_2d = pca.transform((grid_high + velocity * arrow_dt).squeeze(1).cpu().numpy())
        displacement = next_2d - grid_2d
        u = displacement[:, 0].reshape(grid_density, grid_density)
        v = displacement[:, 1].reshape(grid_density, grid_density)
        fields.append((u, v, np.sqrt(u**2 + v**2)))

    output_dir.mkdir(parents=True, exist_ok=True)
    norm = Normalize(vmin=0.0, vmax=max(max(float(speed.max()) for _, _, speed in fields), 1e-8))
    fig = plt.figure(figsize=(6, 6), dpi=120)
    axis = fig.add_axes([0.12, 0.10, 0.70, 0.78])
    color_axis = fig.add_axes([0.86, 0.10, 0.03, 0.78])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="cool"), cax=color_axis)
    colorbar.set_label("Latent velocity")
    axis.set_xlim(-axis_range, axis_range)
    axis.set_ylim(-axis_range, axis_range)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.set_title("PT-Flow Latent Velocity Field", fontweight="bold")
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.scatter(traj_2d[0, 0], traj_2d[0, 1], c="black", s=35, marker="o", zorder=5)

    u, v, speed = fields[0]
    quiver = axis.quiver(
        xx, yy, u * arrow_scale, v * arrow_scale, speed,
        cmap="cool", norm=norm, angles="xy", scale_units="xy", scale=1, width=0.004, alpha=0.55,
    )
    line, = axis.plot([], [], color="red", linewidth=3, zorder=6)
    marker = axis.scatter([], [], c="red", s=150, marker="*", zorder=7)
    time_text = axis.text(
        0.02, 0.97, "", transform=axis.transAxes, fontsize=13, fontweight="bold",
        ha="left", va="top", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    video_path = output_dir / "pca_field.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    for index, (u, v, speed) in enumerate(fields, start=1):
        quiver.set_UVC(u * arrow_scale, v * arrow_scale, speed)
        line.set_data(traj_2d[: index + 1, 0], traj_2d[: index + 1, 1])
        marker.set_offsets(traj_2d[index : index + 1])
        time_text.set_text(f"tau={index * horizon / steps:.3f}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ODEWorld demo results from PNG inputs")
    parser.add_argument("--dataset", choices=sorted(MODEL_DIRS), required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    defaults = MODEL_DIRS[args.dataset]
    dataset_dir = ROOT / "assets" / args.dataset
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    selected_ids = set(args.case_ids) if args.case_ids else None
    cases = [case for case in manifest["cases"] if selected_ids is None or case["id"] in selected_ids]
    if not cases:
        raise ValueError("No matching cases found")

    rae = Dinov2RAE.from_pretrained(str(defaults["rae"])).to(device).eval()
    pt_flow = Dinov2PTflowImgoal.from_pretrained(str(defaults["pt_flow"])).to(device).eval()
    goal_predictor = None
    if defaults["goal_predictor"] is not None:
        goal_predictor = Dinov2GoalPred.from_pretrained(
            str(defaults["goal_predictor"])
        ).to(device).eval()

    output_root = Path(args.output_dir) / args.dataset
    for case in cases:
        case_dir = output_root / case["id"]
        start_image = load_image(dataset_dir / case["start_image"], device)
        goal_image = load_image(dataset_dir / case["goal_image"], device)

        with torch.inference_mode():
            gt_goal_latents, _ = pt_flow.rollout_ode(
                start_image,
                goal_image,
                horizon=args.horizon,
                steps=args.steps,
            )
        gt_goal_frames = decode_latents(rae, gt_goal_latents[0], args.decode_chunk_size)
        save_video(gt_goal_frames, case_dir / "gt_goal_rollout.mp4", args.fps)
        save_pca_field(
            pt_flow,
            start_image,
            goal_image,
            case_dir,
            horizon=args.horizon,
            steps=args.steps,
            fps=args.fps,
        )

        if goal_predictor is not None:
            language = [case["instruction"]]
            with torch.inference_mode():
                predicted_goal_latent = goal_predictor.predict(start_image, language)
                predicted_goal_image = rae.decode(predicted_goal_latent).clamp(0, 1)
                predicted_latents, _ = pt_flow.rollout_ode_lang(
                    start_image,
                    language,
                    goal_predictor,
                    horizon=args.horizon,
                    steps=args.steps,
                )
            save_goal_comparison(
                start_image,
                predicted_goal_image,
                goal_image,
                case_dir / "goal_prediction_comparison.png",
            )
            predicted_frames = decode_latents(rae, predicted_latents[0], args.decode_chunk_size)
            save_video(predicted_frames, case_dir / "predicted_goal_rollout.mp4", args.fps)

        print(f"Saved results for {args.dataset}/{case['id']} to {case_dir}")


if __name__ == "__main__":
    main()
