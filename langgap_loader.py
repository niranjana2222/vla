"""
Loads episodes from the LangGap HuggingFace dataset (LeRobot format).

Expected directory layout (after huggingface-cli download):
    langgap_hf/
        meta/tasks.parquet                          task_index -> instruction
        meta/episodes/chunk-000/file-000.parquet    episode metadata
        data/chunk-000/file-000.parquet             per-frame: index, episode_index,
                                                    frame_index, task_index, ...
        videos/observation.images.image/
            chunk-000/file-000.mp4                  camera frames (all episodes concatenated)

The global `index` column in the data parquet is the frame's position in the
MP4 file, so decoding frame N from the video gives the observation for the
data row where index==N.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


class LangGapLoader:
    """
    Samples frames from LangGap episodes, extracts images from MP4 video,
    and returns {image, instruction, label} dicts ready for probe training.

    Args:
        langgap_dir:        path to HF dataset root (e.g. ./langgap_hf)
        scene_ids:          task indices to include (None = all)
        max_per_scene:      unused, kept for API compatibility
        image_size:         resize frames to this square size
        frames_per_episode: frames sampled per episode (1 = middle frame)
        camera:             which camera stream to use ('image' or 'image2')
    """

    def __init__(
        self,
        langgap_dir: str,
        scene_ids: Optional[list] = None,
        max_per_scene: int = 3,
        image_size: int = 224,
        frames_per_episode: int = 1,
        camera: str = "image",
    ):
        self.root = Path(langgap_dir)
        self.scene_ids = scene_ids
        self.image_size = image_size
        self.frames_per_episode = frames_per_episode
        self.camera = camera

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {langgap_dir}\n"
                "Download with:\n"
                "  huggingface-cli download YC11Hou/langgap_6 "
                "--repo-type dataset --local-dir ./langgap_hf"
            )

    def _load_tasks(self) -> dict[int, str]:
        """Returns {task_index: instruction_string}."""
        df = _read_parquet(self.root / "meta" / "tasks.parquet")
        # tasks.parquet: index = instruction string, column = task_index
        return {int(row["task_index"]): instr for instr, row in df.iterrows()}

    def _load_frames_df(self) -> pd.DataFrame:
        parts = []
        for f in sorted((self.root / "data").rglob("*.parquet")):
            parts.append(_read_parquet(f))
        return pd.concat(parts, ignore_index=True)

    def _load_episodes_df(self) -> pd.DataFrame:
        parts = []
        for f in sorted((self.root / "meta" / "episodes").rglob("*.parquet")):
            parts.append(_read_parquet(f))
        return pd.concat(parts, ignore_index=True)

    def _video_path(self, chunk_index: int, file_index: int) -> Path:
        return (
            self.root
            / "videos"
            / f"observation.images.{self.camera}"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )

    def _extract_frame(self, video_path: Path, frame_pos: int) -> Image.Image:
        """Extract a single frame by its position (0-based) in the MP4."""
        try:
            import decord
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            frame = vr[frame_pos].asnumpy()
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(
                    f"Could not read frame {frame_pos} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert("RGB")
        return img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def load(self) -> list[dict]:
        """
        Returns list of samples:
            {image, instruction, label (=task_index), scene_id, variant}
        """
        tasks = self._load_tasks()
        frames_df = self._load_frames_df()
        episodes_df = self._load_episodes_df()

        # episode_index -> (chunk_index, file_index) for video lookup
        ep_to_chunk = {
            int(row["episode_index"]): (
                int(row["meta/episodes/chunk_index"]),
                int(row["meta/episodes/file_index"]),
            )
            for _, row in episodes_df.iterrows()
        }

        if self.scene_ids is not None:
            frames_df = frames_df[frames_df["task_index"].isin(self.scene_ids)]

        # The global `index` column = frame position in the concatenated MP4.
        # Chunk 0 starts at index 0; subsequent chunks start at their first index.
        chunk_start: dict[tuple, int] = {}
        for (chunk_idx, file_idx), group in frames_df.groupby(
            frames_df["episode_index"].map(ep_to_chunk)
        ):
            key = (chunk_idx, file_idx)
            chunk_start[key] = min(chunk_start.get(key, int(group["index"].min())),
                                   int(group["index"].min()))

        samples = []
        n_episodes = frames_df["episode_index"].nunique()

        for episode_index, ep_frames in frames_df.groupby("episode_index"):
            ep_frames = ep_frames.sort_values("frame_index")
            task_index = int(ep_frames["task_index"].iloc[0])
            instruction = tasks.get(task_index, f"task_{task_index}")

            chunk_idx, file_idx = ep_to_chunk[episode_index]
            video_path = self._video_path(chunk_idx, file_idx)
            start = chunk_start[(chunk_idx, file_idx)]

            n = len(ep_frames)
            if self.frames_per_episode == 1:
                pick_indices = [n // 2]
            else:
                pick_indices = list(
                    np.linspace(0, n - 1, self.frames_per_episode, dtype=int)
                )

            for pick in pick_indices:
                row = ep_frames.iloc[pick]
                video_frame_pos = int(row["index"]) - start
                try:
                    image = self._extract_frame(video_path, video_frame_pos)
                except Exception as e:
                    print(
                        f"  Warning: skipping episode {episode_index} "
                        f"frame {video_frame_pos}: {e}"
                    )
                    continue
                samples.append({
                    "image": image,
                    "instruction": instruction,
                    "label": task_index,
                    "scene_id": f"episode_{episode_index}",
                    "variant": f"task_{task_index}",
                })

        print(f"Loaded {len(samples)} samples from {n_episodes} episodes")
        if samples:
            print(f"  Labels: {sorted(set(s['label'] for s in samples))}")
            print(f"  Example: '{samples[0]['instruction']}'")
        else:
            print("  WARNING: 0 samples — check dataset path and structure")
        return samples

    @property
    def n_classes(self) -> int:
        if self.scene_ids is not None:
            return len(self.scene_ids)
        return len(self._load_tasks())

    @property
    def class_names(self) -> list[str]:
        tasks = self._load_tasks()
        ids = self.scene_ids if self.scene_ids is not None else sorted(tasks)
        return [tasks.get(i, f"task_{i}") for i in ids]
