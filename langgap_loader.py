"""
Loads episodes from the LangGap HuggingFace dataset (LeRobot format).

Expected directory layout (after huggingface-cli download):
    langgap_hf/
        meta/tasks.parquet
        data/chunk-000/file-000.parquet     per-frame: index, episode_index,
                                             frame_index, task_index
        videos/observation.images.image/
            chunk-000/file-000.mp4          AV1-encoded, all episodes concatenated

The global `index` column = frame position in the video (0-based, confirmed
to match video frame count). Requires pyav for AV1 software decoding.
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
    Samples frames from LangGap episodes, extracts images from AV1 video,
    and returns {image, instruction, label} dicts ready for probe training.

    Args:
        langgap_dir:        path to HF dataset root (e.g. ./langgap_hf)
        scene_ids:          task indices to include (None = all)
        max_per_scene:      unused, kept for API compatibility
        image_size:         resize frames to this square size
        frames_per_episode: frames sampled per episode (1 = middle frame)
        camera:             which camera stream ('image' or 'image2')
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
        try:
            import av  # noqa: F401
        except ImportError:
            raise ImportError(
                "AV1 video decoding requires pyav.\n"
                "Install with:  pip install av"
            )

    def _load_tasks(self) -> dict[int, str]:
        df = _read_parquet(self.root / "meta" / "tasks.parquet")
        # tasks.parquet: index = instruction string, 'task_index' column = int
        return {int(row["task_index"]): instr for instr, row in df.iterrows()}

    def _load_frames_df(self) -> pd.DataFrame:
        parts = []
        for f in sorted((self.root / "data").rglob("*.parquet")):
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

    def _extract_frames_pyav(
        self, video_path: Path, frame_positions: list[int]
    ) -> dict[int, np.ndarray]:
        """
        Seek to each requested position and decode one frame.
        Opens the video once for all positions (sorted for efficiency).
        Returns {frame_pos: rgb_array}.
        """
        import av

        results: dict[int, np.ndarray] = {}
        sorted_positions = sorted(set(frame_positions))

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
            avg_fps = float(stream.average_rate)
            time_base = float(stream.time_base)

            for target_pos in sorted_positions:
                # Seek to a keyframe at or before the target
                target_pts = int(target_pos / avg_fps / time_base)
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)

                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    current_frame = int(round(frame.pts * time_base * avg_fps))
                    if current_frame >= target_pos:
                        results[target_pos] = frame.to_ndarray(format="rgb24")
                        break

        return results

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        return (
            Image.fromarray(arr)
            .convert("RGB")
            .resize((self.image_size, self.image_size), Image.BICUBIC)
        )

    def load(self) -> list[dict]:
        """
        Returns list of {image, instruction, label (=task_index), scene_id, variant}.
        """
        tasks = self._load_tasks()
        frames_df = self._load_frames_df()

        if self.scene_ids is not None:
            frames_df = frames_df[frames_df["task_index"].isin(self.scene_ids)]

        # Build episode -> video path from episodes parquet
        ep_to_video: dict[int, Path] = {}
        for ep_parquet in sorted((self.root / "meta" / "episodes").rglob("*.parquet")):
            for _, row in _read_parquet(ep_parquet).iterrows():
                ep_idx = int(row["episode_index"])
                ch_idx = int(row["meta/episodes/chunk_index"])
                fi_idx = int(row["meta/episodes/file_index"])
                ep_to_video[ep_idx] = self._video_path(ch_idx, fi_idx)

        # global `index` == video frame position for each chunk
        # (verified: index 0-40094 matches video frame count 40095)
        # compute per-video offset = minimum index in that video file
        ep_min_idx = frames_df.groupby("episode_index")["index"].min().to_dict()
        video_offset: dict[Path, int] = {}
        for ep_idx, vpath in ep_to_video.items():
            min_idx = int(ep_min_idx.get(ep_idx, 0))
            if vpath not in video_offset or min_idx < video_offset[vpath]:
                video_offset[vpath] = min_idx

        # Plan: collect (video_path, frame_pos, metadata) for all episodes
        plan: list[dict] = []
        for episode_index, ep_frames in frames_df.groupby("episode_index"):
            episode_index = int(episode_index)
            ep_frames = ep_frames.sort_values("frame_index")
            task_index = int(ep_frames["task_index"].iloc[0])

            vpath = ep_to_video.get(episode_index)
            if vpath is None:
                continue
            offset = video_offset.get(vpath, 0)

            n = len(ep_frames)
            if self.frames_per_episode == 1:
                picks = [n // 2]
            else:
                picks = list(np.linspace(0, n - 1, self.frames_per_episode, dtype=int))

            for pick in picks:
                row = ep_frames.iloc[pick]
                plan.append({
                    "video_path": vpath,
                    "frame_pos": int(row["index"]) - offset,
                    "task_index": task_index,
                    "instruction": tasks.get(task_index, f"task_{task_index}"),
                    "episode_index": episode_index,
                })

        # Extract frames grouped by video file (one open per file)
        from collections import defaultdict
        by_video: dict[Path, list] = defaultdict(list)
        for item in plan:
            by_video[item["video_path"]].append(item)

        samples = []
        for vpath, items in by_video.items():
            positions = [it["frame_pos"] for it in items]
            print(f"  Extracting {len(positions)} frames from {vpath.name}...")
            try:
                frames = self._extract_frames_pyav(vpath, positions)
            except Exception as e:
                print(f"  ERROR opening {vpath.name}: {e}")
                continue

            for item in items:
                arr = frames.get(item["frame_pos"])
                if arr is None:
                    print(f"  Warning: frame {item['frame_pos']} not decoded")
                    continue
                samples.append({
                    "image": self._to_pil(arr),
                    "instruction": item["instruction"],
                    "label": item["task_index"],
                    "scene_id": f"episode_{item['episode_index']}",
                    "variant": f"task_{item['task_index']}",
                })

        n_episodes = len(set(it["episode_index"] for it in plan))
        print(f"Loaded {len(samples)} samples from {n_episodes} episodes")
        if samples:
            print(f"  Labels: {sorted(set(s['label'] for s in samples))}")
            print(f"  Example: '{samples[0]['instruction']}'")
        else:
            print("  WARNING: 0 samples — check dataset path and decoder")
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
