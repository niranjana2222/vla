"""
Loads episodes from the LangGap HuggingFace dataset (LeRobot format).

Expected directory layout (after huggingface-cli download):
    langgap_hf/
        meta/tasks.parquet                          task_index -> instruction
        data/chunk-000/file-000.parquet             per-frame: index, episode_index,
                                                    frame_index, task_index, ...
        videos/observation.images.image/
            chunk-000/file-000.mp4                  AV1-encoded, all episodes concatenated

The global `index` column = frame position in the video file (0-based).
Videos are AV1-encoded; requires decord or pyav for software decoding (cv2 fails).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _extract_frame_decord(video_path: Path, frame_pos: int) -> np.ndarray:
    import decord
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    return vr[frame_pos].asnumpy()


def _extract_frame_pyav(video_path: Path, frame_pos: int) -> np.ndarray:
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        # Seek to the frame using pts
        stream.codec_context.skip_frame = "NONKEY"
        avg_fps = float(stream.average_rate)
        target_pts = int(frame_pos / avg_fps / stream.time_base)
        container.seek(target_pts, stream=stream)
        stream.codec_context.skip_frame = "DEFAULT"
        for pkt_frame in container.decode(stream):
            if pkt_frame.pts is not None and pkt_frame.index >= frame_pos:
                return pkt_frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Frame {frame_pos} not found in {video_path}")


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
        self._decoder = self._find_decoder()

    def _find_decoder(self) -> str:
        try:
            import decord  # noqa: F401
            return "decord"
        except ImportError:
            pass
        try:
            import av  # noqa: F401
            return "pyav"
        except ImportError:
            pass
        raise ImportError(
            "AV1 video decoding requires decord or pyav.\n"
            "Install with:  pip install decord\n"
            "          or:  pip install av"
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

    def _extract_frame(self, video_path: Path, frame_pos: int) -> Image.Image:
        if self._decoder == "decord":
            arr = _extract_frame_decord(video_path, frame_pos)
        else:
            arr = _extract_frame_pyav(video_path, frame_pos)
        img = Image.fromarray(arr).convert("RGB")
        return img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def load(self) -> list[dict]:
        """
        Returns list of {image, instruction, label (=task_index), scene_id, variant}.
        """
        tasks = self._load_tasks()
        frames_df = self._load_frames_df()

        if self.scene_ids is not None:
            frames_df = frames_df[frames_df["task_index"].isin(self.scene_ids)]

        # Build episode -> video file mapping from episodes parquet
        ep_to_video: dict[int, Path] = {}
        ep_to_offset: dict[int, int] = {}
        for ep_parquet in sorted((self.root / "meta" / "episodes").rglob("*.parquet")):
            ep_df = _read_parquet(ep_parquet)
            for _, row in ep_df.iterrows():
                ep_idx = int(row["episode_index"])
                ch_idx = int(row["meta/episodes/chunk_index"])
                fi_idx = int(row["meta/episodes/file_index"])
                ep_to_video[ep_idx] = self._video_path(ch_idx, fi_idx)

        # The global `index` column maps directly to video frame position
        # (verified: index range 0-40094 matches video frame count 40095)
        # For multi-chunk datasets the offset per chunk would be needed;
        # here all data is in chunk-000 starting at index 0.
        #
        # Compute per-video start offset defensively:
        ep_min_index = frames_df.groupby("episode_index")["index"].min()
        for ep_idx, video_path in ep_to_video.items():
            if ep_idx in ep_min_index:
                ep_to_offset[ep_idx] = int(ep_min_index[ep_idx])

        # Compute per-video-file minimum index (= position offset in that file)
        video_start: dict[Path, int] = {}
        for ep_idx, video_path in ep_to_video.items():
            if ep_idx in ep_to_offset:
                cur = ep_to_offset[ep_idx]
                if video_path not in video_start or cur < video_start[video_path]:
                    video_start[video_path] = cur

        samples = []
        grouped = list(frames_df.groupby("episode_index"))
        n_episodes = len(grouped)

        for episode_index, ep_frames in grouped:
            episode_index = int(episode_index)
            ep_frames = ep_frames.sort_values("frame_index")
            task_index = int(ep_frames["task_index"].iloc[0])
            instruction = tasks.get(task_index, f"task_{task_index}")

            video_path = ep_to_video.get(episode_index)
            if video_path is None:
                continue
            offset = video_start.get(video_path, 0)

            n = len(ep_frames)
            pick_indices = (
                [n // 2]
                if self.frames_per_episode == 1
                else list(np.linspace(0, n - 1, self.frames_per_episode, dtype=int))
            )

            for pick in pick_indices:
                row = ep_frames.iloc[pick]
                frame_pos = int(row["index"]) - offset
                try:
                    image = self._extract_frame(video_path, frame_pos)
                except Exception as e:
                    print(f"  Warning: episode {episode_index} frame {frame_pos}: {e}")
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
