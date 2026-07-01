"""Render visualization artifacts from a Demonstration run directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


INSTALL_HINT = "pip install imageio pillow matplotlib"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def evidence_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("evidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_success_frame_row(row: dict[str, Any]) -> bool:
    return bool(row.get("success")) or evidence_value(row) >= 1.0


def resolve_frame_path(raw_path: Any, demo_dir: Path) -> Path | None:
    if not raw_path:
        return None
    frame_path = Path(str(raw_path))
    if frame_path.exists():
        return frame_path
    candidate = demo_dir / "rgb_frames" / frame_path.name
    if candidate.exists():
        return candidate
    return None


def save_success_target_frame(
    step_logs: list[dict[str, Any]],
    demo_dir: Path,
) -> Path | None:
    for row in step_logs:
        if not is_success_frame_row(row) or not row.get("frame_path"):
            continue
        source_path = resolve_frame_path(row.get("frame_path"), demo_dir)
        if source_path is None:
            continue

        if row.get("target_bbox_2d") is None:
            print("WARNING: success frame found but target bbox is missing.")

        output_path = demo_dir / "success_target_frame.png"
        shutil.copy2(source_path, output_path)
        print(f"saved: {output_path}")
        return output_path
    return None


def decision_rows(step_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen_decisions = set()
    for row in step_logs:
        if row.get("action") != "Inspect":
            continue
        decision_step = row.get("decision_step")
        if decision_step in seen_decisions:
            continue
        seen_decisions.add(decision_step)
        rows.append(
            {
                "decision_step": decision_step,
                "selected_instance_alias": row.get("selected_instance_alias"),
                "selected_instance_category": row.get(
                    "selected_instance_category"
                ),
                "coverage": row.get("coverage"),
                "evidence": row.get("evidence"),
                "inspect_count": row.get("inspect_count"),
                "reliability_before": row.get("reliability_before"),
                "reliability_after": row.get(
                    "reliability_after", row.get("reliability")
                ),
                "spf_triggered": row.get("spf_triggered"),
                "success": row.get("success"),
            }
        )
    return rows


def save_decision_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "decision_step",
        "selected_instance_alias",
        "selected_instance_category",
        "coverage",
        "evidence",
        "inspect_count",
        "reliability_before",
        "reliability_after",
        "spf_triggered",
        "success",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_selected_sequence(
    summary: dict[str, Any],
    decision_table: list[dict[str, Any]],
    output_path: Path,
) -> None:
    aliases = [
        str(row.get("selected_instance_alias"))
        for row in decision_table
        if row.get("selected_instance_alias")
    ]
    if not aliases:
        aliases = [str(item) for item in summary.get("selected_sequence", [])]
    output_path.write_text(" -> ".join(aliases), encoding="utf-8")


def plot_reliability_curve(
    summary: dict[str, Any],
    step_logs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(f"Missing plotting dependency. Run: {INSTALL_HINT}") from error

    wrong_id = summary.get("wrong_instance_id")
    true_id = summary.get("true_support_instance_id")
    series: dict[str, list[tuple[int, float]]] = {}

    for row in step_logs:
        decision_step = row.get("decision_step")
        if decision_step is None:
            continue

        for candidate in row.get("top_candidate_scores", []) or []:
            instance_id = candidate.get("instance_id")
            reliability = candidate.get("reliability")
            if instance_id in {wrong_id, true_id} and reliability is not None:
                series.setdefault(str(instance_id), []).append(
                    (int(decision_step), float(reliability))
                )

        selected_id = row.get("selected_instance_id")
        reliability = row.get("reliability_after", row.get("reliability"))
        if selected_id and reliability is not None:
            series.setdefault(str(selected_id), []).append(
                (int(decision_step), float(reliability))
            )

    plt.figure(figsize=(7, 4))
    for instance_id, points in series.items():
        if instance_id not in {wrong_id, true_id} and (wrong_id or true_id):
            continue
        points = sorted(set(points))
        if not points:
            continue
        xs, ys = zip(*points)
        label = instance_id
        if instance_id == wrong_id:
            label = "wrong instance"
        elif instance_id == true_id:
            label = "true support"
        plt.plot(xs, ys, marker="o", label=label)

    if not plt.gca().lines:
        selected_points = []
        for row in step_logs:
            reliability = row.get("reliability_after", row.get("reliability"))
            decision_step = row.get("decision_step")
            if reliability is not None and decision_step is not None:
                selected_points.append((int(decision_step), float(reliability)))
        if selected_points:
            xs, ys = zip(*selected_points)
            plt.plot(xs, ys, marker="o", label="selected instance")

    plt.xlabel("decision_step")
    plt.ylabel("reliability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def overlay_text(
    image_path: Path,
    lines: list[str],
    max_height: int = 120,
) -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(f"Missing image dependency. Run: {INSTALL_HINT}") from error

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    padding = 10
    line_height = 20
    box_height = min(max_height, padding * 2 + line_height * len(lines))
    draw.rectangle((0, 0, 430, box_height), fill=(0, 0, 0, 155))
    y = padding
    for line in lines:
        if y + line_height > box_height:
            break
        draw.text((padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height
    return image


def frame_paths_from_demo(demo_dir: Path) -> list[Path]:
    frame_paths = sorted((demo_dir / "rgb_frames").glob("*.png"))
    if not frame_paths:
        raise RuntimeError(
            "No RGB frames found. Please rerun Demonstration/run_demo_ours.sh "
            "with --save-rgb-frames."
        )
    return frame_paths


def rows_for_frames(
    frame_paths: list[Path], step_logs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows_by_name = {
        Path(row["frame_path"]).name: row
        for row in step_logs
        if row.get("frame_path")
    }
    rows = []
    frame_rows = [row for row in step_logs if row.get("frame_path")]
    for index, frame_path in enumerate(frame_paths):
        rows.append(
            rows_by_name.get(
                frame_path.name,
                frame_rows[min(index, len(frame_rows) - 1)] if frame_rows else {},
            )
        )
    return rows


def save_action_gif(
    demo_dir: Path,
    frame_paths: list[Path],
    frame_rows: list[dict[str, Any]],
    max_frames: int,
) -> tuple[Path, int]:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(f"Missing gif dependency. Run: {INSTALL_HINT}") from error

    gif_path = demo_dir / "ours_navigation_actions.gif"
    stride = max(1, len(frame_paths) // max_frames)
    selected_frame_paths = frame_paths[::stride][:max_frames]
    frames = []
    for frame_index, frame_path in enumerate(selected_frame_paths):
        row = frame_rows[min(frame_index * stride, len(frame_rows) - 1)] if frame_rows else {}
        frames.append(
            overlay_text(
                frame_path,
                [
                    f"action: {fmt(row.get('action'))}",
                    f"decision_step: {fmt(row.get('decision_step'))}",
                    f"selected: {fmt(row.get('selected_instance_alias'))}",
                ],
            )
        )
    imageio.mimsave(gif_path, frames, fps=4)
    if not gif_path.exists():
        raise RuntimeError(f"GIF was not created: {gif_path}")
    return gif_path, len(frames)


def decision_keyframe_rows(
    step_logs: list[dict[str, Any]],
    frame_paths: list[Path],
    success_frame_path: Path | None = None,
) -> list[dict[str, Any]]:
    last_frame_by_decision: dict[Any, str] = {}
    for row in step_logs:
        decision_step = row.get("decision_step")
        if decision_step is not None and row.get("frame_path"):
            last_frame_by_decision[decision_step] = row["frame_path"]

    update_rows: dict[Any, dict[str, Any]] = {}
    update_index = 0
    for row in step_logs:
        if row.get("phase") != "decision_update":
            continue
        decision_step = row.get("decision_step")
        if decision_step is None:
            continue
        row = dict(row)
        representative_frame_path = (
            row.get("representative_frame_path")
            or last_frame_by_decision.get(decision_step)
        )
        if success_frame_path is not None and is_success_frame_row(row):
            representative_frame_path = str(success_frame_path)
        if not representative_frame_path and frame_paths:
            representative_frame_path = str(
                frame_paths[min(update_index, len(frame_paths) - 1)]
            )
        update_index += 1
        if not representative_frame_path:
            continue
        row["frame_path"] = representative_frame_path
        update_rows[decision_step] = row
    if update_rows:
        return [update_rows[key] for key in sorted(update_rows)]

    by_decision: dict[Any, dict[str, Any]] = {}
    for row in step_logs:
        decision_step = row.get("decision_step")
        if decision_step is None or not row.get("frame_path"):
            continue
        if success_frame_path is not None and is_success_frame_row(row):
            row = dict(row)
            row["frame_path"] = str(success_frame_path)
        by_decision[decision_step] = row
    if by_decision:
        return [by_decision[key] for key in sorted(by_decision)]

    fallback: dict[Any, dict[str, Any]] = {}
    for index, row in enumerate(step_logs):
        decision_step = row.get("decision_step")
        if decision_step is None or not frame_paths:
            continue
        row = dict(row)
        row["frame_path"] = str(frame_paths[min(index, len(frame_paths) - 1)])
        fallback[decision_step] = row
    return [fallback[key] for key in sorted(fallback)]


def save_decision_outputs(
    demo_dir: Path,
    decision_frame_rows: list[dict[str, Any]],
    max_frames: int,
) -> tuple[Path, int]:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(f"Missing gif dependency. Run: {INSTALL_HINT}") from error

    keyframe_dir = demo_dir / "decision_keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, row in enumerate(decision_frame_rows):
        raw_frame_path = row.get("representative_frame_path") or row.get("frame_path")
        if not raw_frame_path:
            continue
        frame_path = Path(raw_frame_path)
        if not frame_path.exists():
            candidate = demo_dir / "rgb_frames" / frame_path.name
            frame_path = candidate
        if not frame_path.exists():
            continue
        image = overlay_text(
            frame_path,
            [
                f"Decision {fmt(row.get('decision_step'))}",
                f"Selected: {fmt(row.get('selected_instance_alias'))}",
                f"Coverage: {fmt(row.get('coverage'))}",
                f"Evidence: {fmt(row.get('evidence'))}",
                f"Reliability: {fmt(row.get('reliability_after', row.get('reliability')))}",
                f"SPF: {fmt(row.get('spf_triggered'))}    Success: {fmt(row.get('success'))}",
            ],
            max_height=120,
        )
        output_path = keyframe_dir / f"decision_{index:03d}.png"
        image.save(output_path)
        frames.append(image)

    gif_path = demo_dir / "ours_decision_demo.gif"
    if frames:
        stride = max(1, len(frames) // max_frames)
        imageio.mimsave(gif_path, frames[::stride][:max_frames], fps=2)
    return gif_path, len(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("Demonstration/output/ours_mug_misleading"),
    )
    parser.add_argument("--max-gif-frames", type=int, default=80)
    args = parser.parse_args()

    demo_dir = args.demo_dir
    summary = read_json(demo_dir / "episode_summary.json")
    step_logs = read_jsonl(demo_dir / "step_log.jsonl")

    rows = decision_rows(step_logs)
    save_decision_table(rows, demo_dir / "decision_table.csv")
    save_selected_sequence(summary, rows, demo_dir / "selected_sequence.txt")
    plot_reliability_curve(summary, step_logs, demo_dir / "reliability_curve.png")
    success_frame_path = save_success_target_frame(step_logs, demo_dir)
    frame_paths = frame_paths_from_demo(demo_dir)
    frame_rows = rows_for_frames(frame_paths, step_logs)
    action_gif_path, action_frame_count = save_action_gif(
        demo_dir,
        frame_paths,
        frame_rows,
        args.max_gif_frames,
    )
    decision_frame_rows = decision_keyframe_rows(
        step_logs,
        frame_paths,
        success_frame_path,
    )
    decision_gif_path, decision_keyframe_count = save_decision_outputs(
        demo_dir,
        decision_frame_rows,
        args.max_gif_frames,
    )

    print(f"saved: {demo_dir / 'decision_table.csv'}")
    print(f"saved: {demo_dir / 'selected_sequence.txt'}")
    print(f"saved: {demo_dir / 'reliability_curve.png'}")
    print(f"saved: {action_gif_path}")
    print(f"saved: {decision_gif_path}")
    print(f"RGB frames found: {len(frame_paths)}")
    print(f"action GIF frames used: {action_frame_count}")
    print(f"decision keyframes saved: {decision_keyframe_count}")


if __name__ == "__main__":
    main()
