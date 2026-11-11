# Cup perception (YOLO + wrist RGB-D) and camera-driven pick-and-place

Finds the tabletop cup with a fine-tuned YOLOv8n on the `cup_wrist_rgbd` camera,
turns the 2D box + depth into the cup centre in `base_link`, and drives
`scripts/automate_cup.py --cup-source vision`: look, detect, pick, lift, put the
cup back where it was found, release. Simulation only.

## What YOLO is used for

One job: find the cup's 2D box in the wrist RGB image. Everything else is geometry.

1. YOLO box in the RGB frame.
2. Depth pixels inside the box; keep the cup's front surface (drop the table behind it).
3. Back-project with `camera_info`, transform to `base_link` with TF (`/a200_0000/tf`).
4. Push the visible surface one cup radius (3.5 cm) back to the axis; z = top rim - half height.
5. Publish `/cup_detector/pose`. `automate_cup.py` takes the median of 10 samples at the
   look pose and uses it for the MoveIt cup object, the grasp, and the place target.

Gazebo truth is still used for the simulator safety checks (cup moved / tipped) and for
scoring, never to drive the pick.

## Files

| File | Python | What |
|---|---|---|
| `collect_cup_dataset.py` | system `python3` | Teleports the cup around the table, auto-labels boxes by projecting the known cylinder (TF + camera_info), rejects occluded frames via depth. Writes `datasets/cup/` (YOLO format + `meta.jsonl`). |
| `train_cup_yolo.py` | `/opt/yolo_venv/bin/python3` | Fine-tunes `yolov8n.pt` -> `models/cup_yolov8n.pt`; curves in `runs/cup/`. `--threads` caps CPU use. |
| `cup_detector.py` | `/opt/yolo_venv/bin/python3` | ROS node: `/cup_detector/pose` (PoseStamped, base_link), `/cup_detector/detections`, `/cup_detector/image` (annotated). CPU, `--max-rate 3`, `--threads 2`, `--imgsz 416`, `--sync-slop 0.12` (the camera bridge drops frames in long sessions; exact RGB/depth pairs fell to 0.15 Hz). |
| `eval_cup_detector.py` | system `python3` | 3D accuracy vs Gazebo truth at random table spots (detector must be running). |

`/opt/yolo_venv` is a venv with `--system-site-packages` (rclpy, cv_bridge, tf2 from
ROS), pinned in `.devcontainer/yolo-requirements.txt` and rebuilt by Dockerfile layer 4b.
numpy stays at 1.26.4 for cv_bridge.

## Workflow (sim running via `runRobot.sh`)

```bash
cd ~/mobile_manipulator/perception
# 0. arm in the look pose (config/saved_states/cup_look_pose.json): whole table in "Wrist RGB"
python3 collect_cup_dataset.py --count 600              # resumable; re-run adds more
/opt/yolo_venv/bin/python3 train_cup_yolo.py --epochs 30 --imgsz 416
/opt/yolo_venv/bin/python3 cup_detector.py &            # keep running
python3 eval_cup_detector.py --trials 20

cd ../scripts
python3 automate_cup.py --cup-source vision                       # plan only
python3 automate_cup.py --cup-source vision --execute --release   # one full cycle
python3 run_pick_place_cycles.py --cycles 10 [--random]           # scored runs
python3 automate_cup.py --reset                                   # recover: cup back to start
```

## Pick pipeline (vision mode, `--execute --release`)

1. Refuse to move if the cup is still held or not upright on the table.
2. If the open palm is within 15 cm of the cup (after a release), retreat straight back.
3. Move to the look pose; median of 10 detections (refuse if spread > 10 mm).
4. Move the MoveIt `target_cup` to the detected pose.
5. **Grasp search**: rotate the taught grasp about the cup axis (-60..+60 deg) and try
   several IK postures; keep the one with a full straight lift and most joint-limit margin.
6. OMPL to pregrasp (retry x3), then a **straight Cartesian approach** along cup->palm.
7. Gazebo weld (`/cup_grasp/attach`) **first**, then close Dex3, then sync the attached cup
   into MoveIt. Closing first pushed the cup ~6 mm and tilted it ~7 deg before the weld
   froze it, so every place landed 1-2 cm short (`--close-before-attach` = old order).
8. `place_cup_hybrid.py`: **straight lift** (orientation locked; partial lift accepted,
   OMPL covers the rest), transfer, straight descent to release z = 0.83 m.
9. `release_cup_safe.py`: open, detach, cup back into the MoveIt world.

After every stage the arm must end within 0.01 rad of the plan; if it stopped short,
the final point is re-sent once (`settle_to`) before the check fails.

## Results

- Detector: precision 0.999, recall 1.000, mAP50 0.995 on 80 held-out frames.
  3D accuracy: 20/20 spots, mean error 3.2 mm, max 4.0 mm vs Gazebo truth.
- 2026-11-07, close-then-weld grasp, start spot (`logs/cycles_20261107_202200`): 10/10,
  placement error 2.0-23.4 mm (mean 13.5); every place ~1-2 cm short (cup tilted ~7 deg
  in the hand by the closing fingers before the weld).
- 2026-11-08, weld-then-close grasp (current default):
  - Start spot (`logs/cycles_20261108_053726`): 5/6 cycles at **0.0 mm** placement
    error, 0 deg tilt, ~290 s each. Cycle 6: the straight lift stalled, MoveIt cancelled
    it (controller exceeded the 61 s bound) and the cup ended 64 cm away tipped 115 deg.
    Cycle 7: the new guard saw the cup off the table and stopped the run (correct).
  - Random spots (`logs/cycles_20261108_060904`): 6/10, all successes at 0.0 mm.
    Failures: 1 = arm left off-plan by the start-spot crash (look-pose start check);
    3 and 8 (x = 1.00, 0.98) = no OMPL transfer for the rest of the rise, nothing moved;
    6 (x = 1.06) = no reachable grasp. After each failure the runner released the cup
    and carried on, so the held-cup guard works.

- 2026-11-08, fresh sim with `GZ_IP=127.0.0.1` and MoveIt execution scaling 5.0
  (`logs/cycles_20261108_081858`): **10/10 at the start spot, 0.0 mm, 0 deg**, ~290 s per
  cycle; slowest stage 1.25x its planned time; zero gz-transport errors.

## Root causes found 2026-11-08

- **The lift "stall" was MoveIt's watchdog.** It times executions in wall time with
  `allowed_execution_duration_scaling` 2.0. Gazebo dipped to ~0.5x real time, a 30 s lift
  needed 61 s, and MoveIt cancelled it halfway with the cup welded. Now 5.0
  (`move_group_gazebo.launch.py`); the 0.01 rad end checks still catch a real stall.
- **gz-transport was multicasting on the Wi-Fi interface** (host networking): the old
  session's gazebo.log had 1.17M "Network is unreachable" lines, `gz model` reads flaked,
  and finally every `gz` call timed out. `GZ_IP=127.0.0.1` keeps it on loopback.
- Fingers after the weld-first close are re-commanded to their measured contact pose
  (`relax_fingers_to_contact`, gap ~0.05 rad); harmless, but it was not the stall cause.

## Open issues

1. **Far spots (x >= 0.98)** reach the arm's limits: a partial straight lift and then no
   constrained OMPL transfer. A base move or a lower clearance height would help.
   Random spots have not been rerun on the fixed sim.
2. `run_pick_place_cycles.py` stops when the cup is off the table; recovery is still
   manual (`automate_cup.py --reset`).
3. RViz MotionPlanning segfault: still waiting for a gdb backtrace.

## Limits

- Trained on one look pose and one blue cylinder from sim images: another viewpoint,
  cup or a real camera needs new data. In this world a colour mask would also work.
- The lift relies on the Gazebo DetachableJoint weld, not finger friction.
- The base is fixed; random spots far from the taught grasp may have no plan.
