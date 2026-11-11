#!/usr/bin/env python3
"""Repeat the vision pick -> lift -> put-back -> release cycle and score it.

Each cycle runs `automate_cup.py --cup-source vision --execute --release` unchanged
and records Gazebo truth before/after: success, how far the cup ended up from
where it was picked, and its tilt. cup_detector.py must be running.

    python3 run_pick_place_cycles.py --cycles 10              # cup wherever it is (start pose)
    python3 run_pick_place_cycles.py --cycles 10 --random     # random table spot each cycle

--random first moves the arm to the look pose, then teleports the cup (keeps the
hand clear of the teleport). Results: logs/cycles_<time>/results.jsonl + per-cycle logs.

The Gazebo weld state cannot be queried, so after a failed cycle the cup may still be
welded to the palm even when it sits at table height and MoveIt shows it detached.
Before the next arm motion the runner therefore opens the hand, publishes a detach,
and continues only if the cup is upright on the table and MoveIt has it in the world.
Stops early after --max-failures consecutive failures. Simulation only.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'perception'))
import automate_cup as ac          # noqa: E402
import place_cup as pc             # noqa: E402
import collect_cup_dataset as cc   # noqa: E402  (table bounds, set_cup_pose)


def tilt_deg(p):
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(p[3]) * math.cos(p[4])))))


ROBOT_MODEL = 'a200_0000/robot'
HAND_ACTION = '/a200_0000/dex3_joint_trajectory_controller/follow_joint_trajectory'


class CupNotFree(RuntimeError):
    """The cup may be held or is off the table: the arm must not move."""


def with_node(name, fn):
    rclpy.init()
    node = Node(name)
    try:
        return fn(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def require_cup_free(node):
    """Raise unless MoveIt has the cup in the world and Gazebo has it upright on the table."""
    try:
        scene = pc.get_scene(node, node.create_client(GetPlanningScene, '/get_planning_scene'))
        ac.require_world_cup(scene)        # MoveIt still has it attached
        ac.check_table_cup(pc.gazebo_pose('cup'))
    except RuntimeError as exc:
        raise CupNotFree(str(exc)) from exc


def release_after_failure(node):
    """A failed cycle can leave the cup welded (seen: sync failed after the attach)."""
    print('Previous cycle failed: opening the hand and detaching before any arm motion')
    try:
        ac.check_table_cup(pc.gazebo_pose('cup'))   # in the air: don't drop it, stop
        pc.open_dex3(node, ActionClient(node, FollowJointTrajectory, HAND_ACTION))
        pc.gazebo_detach()
        time.sleep(2.0)
        cup = pc.gazebo_pose('cup')
        ac.check_table_cup(cup)
        ac.update_world_scene(node, cup, pc.gazebo_pose(ROBOT_MODEL))
    except RuntimeError as exc:
        raise CupNotFree(str(exc)) from exc
    require_cup_free(node)


def move_to_look(node, planning_time=15.0):
    require_cup_free(node)
    ac.move_to_look_pose(node, argparse.Namespace(look_pose=ac.DEFAULT_LOOK_POSE,
                                                  planning_time=planning_time))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cycles', type=int, default=10)
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--x-range', type=float, nargs=2, default=(0.85, 1.10),
                    help='random cup X range (world); taught grasp reach is limited')
    ap.add_argument('--y-range', type=float, nargs=2, default=(-0.15, 0.15))
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--max-failures', type=int, default=3)
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='extra args passed to automate_cup.py')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = HERE.parent / 'logs' / f'cycles_{stamp}'
    logdir.mkdir(parents=True, exist_ok=True)
    results_path = logdir / 'results.jsonl'
    ok = fail_streak = 0
    needs_release = False
    errors = []
    for i in range(1, args.cycles + 1):
        record = {'cycle': i}
        try:
            if needs_release:
                with_node('cycle_release', release_after_failure)
                needs_release = False
            if args.random:
                with_node('cycle_look_pose', move_to_look)
                x, y = rng.uniform(*args.x_range), rng.uniform(*args.y_range)
                cc.set_cup_pose(x, y, cc.CUP_Z, yaw=rng.uniform(-math.pi, math.pi))
                time.sleep(1.5)
            before = pc.gazebo_pose('cup')
            record['before'] = [round(v, 4) for v in before[:3]]
            log = logdir / f'cycle_{i:02d}.log'
            t0 = time.time()
            with log.open('w') as fh:
                r = subprocess.run([sys.executable, str(HERE / 'automate_cup.py'),
                                    '--cup-source', 'vision', '--execute', '--release',
                                    *args.extra], stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=1500)
            record['seconds'] = round(time.time() - t0, 1)
            time.sleep(1.0)
            after = pc.gazebo_pose('cup')
            record['after'] = [round(v, 4) for v in after[:3]]
            record['place_error_mm'] = round(1000 * math.dist(before[:2], after[:2]), 1)
            record['tilt_deg'] = round(tilt_deg(after), 1)
            on_table = abs(after[2] - cc.CUP_Z) < 0.02 and record['tilt_deg'] < 10
            record['success'] = r.returncode == 0 and on_table
            if r.returncode != 0:
                tail = [l for l in log.read_text().splitlines() if l.startswith('ERROR')]
                record['error'] = tail[-1] if tail else f'exit {r.returncode}'
            elif not on_table:
                record['error'] = 'cup not upright on the table after release'
        except CupNotFree as exc:
            record['success'] = False
            record['cup_not_free'] = True
            record['error'] = f'cup not free: {exc}'
        except Exception as exc:  # noqa: BLE001
            record['success'] = False
            record['error'] = f'{type(exc).__name__}: {exc}'
        with results_path.open('a') as fh:
            fh.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if record.get('cup_not_free'):
            print('Stopping: cup is not free and upright on the table. '
                  'Recover with: automate_cup.py --reset')
            i_stop = True
        else:
            i_stop = False
            needs_release = not record['success']
        if record['success']:
            ok += 1
            fail_streak = 0
            errors.append(record['place_error_mm'])
        else:
            fail_streak += 1
            if fail_streak >= args.max_failures:
                print(f'Stopping: {fail_streak} consecutive failures')
                break
        if i_stop:
            break
    done = i
    summary = (f'{ok}/{done} cycles succeeded'
               + (f'; placement error mean {sum(errors)/len(errors):.1f} mm, '
                  f'max {max(errors):.1f} mm' if errors else '')
               + f'. Results: {results_path}')
    print(summary)
    (logdir / 'summary.txt').write_text(summary + '\n')


if __name__ == '__main__':
    main()
