#!/usr/bin/env python3
"""Pick a cup at a given map location, carry it, and place it at a destination.

Gazebo simulation only (uses Gazebo ground truth for checks and AMCL re-seeding).

Flow:
  1. Choose the stored base station nearest the cup's map XY.
  2. Arm to the look pose; Nav2 to an approach pose 0.40 m behind the station; creep
     onto the station along its heading.
  3. Detect the cup with YOLO, check it against the requested location, grasp it.
  4. Lift the cup and pull it over the base (straight-line move, palm locked).
  5. Drive to the station nearest the drop XY the same way.
  6. Place the cup with a straight-line move, release it, back the hand away.

Stations live in config/saved_states/base_stations.json (map frame = Gazebo world).
  python3 station_pick_and_place.py --list-stations
  python3 station_pick_and_place.py --save-station A      # park the robot there first
  python3 station_pick_and_place.py --delete-station A

Dry run (station choice only, no ROS needed):
  python3 station_pick_and_place.py --cup-x 0.95 --cup-y 0.0 --drop-x 1.3 --drop-y -0.1
Full cycle (runRobot.sh must be running; it also starts the cup detector):
  python3 station_pick_and_place.py --cup-x 0.95 --cup-y 0.0 --drop-x 1.3 --drop-y -0.1 --execute
Pick only (stops with the cup held in the carry pose), then place it later:
  python3 station_pick_and_place.py --pick-only --cup-x 0.95 --cup-y 0.0 --execute
  python3 station_pick_and_place.py --place-only --drop-x 1.3 --drop-y -0.1 --execute
--place-only also finishes a run that stopped with the cup in the hand.

The run stops on uncertain AMCL, a failed Nav2 goal, a missed detection, an unconfirmed
grasp, a bad carry pose or a failed placement. It never detaches the cup after an error.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
STATION_FILE = ROOT / 'config' / 'saved_states' / 'base_stations.json'

# Used only if base_stations.json is missing: the two original AMCL readings.
DEFAULT_STATIONS = {
    '1': {'x': 0.16312104065772634, 'y': 0.24463882064621267,
          'yaw': 2.0 * math.atan2(0.005351018760333591, 0.9999856831966278)},
    '2': {'x': 1.2863049272411609, 'y': -0.9333676693926033,
          'yaw': 2.0 * math.atan2(0.7121312067087355, 0.7020463976344871)},
}

# Tabletop placement limits in the Gazebo world frame. They match
# place_cup_hybrid.validate_args.
TABLE_X = (0.80, 1.50)
TABLE_Y = (-0.25, 0.25)


# --------------------------------------------------------------------------- #
# Station storage and selection (no ROS)
# --------------------------------------------------------------------------- #
def load_stations():
    if STATION_FILE.exists():
        data = json.loads(STATION_FILE.read_text()).get('stations', {})
        if data:
            return _with_refs(data)
    return _with_refs({k: dict(v) for k, v in DEFAULT_STATIONS.items()})


def _with_refs(stations):
    """Map frame == Gazebo world here (map is generated from the world), so a
    station without a recorded Gazebo pose is checked against its own map pose."""
    for s in stations.values():
        if not s.get('gazebo'):
            s['gazebo'] = [s['x'], s['y'], 0.0, 0.0, 0.0, s['yaw']]
    return stations


def save_stations(stations):
    STATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATION_FILE.write_text(json.dumps({'stations': stations}, indent=2) + '\n')


def nearest_station(stations, x, y, forced=None):
    if forced is not None:
        if forced not in stations:
            raise SystemExit(f'Unknown station {forced!r}; have {sorted(stations)}')
        s = stations[forced]
        return forced, math.hypot(x - s['x'], y - s['y'])
    name = min(stations, key=lambda n: math.hypot(x - stations[n]['x'],
                                                  y - stations[n]['y']))
    s = stations[name]
    return name, math.hypot(x - s['x'], y - s['y'])


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


_T0 = time.monotonic()


def say(label, message):
    # Elapsed wall time since the script started, so a log shows where a cycle spends it.
    print(f'\n[{label}] {message}   (+{time.monotonic() - _T0:.0f} s)', flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cup-x', type=float, help='Approximate cup X (map)')
    ap.add_argument('--cup-y', type=float, help='Approximate cup Y (map)')
    ap.add_argument('--drop-x', type=float, help='Drop X (map frame = Gazebo world)')
    ap.add_argument('--drop-y', type=float, help='Drop Y')
    ap.add_argument('--pick-station', help='Override automatic pick station choice')
    ap.add_argument('--drop-station', help='Override automatic drop station choice')
    ap.add_argument('--max-station-dist', type=float, default=1.0,
                    help='Refuse if the nearest station is farther than this from the cup or drop point (m)')
    ap.add_argument('--execute', action='store_true', help='Enable robot motion')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--pick-only', action='store_true',
                      help='Go to the pick station, grasp, lift to the carry pose and stop '
                           'with the cup held (needs --cup-x/--cup-y only)')
    mode.add_argument('--place-only', action='store_true',
                      help='Cup is already held: skip the pick, go to the drop station and '
                           'place (needs --drop-x/--drop-y only)')
    ap.add_argument('--save-station', metavar='NAME',
                    help='Record the current localized base pose as station NAME')
    ap.add_argument('--delete-station', metavar='NAME')
    ap.add_argument('--list-stations', action='store_true')
    # Navigation and localization checks
    ap.add_argument('--nav-timeout', type=float, default=360.0)
    ap.add_argument('--goal-xy-tol', type=float, default=0.10)
    ap.add_argument('--goal-yaw-tol-deg', type=float, default=7.0)
    ap.add_argument('--max-amcl-xy-std', type=float, default=0.20)
    ap.add_argument('--max-amcl-yaw-std-deg', type=float, default=12.0)
    ap.add_argument('--sim-allow-uncertain-amcl', action='store_true')
    # Skid-steer odom drifts ~0.3 m per Nav2 leg on curves; AMCL correcting that moves
    # map->odom by as much, so only gross jumps are treated as a fault.
    ap.add_argument('--max-map-odom-jump', type=float, default=0.60)
    ap.add_argument('--strict-jump', action='store_true',
                    help='Abort (not just warn) when map->odom moves more than the limits')
    ap.add_argument('--no-sim-reseed', action='store_true',
                    help='Do not re-seed AMCL from Gazebo truth after each Nav2 leg')
    ap.add_argument('--max-map-odom-jump-deg', type=float, default=25.0)
    ap.add_argument('--align-attempts', type=int, default=5,
                    help='Slow direct-velocity corrections if Nav2 stops outside tolerance')
    ap.add_argument('--approach-offset', type=float, default=0.40,
                    help='Nav2 stops this far (m) behind the station along its heading, '
                         'outside the table inflation; the rest is a straight slow creep')
    ap.add_argument('--max-align-dist', type=float, default=0.55,
                    help='Refuse to creep further than this (m); bypasses collision monitor')
    ap.add_argument('--align-vx', type=float, default=0.08, help='Max creep speed (m/s)')
    ap.add_argument('--align-wz', type=float, default=0.30, help='Max creep turn rate (rad/s)')
    ap.add_argument('--cup-association-radius', type=float, default=0.50)
    # Manipulation
    ap.add_argument('--carry-cup-x', type=float, default=0.28)
    ap.add_argument('--carry-cup-y', type=float, default=0.0)
    ap.add_argument('--carry-cup-z-world', type=float, default=1.0)
    ap.add_argument('--release-z', type=float, default=0.83)
    ap.add_argument('--clearance-z', type=float, default=1.02)
    return ap, ap.parse_args()


# --------------------------------------------------------------------------- #
# ROS side
# --------------------------------------------------------------------------- #
class Robot:
    """TF, AMCL, Nav2 and MoveIt helpers for the pick-and-place task."""

    def __init__(self, a):
        import rclpy
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from tf2_ros import Buffer, TransformListener
        self.rclpy, self.a = rclpy, a
        rclpy.init(args=['--ros-args', '-r', '/tf:=/a200_0000/tf',
                         '-r', '/tf_static:=/a200_0000/tf_static'])
        self.node = Node('station_pick_and_place',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self.node)
        self.last_amcl = None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.amcl_sub = self.node.create_subscription(
            PoseWithCovarianceStamped, '/a200_0000/amcl_pose',
            lambda m: setattr(self, 'last_amcl', m), qos)

    def close(self):
        self.node.destroy_subscription(self.amcl_sub)
        self.node.destroy_node()
        self.rclpy.try_shutdown()

    # ---- TF / localization --------------------------------------------------
    def tf_xy_yaw(self, target, source, wait=10.0):
        from rclpy.time import Time
        end, last_exc = time.monotonic() + wait, None
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                tr = self.tf_buf.lookup_transform(target, source, Time()).transform
                t, q = tr.translation, tr.rotation
                return (t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                             1 - 2 * (q.y * q.y + q.z * q.z)))
            except Exception as exc:  # noqa: BLE001 - tf2 raises several types
                last_exc = exc
        raise RuntimeError(f'TF {target}<-{source} unavailable: {last_exc}')

    @staticmethod
    def amcl_sigma(msg):
        cov = msg.pose.covariance
        xy = max(math.sqrt(max(0.0, cov[0])), math.sqrt(max(0.0, cov[7])))
        return xy, math.degrees(math.sqrt(max(0.0, cov[35])))

    def converge_amcl(self, rounds=15, min_rounds=0):
        """AMCL only updates after the base moves (update_min_d/min_a), so a
        stationary robot keeps a stale covariance. Force scan updates in place."""
        from std_srvs.srv import Empty
        a = self.a
        client = self.node.create_client(Empty, '/a200_0000/request_nomotion_update')
        if not client.wait_for_service(timeout_sec=5.0):
            print('WARN: request_nomotion_update unavailable; cannot converge in place')
            return
        for i in range(rounds):
            prev = self.last_amcl
            fut = client.call_async(Empty.Request())
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
            end = time.monotonic() + 3.0   # wait for the next published pose
            while self.last_amcl is prev and time.monotonic() < end:
                self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.last_amcl is None:
                continue
            xy, yaw = self.amcl_sigma(self.last_amcl)
            print(f'  nomotion update {i + 1}/{rounds}: xy={xy:.3f} m, yaw={yaw:.1f} deg',
                  flush=True)
            if (i + 1 >= min_rounds and xy <= a.max_amcl_xy_std
                    and yaw <= a.max_amcl_yaw_std_deg):
                return

    def check_amcl(self, timeout=18.0):
        a = self.a
        deadline = time.monotonic() + timeout
        while self.last_amcl is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        if self.last_amcl is not None:
            xy, yaw = self.amcl_sigma(self.last_amcl)
            if xy > a.max_amcl_xy_std or yaw > a.max_amcl_yaw_std_deg:
                print(f'AMCL sigma {xy:.3f} m / {yaw:.1f} deg over limit; '
                      'forcing in-place updates...', flush=True)
                self.converge_amcl()
        deadline = time.monotonic() + timeout
        while self.last_amcl is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        msg = self.last_amcl
        if msg is None:
            raise RuntimeError('No AMCL pose on /a200_0000/amcl_pose')
        now = self.node.get_clock().now().nanoseconds * 1e-9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now > 0 and now - stamp > 5.0:
            # amcl_pose is latched. The robot may be at rest, so check it against live TF.
            mx, my, myaw = self.tf_xy_yaw('map', 'base_link', wait=5)
            p = msg.pose.pose
            ayaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
            if (math.hypot(mx - p.position.x, my - p.position.y) > 0.10
                    or math.degrees(abs(angle_diff(myaw, ayaw))) > 7.0):
                raise RuntimeError('Stale AMCL pose disagrees with map->base_link TF; '
                                   'nudge the robot or force a localization update')
        cov = msg.pose.covariance
        xy = max(math.sqrt(max(0.0, cov[0])), math.sqrt(max(0.0, cov[7])))
        yaw = math.degrees(math.sqrt(max(0.0, cov[35])))
        print(f'AMCL sigma: xy={xy:.3f} m, yaw={yaw:.1f} deg')
        if xy > a.max_amcl_xy_std or yaw > a.max_amcl_yaw_std_deg:
            if not a.sim_allow_uncertain_amcl:
                raise RuntimeError('AMCL uncertainty exceeds limits; converge localization first '
                                   '(or pass --sim-allow-uncertain-amcl under supervision)')
            print('WARNING: continuing with uncertain AMCL (--sim-allow-uncertain-amcl)')
        return msg

    # ---- Navigation ---------------------------------------------------------
    def goto(self, name, st):
        a = self.a
        say('NAV', f'Navigate to station {name} '
                   f'({st["x"]:+.3f}, {st["y"]:+.3f}, {math.degrees(st["yaw"]):+.1f} deg)')
        self.check_amcl()
        base0 = self.tf_xy_yaw('map', 'odom')

        warned = []

        def jump_check():
            now = self.tf_xy_yaw('map', 'odom', wait=2)
            d = math.hypot(now[0] - base0[0], now[1] - base0[1])
            dy = math.degrees(abs(angle_diff(now[2], base0[2])))
            if d > a.max_map_odom_jump or dy > a.max_map_odom_jump_deg:
                if a.strict_jump:
                    raise RuntimeError(f'Localization jumped ({d:.2f} m, {dy:.1f} deg)')
                if not warned:
                    warned.append(1)
                    print(f'WARN: map->odom moved {d:.2f} m / {dy:.1f} deg during this leg '
                          '(odom drift being corrected by AMCL); arrival is verified '
                          'separately', flush=True)

        # Stations sit ~15 cm from the table so the arm can reach: inside the costmap's
        # inscribed/inflated band, where MPPI stalls ("Failed to make progress").
        # Nav2 therefore drives to an approach pose pulled back along the station
        # heading; fine_align() creeps the last part straight in.
        ax = st['x'] - a.approach_offset * math.cos(st['yaw'])
        ay = st['y'] - a.approach_offset * math.sin(st['yaw'])
        x0, y0, yaw0 = self.tf_xy_yaw('map', 'base_link')
        close = (math.hypot(st['x'] - x0, st['y'] - y0) <= a.max_align_dist
                 and abs(angle_diff(yaw0, st['yaw'])) <= math.radians(20))
        if close:
            print('Already near the station and roughly aligned; skipping Nav2.')
        else:
            print(f'Nav2 goal = approach pose ({ax:+.3f}, {ay:+.3f}), '
                  f'{a.approach_offset:.2f} m back from the station')
            self.nav2_to(ax, ay, st['yaw'], jump_check)
        self.check_amcl()
        self.verify_arrival(name, st, lambda: None)
        self.sync_scene()

    def nav2_to(self, x, y, yaw, jump_check):
        from rclpy.action import ActionClient
        from action_msgs.msg import GoalStatus
        from nav2_msgs.action import NavigateToPose
        a, rclpy, node = self.a, self.rclpy, self.node
        client = ActionClient(node, NavigateToPose, '/a200_0000/navigate_to_pose')
        if not client.wait_for_server(timeout_sec=20):
            raise RuntimeError('Nav2 NavigateToPose action not available')
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        req = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, req, timeout_sec=35)
        if not req.done() or req.result() is None or not req.result().accepted:
            raise RuntimeError('Nav2 goal not accepted')
        handle = req.result()
        fut = handle.get_result_async()
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(node, timeout_sec=0.25)
            try:
                jump_check()
                if time.monotonic() - t0 > a.nav_timeout:
                    raise RuntimeError('Nav2 timed out')
            except RuntimeError:
                rclpy.spin_until_future_complete(node, handle.cancel_goal_async(),
                                                 timeout_sec=12)
                raise
        if fut.result() is None or fut.result().status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError('Nav2 failed/cancelled; no manipulation attempted')

    def settle_amcl(self, max_rounds=25, still_m=0.005, still_n=3):
        """In-place AMCL scan updates until its pose stops moving. Short creeps are
        below update_min_d, and after skid-steer slip AMCL needs ~10-25 updates to
        pull back (measured: 5.6 -> 2.8 cm), so a fixed 3 rounds was not enough."""
        from std_srvs.srv import Empty
        client = self.node.create_client(Empty, '/a200_0000/request_nomotion_update')
        if not client.wait_for_service(timeout_sec=5.0):
            print('WARN: request_nomotion_update unavailable')
            return
        prev, still = None, 0
        for i in range(max_rounds):
            fut = client.call_async(Empty.Request())
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
            x, y, _ = self.tf_xy_yaw('map', 'base_link', wait=2)
            if prev is not None:
                still = still + 1 if math.hypot(x - prev[0], y - prev[1]) < still_m else 0
                if still >= still_n:
                    break
            prev = (x, y)
        print(f'  AMCL settled after {i + 1} scan updates', flush=True)

    def verify_arrival(self, name, st, jump_check):
        import place_cup as pc
        a = self.a
        ref = st.get('gazebo')
        aim = 0.5 * a.goal_xy_tol   # creep target: leaves margin for ~3-5 cm AMCL error

        def measure():
            d, dy = self.station_error(st)
            msg = f'Arrival error: AMCL {d:.3f} m / {dy:.1f} deg'
            ok_truth = True
            if ref:
                g = pc.gazebo_pose('a200_0000/robot')
                gd = math.hypot(g[0] - ref[0], g[1] - ref[1])
                gy = math.degrees(abs(angle_diff(g[5], ref[5])))
                ok_truth = gd <= a.goal_xy_tol and gy <= a.goal_yaw_tol_deg
                msg += f' | Gazebo truth {gd:.3f} m / {gy:.1f} deg'
            print(msg + f' (tolerance {a.goal_xy_tol:.2f} m / {a.goal_yaw_tol_deg:.0f} deg)',
                  flush=True)
            return d, dy, ok_truth, msg

        if a.no_sim_reseed:
            self.settle_amcl()
        else:
            self.reseed_amcl()      # simulation: start the creep from the true pose
        for attempt in range(a.align_attempts + 1):
            d, dy, ok_truth, msg = measure()
            ok_amcl = d <= a.goal_xy_tol and dy <= a.goal_yaw_tol_deg
            if ok_amcl and not ok_truth:
                # AMCL believes we arrived but we did not. Let it settle on fresh scans
                # first; re-seed from Gazebo (simulation only) if that is not enough.
                say('ALIGN', 'AMCL disagrees with Gazebo truth; settling, then re-seeding')
                self.settle_amcl(max_rounds=25)
                d, dy, ok_truth, msg = measure()
                ok_amcl = d <= a.goal_xy_tol and dy <= a.goal_yaw_tol_deg
                if ok_amcl and not ok_truth:
                    self.reseed_amcl()
                    d, dy, ok_truth, msg = measure()
                    ok_amcl = d <= a.goal_xy_tol and dy <= a.goal_yaw_tol_deg
            if ok_amcl and ok_truth and (d <= aim or attempt == a.align_attempts):
                break
            if attempt == a.align_attempts:
                raise RuntimeError(f'Still outside station tolerance after '
                                   f'{a.align_attempts} fine-alignment attempts ({msg})')
            if d > a.max_align_dist:
                raise RuntimeError(f'Nav2 stopped {d:.2f} m from the station (> '
                                   f'--max-align-dist {a.max_align_dist}); not creeping that '
                                   'far without collision checking')
            say('ALIGN', f'Fine alignment {attempt + 1}/{a.align_attempts}')
            self.fine_align(st)
            if a.no_sim_reseed:
                self.settle_amcl()
                self.check_amcl()
            else:
                self.reseed_amcl()      # also re-checks AMCL
            jump_check()
        else:
            raise RuntimeError('Arrival not verified within the alignment attempts')
        print(f'ARRIVED at station {name}', flush=True)

    def reseed_amcl(self):
        """Gazebo only: reset AMCL to the true pose (seed_amcl_from_gazebo.py)."""
        res = subprocess.run([sys.executable, str(SCRIPTS / 'seed_amcl_from_gazebo.py')],
                             timeout=120)
        if res.returncode != 0:
            raise RuntimeError('seed_amcl_from_gazebo.py could not re-seed AMCL')
        self.last_amcl = None          # wait for the post-seed pose, not the stale one
        self.check_amcl()

    def station_error(self, st):
        """Distance (m) and heading error (deg) from the AMCL pose to station st."""
        p = self.last_amcl.pose.pose
        yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
        return (math.hypot(p.position.x - st['x'], p.position.y - st['y']),
                math.degrees(abs(angle_diff(yaw, st['yaw']))))

    def fine_align(self, st):
        """Creep onto the station with direct, slow velocity commands.

        Nav2 is idle here, so twist_mux passes these through. They bypass the
        collision monitor, which is why the correction distance is capped
        (--max-align-dist) and the speeds are low. Tracks map->base_link TF (odom
        is accurate over these short distances); AMCL re-checks afterwards.
        """
        from geometry_msgs.msg import TwistStamped
        a, rclpy, node = self.a, self.rclpy, self.node
        pub = node.create_publisher(TwistStamped, '/a200_0000/cmd_vel', 10)

        def send(vx, wz):
            m = TwistStamped()
            m.header.stamp = node.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.twist.linear.x, m.twist.angular.z = float(vx), float(wz)
            pub.publish(m)

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        def turn_to(target_yaw, tol=math.radians(1.5), limit=20.0):
            end = time.monotonic() + limit
            while time.monotonic() < end:
                _, _, yaw = self.tf_xy_yaw('map', 'base_link', wait=1)
                e = angle_diff(target_yaw, yaw)
                if abs(e) < tol:
                    return
                w = clamp(1.2 * e, -a.align_wz, a.align_wz)
                send(0.0, math.copysign(max(abs(w), 0.06), w))
                rclpy.spin_once(node, timeout_sec=0.05)
            raise RuntimeError('Fine alignment: rotation timed out')

        # Follow the line through the station along its heading. Never spin toward a
        # bearing: near the table the footprint corners (0.71 m from centre) would
        # sweep into it. Lateral error is steered out while driving.
        h = st['yaw']
        ux, uy = math.cos(h), math.sin(h)
        try:
            _, _, yaw = self.tf_xy_yaw('map', 'base_link')
            if abs(angle_diff(yaw, h)) > math.radians(20):
                raise RuntimeError('Fine alignment: heading is >20 deg off the station; '
                                   'refusing to rotate near the table')
            def errors():
                x, y, yaw = self.tf_xy_yaw('map', 'base_link', wait=1)
                ex, ey = st['x'] - x, st['y'] - y
                return ex * ux + ey * uy, -ex * uy + ey * ux, yaw   # along(+ahead), lat(+left)

            along, lat, _ = errors()
            # A diff-drive base cannot slide sideways. With a large lateral error and
            # little runway, first reverse STRAIGHT (no rotation near the table) to get
            # room, then drive in while steering onto the station line.
            runway = min(0.55, max(0.30, 2.5 * abs(lat)))
            if abs(lat) > 0.03 and along < runway:
                print(f'  lateral {lat:+.3f} m: reversing to {runway:.2f} m runway first')
                end = time.monotonic() + 30.0
                while along < runway:
                    if time.monotonic() > end:
                        raise RuntimeError('Fine alignment: reverse timed out')
                    send(-a.align_vx, 0.0)
                    rclpy.spin_once(node, timeout_sec=0.05)
                    along, lat, _ = errors()
                send(0.0, 0.0)
            end = time.monotonic() + 60.0
            while True:
                along, lat, yaw = errors()
                if abs(along) < 0.01:
                    break
                if time.monotonic() > end:
                    raise RuntimeError('Fine alignment: translation timed out')
                sign = 1.0 if along > 0 else -1.0
                v = clamp(0.8 * abs(along), 0.03, a.align_vx)
                # Bigger corrective heading only while well clear of the table.
                limit = math.radians(35 if abs(along) > 0.25 else 10)
                steer = clamp(math.atan(3.0 * lat), -limit, limit)
                yaw_des = h + sign * steer
                send(sign * v, clamp(1.5 * angle_diff(yaw_des, yaw), -a.align_wz, a.align_wz))
                rclpy.spin_once(node, timeout_sec=0.05)
            send(0.0, 0.0)
            turn_to(h)
        finally:
            for _ in range(5):
                send(0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.02)
            node.destroy_publisher(pub)
        time.sleep(0.5)

    def require_detector(self, timeout=15.0):
        """cup_detector publishes a Detection2DArray for every processed frame,
        even when it sees no cup. Silence means it is not running or not syncing."""
        from vision_msgs.msg import Detection2DArray
        got = []
        sub = self.node.create_subscription(Detection2DArray, '/cup_detector/detections',
                                            lambda m: got.append(m), 10)
        end = time.monotonic() + timeout
        while not got and time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        self.node.destroy_subscription(sub)
        if not got:
            raise RuntimeError(
                f'No /cup_detector/detections in {timeout:.0f} s. Either the detector is '
                'not running, or it was started after the robot stack and got a Fast DDS '
                'participant slot >= 32 (LOCALHOST discovery only probes 0-31), so this '
                'process cannot discover it. Let runRobot.sh start it (default), or free a '
                'slot and restart it. See runRobot.sh "YOLO cup detector".')
        print('cup_detector is alive.')

    # ---- MoveIt -------------------------------------------------------------
    def scene(self, attempts=3):
        from moveit_msgs.srv import GetPlanningScene
        import place_cup as pc
        client = self.node.create_client(GetPlanningScene, '/get_planning_scene')
        if not client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('/get_planning_scene unavailable; check move_group')
        for i in range(1, attempts + 1):
            try:
                return pc.get_scene(self.node, client)
            except RuntimeError as exc:
                if i == attempts:
                    raise RuntimeError(f'Planning scene unavailable: {exc}') from exc
                print(f'WARN: scene request {i}/{attempts} failed; retrying', flush=True)
                time.sleep(1.0)

    def execute_checked(self, traj, label):
        import automate_cup as cup
        import place_cup_hybrid as hybrid
        hybrid.check_joint_match(self.scene().robot_state, traj,
                                 traj.joint_trajectory.points[0], 0.012, label + ' start')
        cup.execute_planned(self.node, traj, label)
        hybrid.check_joint_match(self.scene().robot_state, traj,
                                 traj.joint_trajectory.points[-1], 0.018, label + ' end')

    def cartesian_path(self, cart, fk, start, deltas_base, label, min_fraction=0.99):
        """Straight-line palm path through cumulative base_link translations, palm
        orientation unchanged, collision checking on. Returns a RobotTrajectory."""
        from moveit_msgs.msg import MoveItErrorCodes
        from moveit_msgs.srv import GetCartesianPath
        import place_cup as pc
        import place_cup_hybrid as hybrid
        pose = pc.palm_fk(self.node, fk, start)
        waypoints = []
        for d in deltas_base:
            if math.hypot(*d) < 1e-3:
                continue
            pose = pc.waypoint(pose, d)
            waypoints.append(pose)
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.start_state = pc.normalized_start(start)
        req.start_state.is_diff = False
        req.group_name = pc.GROUP
        req.link_name = pc.PALM
        req.waypoints = waypoints
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.revolute_jump_threshold = 0.20
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = 0.08
        req.max_acceleration_scaling_factor = 0.08
        ans = pc.service(self.node, cart, req, 'GetCartesianPath', timeout=60.0)
        print(f'{label}: fraction={ans.fraction:.3f}, code={ans.error_code.val}')
        if ans.error_code.val != MoveItErrorCodes.SUCCESS or ans.fraction < min_fraction:
            raise RuntimeError(f'{label}: only {ans.fraction:.0%} of the straight path is '
                               'collision-free/reachable')
        traj = pc.unwrap_to_start(ans.solution.joint_trajectory, start)
        if len(traj.points) < 2 or set(traj.joint_names) != set(hybrid.ARM):
            raise RuntimeError(f'{label}: unusable trajectory')
        end = traj.points[-1].time_from_start
        if not (end.sec or end.nanosec):
            raise RuntimeError(f'{label}: zero-duration trajectory')
        for i in range(1, len(traj.points)):
            jump = max(abs(b - a_) for a_, b in zip(traj.points[i - 1].positions,
                                                    traj.points[i].positions))
            if jump > 0.20:
                raise RuntimeError(f'{label}: joint jump {jump:.3f} rad at point {i}')
        ans.solution.joint_trajectory = traj
        return ans.solution

    def sync_scene(self):
        """Re-place the MoveIt table (and the cup, when it stands free) from Gazebo truth.

        add_scene_objects.py places them relative to the robot once at startup and
        MoveIt then keeps them fixed in the odom frame. Odom drifts ~0.6 m per Nav2 leg,
        so after driving the table ends up inside the robot in MoveIt's view and every
        plan fails CheckStartStateCollision (base_link - work_table ...)."""
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        import add_scene_objects as aso
        import place_cup as pc
        base = pc.gazebo_pose('a200_0000/robot')
        table = pc.gazebo_pose('table')
        centre = (table[0], table[1], table[2] + 0.375, *table[3:])
        objs = [aso.make_box('work_table', pc.cup_pose_in_base(centre, base), (0.8, 0.6, 0.75))]
        scene = self.scene()
        held = any(o.object.id == pc.CUP_ID for o in scene.robot_state.attached_collision_objects)
        if not held:
            objs.append(aso.make_cylinder(pc.CUP_ID, pc.cup_pose_in_base(pc.gazebo_pose('cup'), base),
                                          0.12, 0.035))
        diff = PlanningScene()
        diff.is_diff = True
        diff.world.collision_objects = objs
        client = self.node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError('/apply_planning_scene unavailable')
        req = ApplyPlanningScene.Request()
        req.scene = diff
        res = pc.service(self.node, client, req, 'ApplyPlanningScene', timeout=20)
        if not res.success:
            raise RuntimeError('MoveIt refused the planning-scene update')
        print(f'MoveIt scene re-synced from Gazebo (table{"" if held else " + cup"}).', flush=True)

    def place_cartesian(self, wx, wy):
        """Move the held cup in straight lines (palm orientation locked, collision
        checked) to above the drop point, then straight down to the release height.
        OMPL transfers twisted the palm 60-180 deg and were always rejected."""
        from moveit_msgs.srv import GetPositionFK, GetCartesianPath
        import place_cup as pc
        import place_cup_hybrid as hybrid
        a, node = self.a, self.node
        scene = self.scene()
        hybrid.check_setup(scene)
        fk = node.create_client(GetPositionFK, '/compute_fk')
        cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        base = pc.gazebo_pose('a200_0000/robot')
        cup = pc.gazebo_pose('cup')
        qinv = pc.qinv(pc.quat(*base[3:]))
        traj, errs = None, []
        # Lower transfer heights are easier to reach at ~0.9 m; 0.90 still clears the
        # table top (0.75) by 9 cm under the cup.
        for zc in sorted({0.92, 0.96, 0.90, round(a.clearance_z, 2)}, key=lambda z: abs(z - 0.92)):
            if zc < a.release_z + 0.05:
                continue
            deltas = [pc.rotate(qinv, (wx - cup[0], wy - cup[1], zc - cup[2])),
                      pc.rotate(qinv, (0.0, 0.0, a.release_z - zc))]
            try:
                traj = self.cartesian_path(cart, fk, scene.robot_state, deltas,
                                           f'transfer at z={zc:.2f} + lower')
                break
            except RuntimeError as exc:
                errs.append(str(exc))
        if traj is None:
            raise RuntimeError('; '.join(errs))
        self.execute_checked(traj, 'transfer + lower held cup')
        end = pc.gazebo_pose('cup')
        err = math.dist(end[:3], (wx, wy, a.release_z))
        print(f'Cup at ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f}); placement error {err*1000:.0f} mm')
        if err > 0.04:
            raise RuntimeError(f'Cup is {err*1000:.0f} mm from the drop pose after the move')

    def carry_pose(self):
        """Lift the held cup clear of the table and bring it over the base footprint."""
        from moveit_msgs.srv import GetPositionFK, GetMotionPlan, GetCartesianPath
        import place_cup as pc
        import place_cup_hybrid as hybrid
        a, node = self.a, self.node
        scene = self.scene()
        hybrid.check_setup(scene)
        actual = pc.gazebo_pose('cup')
        base = pc.gazebo_pose('a200_0000/robot')
        fk = node.create_client(GetPositionFK, '/compute_fk')
        motion = node.create_client(GetMotionPlan, '/plan_kinematic_path')
        cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        start = scene.robot_state
        palm0 = pc.palm_fk(node, fk, start)
        upright = hybrid.create_upright_path_constraints(palm0)
        q0 = (palm0.orientation.x, palm0.orientation.y, palm0.orientation.z, palm0.orientation.w)
        qb = pc.quat(*base[3:])
        dz = max(0.14, 1.02 - actual[2]) if actual[2] < 0.93 else 0.0
        lifted = (actual[0], actual[1], actual[2] + dz)
        v = pc.rotate(qb, (a.carry_cup_x, a.carry_cup_y, 0.0))
        target = (base[0] + v[0], base[1] + v[1], a.carry_cup_z_world)
        delta_base = pc.rotate(pc.qinv(qb), tuple(target[i] - lifted[i] for i in range(3)))

        # 1) One collision-checked Cartesian path with the palm orientation locked (cup
        #    stays upright by construction): a small lift off the table, then a diagonal
        #    pull-back that rises the rest of the way to the carry height.
        #    A tall straight-up lift first is NOT reachable when the cup is ~0.95 m
        #    ahead (arm near full extension), so try the full lift last-resort only.
        try:
            traj, errs = None, []
            total_rise = target[2] - actual[2]
            for small in (0.10, 0.06, 0.14, dz):
                if small <= 0.0 or small > total_rise + 1e-6:
                    continue
                up = pc.rotate(pc.qinv(qb), (0.0, 0.0, small))
                rest = pc.rotate(pc.qinv(qb), (target[0] - actual[0], target[1] - actual[1],
                                               total_rise - small))
                try:
                    traj = self.cartesian_path(cart, fk, start, [up, rest],
                                               f'lift {small:.2f} m + diagonal retract')
                    break
                except RuntimeError as exc:
                    errs.append(str(exc))
            if traj is None:
                raise RuntimeError('; '.join(errs) or 'no Cartesian candidate')
            self.execute_checked(traj, 'lift + retract held cup')
        except RuntimeError as exc:
            print(f'Cartesian carry not possible: {exc}')
            # 2) Separate lift, then OMPL retract. Unconstrained RRTConnect almost
            #    always twists the palm >20 deg over a 0.6 m retract (every attempt
            #    rejected), so first ask OMPL to hold the palm upright itself.
            virt, lift = start, None
            if dz > 0:
                lift = hybrid.plan_vertical_lowering(node, cart, fk, start, base, dz,
                                                     'carry lift', min_fraction=0.40)
                frac = hybrid.LAST_CARTESIAN_FRACTION[0]
                if actual[2] + frac * dz < 0.89:
                    raise RuntimeError('Lift cannot clear the tabletop')
                delta_base = pc.rotate(pc.qinv(qb), tuple(
                    target[i] - (actual[i] + (frac * dz if i == 2 else 0.0)) for i in range(3)))
                virt = hybrid.state_after(virt, lift)
            desired = pc.waypoint(pc.palm_fk(node, fk, virt), delta_base)
            carry = None
            for constrained, tries, ptime in ((True, 2, 30.0), (False, 5, 18.0)):
                cfg = SimpleNamespace(pipeline='', planner='', attempts=2,
                                      planning_time=ptime, constrain_path=constrained,
                                      max_palm_rotation_deg=20.0, rotation_retries=tries)
                kind = 'upright-constrained' if constrained else 'unconstrained'
                print(f'Retract planning: OMPL {kind} ({tries} tries)')
                try:
                    carry = hybrid.safe_plan_segment(node, motion, fk, virt, desired, upright,
                                                     cfg, q0, f'retract ({kind})')
                    break
                except RuntimeError as exc2:
                    print(f'  {exc2}')
            if carry is None:
                raise RuntimeError('No retract plan kept the palm within 20 deg of upright '
                                   '(Cartesian, constrained and unconstrained OMPL all failed). '
                                   'The cup is still held; nothing was released.')
            if lift is not None:
                self.execute_checked(lift, 'lift held cup')
            self.execute_checked(carry, 'retract cup within base footprint')
        phys = pc.gazebo_pose('cup')
        cp = pc.cup_pose_in_base(phys, pc.gazebo_pose('a200_0000/robot')).position
        print(f'Carry cup base_link XY ({cp.x:.3f}, {cp.y:.3f}), Z world {phys[2]:.3f}')
        if math.hypot(cp.x - a.carry_cup_x, cp.y - a.carry_cup_y) > 0.10 or phys[2] < 0.91:
            raise RuntimeError('Held cup not in the carry pose; refusing to drive')
        if abs(cp.x) > 0.42 or abs(cp.y) > 0.33:
            raise RuntimeError('Held cup too close to the base footprint edge')
        hybrid.check_setup(self.scene())
        return (cp.x, cp.y, cp.z)


def run_script(script, *params):
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, params)]
    if subprocess.run(cmd, timeout=900).returncode != 0:
        raise RuntimeError(f'{script} failed; cup NOT automatically released')


def check_table(wx, wy):
    if not (TABLE_X[0] <= wx <= TABLE_X[1] and TABLE_Y[0] <= wy <= TABLE_Y[1]):
        raise RuntimeError(f'Drop point world ({wx:.3f}, {wy:.3f}) is outside the tabletop '
                           f'placement limits x{TABLE_X} y{TABLE_Y}')


# --------------------------------------------------------------------------- #
def main():
    ap, a = parse_args()
    stations = load_stations()

    if a.list_stations:
        print(f'Stations ({STATION_FILE if STATION_FILE.exists() else "built-in defaults"}):')
        for n, s in sorted(stations.items()):
            tag = '  [gazebo ref]' if s.get('gazebo') else ''
            print(f'  {n:>8}: x={s["x"]:+.3f} y={s["y"]:+.3f} '
                  f'yaw={math.degrees(s["yaw"]):+.1f} deg{tag}')
        return 0
    if a.delete_station:
        stations.pop(a.delete_station, None)
        save_stations(stations)
        print(f'Deleted station {a.delete_station}; {len(stations)} left')
        return 0

    if a.save_station is None:
        need = (('cup_x', 'cup_y') if not a.place_only else ()) + \
               (('drop_x', 'drop_y') if not a.pick_only else ())
        for k in need:
            v = getattr(a, k)
            if v is None or not math.isfinite(v):
                ap.error(', '.join('--' + n.replace('_', '-') for n in need)
                         + ' are required (finite)')
        pick = drop = None
        dpick = ddrop = 0.0
        if not a.place_only:
            pick, dpick = nearest_station(stations, a.cup_x, a.cup_y, a.pick_station)
            print(f'Cup   ({a.cup_x:+.3f}, {a.cup_y:+.3f}) map -> PICK station {pick} '
                  f'({dpick:.2f} m)')
        if not a.pick_only:
            check_table(a.drop_x, a.drop_y)   # fail before any motion if off the table
            drop, ddrop = nearest_station(stations, a.drop_x, a.drop_y, a.drop_station)
            print(f'Drop  ({a.drop_x:+.3f}, {a.drop_y:+.3f}) map -> DROP station {drop} '
                  f'({ddrop:.2f} m)')
        if dpick > a.max_station_dist or ddrop > a.max_station_dist:
            raise RuntimeError(f'Nearest station is farther than {a.max_station_dist} m; '
                               'add a closer station with --save-station')
        if not a.execute:
            print('DRY RUN: station selection only. Add --execute to move.')
            return 0

    import automate_cup as cup
    import place_cup as pc
    import place_cup_hybrid as hybrid

    r = Robot(a)
    try:
        if a.save_station is not None:
            r.check_amcl()
            x, y, yaw = r.tf_xy_yaw('map', 'base_link')
            entry = {'x': x, 'y': y, 'yaw': yaw}
            try:
                g = list(pc.gazebo_pose('a200_0000/robot'))
                entry['gazebo'] = g
                off = math.hypot(g[0] - x, g[1] - y)
                if off > 0.03:
                    # map == Gazebo world here, so a gap means AMCL was mislocalized.
                    # Store the true pose so AMCL and ground-truth checks agree.
                    print(f'WARN: AMCL was {off:.3f} m off Gazebo truth; saving the true pose '
                          '(run seed_amcl_from_gazebo.py to fix AMCL)')
                    entry.update(x=g[0], y=g[1], yaw=g[5], amcl_at_save=[x, y, yaw])
            except Exception as exc:  # noqa: BLE001 - optional ground truth
                print(f'WARN: no Gazebo reference recorded: {exc}')
            stations[a.save_station] = entry
            save_stations(stations)
            print(f'Saved station {a.save_station}: x={x:+.3f} y={y:+.3f} '
                  f'yaw={math.degrees(yaw):+.1f} deg -> {STATION_FILE}')
            return 0

        r.check_amcl()

        r.sync_scene()
        if a.place_only:
            # Recovery / test path: the cup is already in the hand (e.g. after a run that
            # stopped during carry or placement). Skip detection and grasping.
            hybrid.check_setup(r.scene())
            cp0 = pc.cup_pose_in_base(pc.gazebo_pose('cup'),
                                      pc.gazebo_pose('a200_0000/robot')).position
            carry = (cp0.x, cp0.y, cp0.z)
            print(f'PLACE ONLY: cup already held at base_link ({cp0.x:.3f}, {cp0.y:.3f})')
        else:
            # Do nothing if an earlier attempt left the cup attached.
            try:
                cup.require_world_cup(r.scene())
            except RuntimeError as exc:
                raise RuntimeError(f'{exc}. A previous run probably stopped while holding the cup. '
                                   'If the cup is free on the table in Gazebo, run: '
                                   'python3 automate_cup.py --reset') from exc
            truth = pc.gazebo_pose('cup')
            cup.check_table_cup(truth)
            r.require_detector()

            # Simulation sanity check: the requested location should match the real cup.
            bx, by, byaw = r.tf_xy_yaw('map', 'base_link')
            g = pc.gazebo_pose('a200_0000/robot')
            dyaw = byaw - g[5]
            wx, wy = truth[0] - g[0], truth[1] - g[1]
            true_map = (bx + math.cos(dyaw) * wx - math.sin(dyaw) * wy,
                        by + math.sin(dyaw) * wx + math.cos(dyaw) * wy)
            off = math.hypot(true_map[0] - a.cup_x, true_map[1] - a.cup_y)
            print(f'Gazebo cup is at map ({true_map[0]:+.3f}, {true_map[1]:+.3f}); '
                  f'{off:.2f} m from --cup-x/--cup-y')
            if off > a.cup_association_radius:
                raise RuntimeError('Requested cup location does not match the cup in Gazebo')
            best, _ = nearest_station(stations, *true_map)
            if best != pick and a.pick_station is None:
                print(f'WARNING: station {best} is nearer the real cup than station {pick}; '
                      'consider passing the real cup coordinates')

            cup_args = SimpleNamespace(
                look_pose=cup.DEFAULT_LOOK_POSE, calibration=cup.DEFAULT_CALIBRATION,
                robot_model='a200_0000/robot', planning_time=15.0,
                vision_samples=10, vision_timeout=30.0, vision_max_spread=0.010,
                grasp_yaws=[-60, -45, -30, -15, 0, 15, 30, 45, 60], grasp_seeds=4,
                clearance_z=1.00, release_z=a.release_z, backoff=0.08, backoff_dir='palm',
                pregrasp_raise=0.04, approach='cartesian', no_finger_relax=False,
                close_before_attach=False, require_start_cup=False, vision_cup=None)
            calibration = cup.load_calibration(cup_args.calibration)

            say('1/6', 'Arm to wrist-camera look pose')
            cup.move_to_look_pose(r.node, cup_args)

            say('2/6', f'Navigate to pick station {pick}')
            r.goto(pick, stations[pick])

            say('3/6', 'Detect and grasp cup')
            cup_args.vision_cup, base = cup.observe_cup(r.node, cup_args)
            bx, by, byaw = r.tf_xy_yaw('map', 'base_link')
            c = pc.cup_pose_in_base(cup_args.vision_cup, base).position
            cmx = bx + math.cos(byaw) * c.x - math.sin(byaw) * c.y
            cmy = by + math.sin(byaw) * c.x + math.cos(byaw) * c.y
            mismatch = math.hypot(cmx - a.cup_x, cmy - a.cup_y)
            print(f'Detected cup in map ({cmx:+.3f}, {cmy:+.3f}), {mismatch:.3f} m from request')
            if mismatch > a.cup_association_radius:
                raise RuntimeError('Detected cup is not near the requested location')
            if math.dist(cup_args.vision_cup[:3], pc.gazebo_pose('cup')[:3]) > 0.045:
                raise RuntimeError('Vision and Gazebo cup positions disagree by >45 mm')
            cup.update_world_scene(r.node, cup_args.vision_cup, base)
            cup.select_grasp(r.node, r.scene(), cup_args, calibration)
            cup.execute_approach(r.node, cup_args, calibration)

            say('4/6', 'Lift and stow the cup for driving')
            carry = r.carry_pose()
            if a.pick_only:
                say('DONE', f'Cup picked at station {pick} and held in the carry pose. '
                            'Place it with: --place-only --drop-x X --drop-y Y --execute')
                return 0

        if drop != pick or a.place_only:
            say('5/6', f'Navigate with cup to drop station {drop}')
            r.goto(drop, stations[drop])
            cp = pc.cup_pose_in_base(pc.gazebo_pose('cup'),
                                     pc.gazebo_pose('a200_0000/robot')).position
            if math.dist((cp.x, cp.y, cp.z), carry) > 0.08:
                raise RuntimeError('Cup shifted in the hand during the drive; refusing to place')
        else:
            say('5/6', f'Drop point is also nearest station {pick}; base stays put')

        # The map is generated from the world (make_map_from_world.py), so map XY is
        # Gazebo world XY. Converting through the live AMCL pose only added its error.
        wx, wy = a.drop_x, a.drop_y
        check_table(wx, wy)
        hybrid.check_setup(r.scene())

        say('6/6', f'Place at world ({wx:.3f}, {wy:.3f}, {a.release_z:.3f}) and release')
        try:
            r.place_cartesian(wx, wy)
        except RuntimeError as exc:
            print(f'Straight-line placement not possible ({exc}); falling back to '
                  'place_cup_hybrid.py', flush=True)
            run_script('place_cup_hybrid.py', '--place-x', wx, '--place-y', wy,
                       '--release-z', a.release_z, '--clearance-z', a.clearance_z,
                       '--attempts', 2, '--execute')
        run_script('release_cup_safe.py', '--place-x', wx, '--place-y', wy,
                   '--release-z', a.release_z, '--execute')
        # Leave the open hand clear of the cup so the next plan does not clip it.
        cup.retreat_from_cup_if_touching(
            r.node, SimpleNamespace(calibration=cup.DEFAULT_CALIBRATION,
                                    robot_model='a200_0000/robot'))
        say('DONE', 'Cup placed at station ' + str(drop) if a.place_only else
            f'Cup moved: picked at station {pick}, dropped at station {drop}')
        return 0
    finally:
        r.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f'\nABORTED: {exc}', file=sys.stderr)
        print('Inspect the robot and cup in Gazebo. Nothing was detached or reset.',
              file=sys.stderr)
        sys.exit(1)
