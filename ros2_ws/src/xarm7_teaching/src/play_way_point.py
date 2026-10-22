#!/usr/bin/env python3
# ==========================
# xarm7 モジュールのパスを明示的に追加（最重要）
# ==========================
import sys
sys.path.append("/home/book/pro_book_SAM3/pro_hand_book_python")

# ==========================
# 通常import
# ==========================
import inspect
import math
import os
import time

import rclpy
from rclpy.node import Node
import yaml

from xarm7.control.xarm7 import XArm7


# ==========================
# 設定
# ==========================
XARM_HOST = "192.168.2.197"

BASE_DIR = os.path.expanduser(
    "~/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"
)

# ==========================
# 既存MoveJ用
# ==========================
# 既存コードから参照されている可能性があるため、
# DEFAULT_SPEED / DEFAULT_ACCEL の名前は変更しない
DEFAULT_SPEED = 0.5     # rad/s
DEFAULT_ACCEL = 1.0     # rad/s^2

# ==========================
# MoveL / MoveL Relative用
# ==========================
DEFAULT_MOVEL_SPEED = 50.0      # mm/s
DEFAULT_MOVEL_ACCEL = 200.0     # mm/s^2

DEFAULT_MOVEL_RELATIVE_SPEED = DEFAULT_MOVEL_SPEED
DEFAULT_MOVEL_RELATIVE_ACCEL = DEFAULT_MOVEL_ACCEL


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


def get_ret_code(ret):
    """
    xArm SDK / 独自ラッパーの戻り値が
      int
      list
      tuple
      None
    のどれでも処理できるようにする。
    """
    if isinstance(ret, (list, tuple)):
        return ret[0] if ret else None

    return ret


def _parameter_accepts_keyword(param):
    """
    inspect.Parameter がキーワード引数を受け取れるか。
    """
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def call_moveL_relative(
    arm,
    relative_pose,
    speed,
    accel,
    wait,
    speed_explicit=False,
    accel_explicit=False,
):
    """
    既存の arm.moveL_relative(...) を壊さずに呼ぶ。

    moveL_relative() が speed / accel / mvacc / wait を
    引数として受け取れる実装なら自動で渡す。

    例:
        def moveL_relative(self, pose, speed=50, accel=200, wait=True)
        def moveL_relative(self, pose, speed=50, mvacc=200, wait=True)

    一方、現在の実装が

        def moveL_relative(self, pose):

    のようにposeしか受け取らない場合、
    YAMLに speed / accel を明示していなければ従来通り
    arm.moveL_relative(relative_pose) を呼ぶ。

    YAMLに speed / accel を明示したのに関数側が受け取れない場合は、
    「速度指定したつもりなのに実際は無視された」という危険を避けるため
    RuntimeErrorにする。
    """

    func = getattr(arm, "moveL_relative", None)

    if func is None or not callable(func):
        raise RuntimeError(
            "XArm7にmoveL_relative()がありません"
        )

    try:
        sig = inspect.signature(func)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    kwargs = {}

    # --------------------------
    # speed
    # --------------------------
    speed_supported = False

    if "speed" in params and _parameter_accepts_keyword(params["speed"]):
        kwargs["speed"] = speed
        speed_supported = True
    elif has_var_kwargs:
        kwargs["speed"] = speed
        speed_supported = True

    # --------------------------
    # accel
    # accel または mvacc に対応
    # --------------------------
    accel_supported = False

    if "accel" in params and _parameter_accepts_keyword(params["accel"]):
        kwargs["accel"] = accel
        accel_supported = True

    elif "mvacc" in params and _parameter_accepts_keyword(params["mvacc"]):
        kwargs["mvacc"] = accel
        accel_supported = True

    elif has_var_kwargs:
        kwargs["accel"] = accel
        accel_supported = True

    # --------------------------
    # wait
    # --------------------------
    if "wait" in params and _parameter_accepts_keyword(params["wait"]):
        kwargs["wait"] = wait

    elif has_var_kwargs:
        kwargs["wait"] = wait

    # --------------------------
    # YAMLで速度指定されたのに
    # moveL_relative側が対応していない場合
    # --------------------------
    if speed_explicit and not speed_supported:
        raise RuntimeError(
            "YAMLにspeedが指定されていますが、"
            "現在のXArm7.moveL_relative()は"
            "speed引数を受け取れません。"
            "moveL_relative側にもspeed引数を追加してください。"
        )

    if accel_explicit and not accel_supported:
        raise RuntimeError(
            "YAMLにaccelが指定されていますが、"
            "現在のXArm7.moveL_relative()は"
            "accel/mvacc引数を受け取れません。"
            "moveL_relative側にも加速度引数を追加してください。"
        )

    # speed / accel / wait を一つでも渡せるなら渡す
    if kwargs:
        print(
            "[WaypointPlayer] moveL_relative kwargs="
            f"{kwargs}"
        )

        return func(
            relative_pose,
            **kwargs,
        )

    # 完全に従来型:
    # arm.moveL_relative([dx,dy,dz,droll,dpitch,dyaw])
    return func(relative_pose)


def execute_waypoint(
    arm,
    wp,
    movej_speed=DEFAULT_SPEED,
    movej_accel=DEFAULT_ACCEL,
    movel_speed=DEFAULT_MOVEL_SPEED,
    movel_accel=DEFAULT_MOVEL_ACCEL,
    movel_relative_speed=DEFAULT_MOVEL_RELATIVE_SPEED,
    movel_relative_accel=DEFAULT_MOVEL_RELATIVE_ACCEL,
    wait=True,
):
    """
    Waypoint 1個を実行する共通関数。

    ============================================================
    1. 既存YAML
    ============================================================

    - name: pose1
      q: [10, 20, 30, 40, 50, 60, 70]

    type省略時は自動的にmovej。
    既存YAMLはそのまま使用可能。

    ============================================================
    2. MoveJ
    ============================================================

    - name: pose1
      type: movej
      q: [10, 20, 30, 40, 50, 60, 70]
      speed: 0.3
      accel: 0.8

    speed: rad/s
    accel: rad/s^2

    ============================================================
    3. MoveL（絶対座標）
    ============================================================

    - name: pose2
      type: movel
      pose: [x, y, z, roll, pitch, yaw]
      speed: 30
      accel: 100

    XYZ: mm
    RPY: deg
    speed: mm/s
    accel: mm/s^2

    ============================================================
    4. MoveL Relative
    ============================================================

    - name: relative1
      type: movel_relative
      pose: [-80, 610, 55, 0, 0, 0]

    内部では基本的に:

        arm.moveL_relative(
            [-80, 610, 55, 0, 0, 0]
        )

    を実行する。

    moveL_relative() 側がspeed/accelを受け取れる場合は:

    - name: relative1
      type: movel_relative
      pose: [-80, 610, 55, 0, 0, 0]
      speed: 30
      accel: 100

    も使用可能。
    """

    name = wp.get("name", "")

    if not name:
        raise RuntimeError(
            f"nameがないWaypointがあります: {wp}"
        )

    # typeがない既存YAMLはすべてMoveJとして扱う
    move_type = str(
        wp.get("type", "movej")
    ).strip().lower()

    # =========================================================
    # MoveJ
    # =========================================================
    if move_type in (
        "movej",
        "joint",
        "j",
    ):
        q_deg = wp.get("q")

        if q_deg is None:
            raise RuntimeError(
                f"{name}: MoveJなのにqがありません"
            )

        if len(q_deg) != 7:
            raise RuntimeError(
                f"{name}: 関節数が不正です: "
                f"{len(q_deg)}"
            )

        q_rad = deg2rad_list(q_deg)

        # YAMLに書いてあればYAMLを優先。
        # なければ従来のplay_with_arm(speed, accel)を使用。
        speed = float(
            wp.get(
                "speed",
                movej_speed,
            )
        )

        accel = float(
            wp.get(
                "accel",
                movej_accel,
            )
        )

        print(
            f"[WaypointPlayer] MoveJ -> {name}"
        )

        print(
            f"[WaypointPlayer] "
            f"speed={speed} rad/s, "
            f"accel={accel} rad/s^2"
        )

        ret = arm.arm.set_servo_angle(
            angle=q_rad,
            speed=speed,
            mvacc=accel,
            is_radian=True,
            wait=wait,
        )

    # =========================================================
    # MoveL（絶対座標）
    # =========================================================
    elif move_type in (
        "movel",
        "linear",
        "l",
    ):
        pose = wp.get("pose")

        if pose is None:
            raise RuntimeError(
                f"{name}: MoveLなのにposeがありません"
            )

        if len(pose) != 6:
            raise RuntimeError(
                f"{name}: poseは6要素必要です。"
                f"[x, y, z, roll, pitch, yaw]"
            )

        x, y, z, roll, pitch, yaw = pose

        speed = float(
            wp.get(
                "speed",
                movel_speed,
            )
        )

        accel = float(
            wp.get(
                "accel",
                movel_accel,
            )
        )

        print(
            f"[WaypointPlayer] MoveL -> {name}"
        )

        print(
            f"[WaypointPlayer] "
            f"pose=["
            f"{x}, {y}, {z}, "
            f"{roll}, {pitch}, {yaw}"
            f"]"
        )

        print(
            f"[WaypointPlayer] "
            f"speed={speed} mm/s, "
            f"accel={accel} mm/s^2"
        )

        ret = arm.arm.set_position(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            speed=speed,
            mvacc=accel,
            is_radian=False,
            wait=wait,
        )

    # =========================================================
    # MoveL Relative
    # =========================================================
    elif move_type in (
        "movel_relative",
        "movel-relative",
        "move_l_relative",
        "linear_relative",
        "relative",
        "lr",
    ):
        # poseを基本とする。
        # deltaという名前でも書けるようにしておく。
        relative_pose = wp.get(
            "pose",
            wp.get("delta"),
        )

        if relative_pose is None:
            raise RuntimeError(
                f"{name}: MoveL Relativeなのに"
                f"poseがありません"
            )

        if len(relative_pose) != 6:
            raise RuntimeError(
                f"{name}: poseは6要素必要です。"
                f"[dx, dy, dz, droll, dpitch, dyaw]"
            )

        relative_pose = [
            float(v)
            for v in relative_pose
        ]

        speed_explicit = "speed" in wp
        accel_explicit = "accel" in wp

        speed = float(
            wp.get(
                "speed",
                movel_relative_speed,
            )
        )

        accel = float(
            wp.get(
                "accel",
                movel_relative_accel,
            )
        )

        print(
            f"[WaypointPlayer] "
            f"MoveL Relative -> {name}"
        )

        print(
            f"[WaypointPlayer] "
            f"relative_pose={relative_pose}"
        )

        if speed_explicit:
            print(
                f"[WaypointPlayer] "
                f"requested speed={speed} mm/s"
            )

        if accel_explicit:
            print(
                f"[WaypointPlayer] "
                f"requested accel={accel} mm/s^2"
            )

        ret = call_moveL_relative(
            arm=arm,
            relative_pose=relative_pose,
            speed=speed,
            accel=accel,
            wait=wait,
            speed_explicit=speed_explicit,
            accel_explicit=accel_explicit,
        )

    # =========================================================
    # 不明なtype
    # =========================================================
    else:
        raise RuntimeError(
            f"{name}: 不明なmotion typeです: "
            f"{move_type}"
        )

    # =========================================================
    # 戻り値確認
    # =========================================================
    ret_code = get_ret_code(ret)

    print(
        f"[WaypointPlayer] "
        f"{name} completed: ret={ret}"
    )

    # Noneは独自moveL_relativeなどで正常終了時にあり得るので
    # エラー扱いしない。
    if ret_code == 3:
        print(
            f"[WaypointPlayer] "
            f"Motion warning at {name}, "
            f"code=3"
        )

    elif (
        isinstance(ret_code, int)
        and ret_code != 0
    ):
        raise RuntimeError(
            f"Fatal motion error at "
            f"{name}, code={ret_code}"
        )

    return ret


class WaypointPlayer(Node):

    def __init__(self, yaml_path: str):
        super().__init__("waypoint_player")

        self.yaml_path = yaml_path

        self.get_logger().info(
            "Connecting to xArm..."
        )

        self.arm = XArm7(
            self,
            host=XARM_HOST,
        )

        time.sleep(0.3)

        self.play()

    # =========================================================
    # 単体再生
    # =========================================================
    def play(self):

        with open(
            self.yaml_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

        if not data:
            raise RuntimeError(
                f"YAMLが空です: "
                f"{self.yaml_path}"
            )

        waypoints = data.get(
            "waypoints"
        )

        if not isinstance(
            waypoints,
            list,
        ):
            raise RuntimeError(
                f"waypointsがありません: "
                f"{self.yaml_path}"
            )

        self.get_logger().info(
            f"Loaded {len(waypoints)} "
            f"waypoints from:\n"
            f"  {self.yaml_path}"
        )

        for wp in waypoints:

            name = wp.get(
                "name",
                "",
            )

            self.get_logger().info(
                f"Execute {name}"
            )

            try:
                # 従来play()はwait=Falseだったため、
                # 既存挙動を維持する。
                ret = execute_waypoint(
                    arm=self.arm,
                    wp=wp,
                    movej_speed=DEFAULT_SPEED,
                    movej_accel=DEFAULT_ACCEL,
                    movel_speed=DEFAULT_MOVEL_SPEED,
                    movel_accel=DEFAULT_MOVEL_ACCEL,
                    movel_relative_speed=DEFAULT_MOVEL_RELATIVE_SPEED,
                    movel_relative_accel=DEFAULT_MOVEL_RELATIVE_ACCEL,
                    wait=True,
                )

                ret_code = get_ret_code(ret)

                if ret_code == 3:
                    self.get_logger().warn(
                        f"Motion warning at "
                        f"{name}, "
                        f"code=3 "
                        f"(continue async)"
                    )

                    time.sleep(1.2)

                else:
                    # 既存play()の動作を維持
                    time.sleep(1.0)

            except Exception as e:
                self.get_logger().error(
                    f"Motion failed at "
                    f"{name}: {e}"
                )

                break

    # =========================================================
    # 既存のarmを使ってWaypoint再生
    # =========================================================
    @staticmethod
    def play_with_arm(
        arm,
        yaml_path,
        start_name=None,
        end_name=None,
        skip_names=None,
        speed=DEFAULT_SPEED,
        accel=DEFAULT_ACCEL,
        wait=True,
        pause_sec=0.0,
    ):
        """
        既に接続済みのXArm7を使ってWaypointを再生する。

        ==================================================
        ★ 後方互換性
        ==================================================

        既存コード:

        WaypointPlayer.play_with_arm(
            arm,
            yaml_path,
            start_name="xxx",
            end_name="yyy",
            speed=0.5,
            accel=1.0,
        )

        は変更不要。

        --------------------------------------------------
        MoveJ
        --------------------------------------------------
        YAMLにspeed/accelが書いてある:
            YAML側を優先。

        YAMLにspeed/accelがない:
            この関数のspeed/accelを使用。

        --------------------------------------------------
        MoveL / MoveL Relative
        --------------------------------------------------
        YAMLにspeed/accelがある:
            YAML側を使用。

        YAMLにない:
            DEFAULT_MOVEL_SPEED /
            DEFAULT_MOVEL_ACCELを使用。

        ※ MoveL Relativeのspeed/accelは、
           XArm7.moveL_relative()側がその引数を
           受け取れる場合に使用可能。
        """

        skip_names = set(
            skip_names or []
        )

        yaml_path = os.path.expanduser(
            str(yaml_path)
        )

        if not os.path.exists(
            yaml_path
        ):
            raise FileNotFoundError(
                f"Waypoint YAMLが"
                f"見つかりません: "
                f"{yaml_path}"
            )

        with open(
            yaml_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

        if not data:
            raise RuntimeError(
                f"YAMLが空です: "
                f"{yaml_path}"
            )

        waypoints = data.get(
            "waypoints"
        )

        if not isinstance(
            waypoints,
            list,
        ):
            raise RuntimeError(
                f"waypointsがありません: "
                f"{yaml_path}"
            )

        print(
            f"[WaypointPlayer] Loaded "
            f"{len(waypoints)} "
            f"waypoints from:"
            f"\n  {yaml_path}"
        )

        started = (
            start_name is None
        )

        start_found = (
            start_name is None
        )

        end_found = (
            end_name is None
        )

        executed_names = []

        for wp in waypoints:

            name = wp.get(
                "name",
                "",
            )

            if not name:
                raise RuntimeError(
                    f"nameがないWaypointがあります: "
                    f"{wp}"
                )

            # =================================================
            # start_name
            # =================================================
            if not started:

                if name != start_name:
                    continue

                started = True
                start_found = True

            # =================================================
            # skip
            # =================================================
            if name in skip_names:

                print(
                    f"[WaypointPlayer] "
                    f"Skip {name}"
                )

                # end_name自体をskipした場合は
                # endに到達したとはみなさない。
                continue

            # =================================================
            # Waypoint実行
            # =================================================
            execute_waypoint(
                arm=arm,
                wp=wp,

                # 従来のspeed/accelは
                # MoveJのデフォルトとしてそのまま使用。
                movej_speed=speed,
                movej_accel=accel,

                # MoveL / MoveL Relativeは専用デフォルト。
                movel_speed=DEFAULT_MOVEL_SPEED,
                movel_accel=DEFAULT_MOVEL_ACCEL,
                movel_relative_speed=DEFAULT_MOVEL_RELATIVE_SPEED,
                movel_relative_accel=DEFAULT_MOVEL_RELATIVE_ACCEL,

                wait=wait,
            )

            executed_names.append(
                name
            )

            # =================================================
            # pause
            # =================================================
            if pause_sec > 0.0:
                time.sleep(
                    pause_sec
                )

            # =================================================
            # end_name
            # =================================================
            if (
                end_name is not None
                and name == end_name
            ):
                end_found = True
                break

        # =====================================================
        # 検査
        # =====================================================
        if not start_found:
            raise RuntimeError(
                f"開始Waypointが"
                f"見つかりません: "
                f"{start_name}"
            )

        if not end_found:
            raise RuntimeError(
                f"終了Waypointが"
                f"見つかりません: "
                f"{end_name}"
            )

        if not executed_names:
            raise RuntimeError(
                "実行されたWaypointが"
                "ありません"
            )

        print(
            "[WaypointPlayer] completed: "
            + " -> ".join(
                executed_names
            )
        )

        return executed_names

    # =========================================================
    # 終了処理
    # =========================================================
    def shutdown(self):

        try:
            self.arm.disconnect()

        except Exception:
            pass


# =============================================================
# YAML選択
# =============================================================
def select_yaml():

    if not os.path.isdir(
        BASE_DIR
    ):
        print(
            f"Config directory "
            f"not found:\n"
            f"  {BASE_DIR}"
        )

        sys.exit(1)

    yamls = sorted(
        [
            f
            for f in os.listdir(
                BASE_DIR
            )
            if f.endswith(
                ".yaml"
            )
        ]
    )

    if not yamls:
        print(
            "No YAML files "
            "found in config/"
        )

        sys.exit(1)

    # =========================================================
    # YAML一覧を番号付きで表示
    # =========================================================
    print(
        "\nAvailable YAML files:"
    )

    for i, filename in enumerate(
        yamls,
        start=1,
    ):
        print(
            f"  [{i}] {filename}"
        )

    # =========================================================
    # 番号入力
    # =========================================================
    while True:

        choice = input(
            "\nEnter YAML number to play: "
        ).strip()

        if not choice:
            print(
                "番号を入力してください。"
            )
            continue

        try:
            index = int(choice)

        except ValueError:
            print(
                "数字で入力してください。"
            )
            continue

        if (
            index < 1
            or index > len(yamls)
        ):
            print(
                f"1〜{len(yamls)}の"
                f"番号を入力してください。"
            )
            continue

        selected_name = yamls[
            index - 1
        ]

        path = os.path.join(
            BASE_DIR,
            selected_name,
        )

        print(
            f"\nSelected YAML:"
            f"\n  [{index}] {selected_name}"
        )

        return path
# =============================================================
# main
# =============================================================
def main():

    # ROS初期化前にYAML選択
    yaml_path = select_yaml()

    rclpy.init()

    node = None

    try:
        node = WaypointPlayer(
            yaml_path
        )

    except KeyboardInterrupt:
        pass

    finally:
        if node:
            node.shutdown()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()