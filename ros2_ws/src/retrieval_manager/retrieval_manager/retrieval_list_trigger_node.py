#!/usr/bin/env python3

import json
import time
from pathlib import Path
import math
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import String, Bool, Int32, Float32


DEFAULT_CONFIG_PATH = (
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "Retrieval_integration.yaml"
)


class RetrievalListTriggerNode(Node):
    """
    master JSON の順番に書籍出庫要求を送るノード。

    起動・送信シーケンス:
        1. /retrieval_system_ready == True を待つ
        2. /shelf_id の Subscriber 接続を待つ
        3. /shelf_id を送信
        4. 出庫側で /navigation_goal と
           /navigation_goal_final の Subscriber が作られるのを待つ
        5. /navigation_goal = True を送信
        6. 指定時間後に /navigation_goal_final = True を送信
        7. /retrieval_done = True を待つ
        8. 次の本へ進む

    重要:
        timer callback 内で while / sleep による長時間ブロックを行わず、
        状態機械で段階的に進める。
    """

    STATE_INITIAL_WAIT = "initial_wait"
    STATE_WAIT_SYSTEM_READY = "wait_system_ready"
    STATE_WAIT_SHELF_SUBSCRIBER = "wait_shelf_subscriber"
    STATE_AFTER_SHELF_ID = "after_shelf_id"
    STATE_WAIT_NAV_SUBSCRIBERS = "wait_nav_subscribers"
    STATE_AFTER_NAVIGATION_GOAL = "after_navigation_goal"
    STATE_WAIT_RETRIEVAL_DONE = "wait_retrieval_done"
    STATE_ALL_DONE = "all_done"

    def __init__(self):
        super().__init__("retrieval_list_trigger_node")

        # ==========================================
        # ROS parameters
        # ==========================================
        self.declare_parameter(
            "config_path",
            DEFAULT_CONFIG_PATH,
        )
        self.declare_parameter(
            "initial_wait_sec",
            2.0,
        )
        self.declare_parameter(
            "after_shelf_id_wait_sec",
            0.5,
        )
        self.declare_parameter(
            "after_navigation_goal_wait_sec",
            0.5,
        )
        self.declare_parameter(
            "connection_log_interval_sec",
            2.0,
        )

        self.config_path = Path(
            str(
                self.get_parameter(
                    "config_path"
                ).value
            )
        ).expanduser().resolve()

        self.initial_wait_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "initial_wait_sec"
                ).value
            ),
        )

        self.after_shelf_id_wait_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "after_shelf_id_wait_sec"
                ).value
            ),
        )

        self.after_navigation_goal_wait_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "after_navigation_goal_wait_sec"
                ).value
            ),
        )

        self.connection_log_interval_sec = max(
            0.5,
            float(
                self.get_parameter(
                    "connection_log_interval_sec"
                ).value
            ),
        )

        # ==========================================
        # Retrieval_integration.yaml
        # ==========================================
        self.config = self.load_yaml(
            self.config_path
        )

        master_file = (
            self.config
            .get("books", {})
            .get("master_file")
        )

        if not master_file:
            raise RuntimeError(
                "Retrieval_integration.yaml に "
                "books.master_file が設定されていません。"
            )

        master_path = Path(
            str(master_file)
        ).expanduser()

        if master_path.is_absolute():
            self.master_path = (
                master_path.resolve()
            )
        else:
            self.master_path = (
                self.config_path.parent
                / master_path
            ).resolve()

        # ==========================================
        # master JSON
        # ==========================================
        self.books = self.load_master_json(
            self.master_path
        )

        # ==========================================
        # Publisher
        # ==========================================
        # ==========================================
        # READY / retained-data QoS
        # ==========================================
        ready_qos = QoSProfile(
            depth=1,
        )

        ready_qos.reliability = (
            ReliabilityPolicy.RELIABLE
        )

        ready_qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ==========================================
        # shelf_id
        #
        # 統合側が途中再起動しても、
        # 現在処理中の最後の shelf_id を取得できる。
        # ==========================================
        self.shelf_id_pub = (
            self.create_publisher(
                String,
                "/shelf_id",
                ready_qos,
            )
        )

        self.navigation_goal_pub = (
            self.create_publisher(
                Bool,
                "/navigation_goal",
                10,
            )
        )

        self.navigation_goal_final_pub = (
            self.create_publisher(
                Bool,
                "/navigation_goal_final",
                10,
            )
        )


        # ==========================================
        # 現在処理中の本index Publisher
        #
        # 0-based:
        #   0 = 1冊目
        #   1 = 2冊目
        #
        # TRANSIENT_LOCAL にすることで、
        # 統合側が途中で再起動しても
        # 最後に送ったindexを取得できる。
        # ==========================================
        self.book_index_pub = (
            self.create_publisher(
                Int32,
                "/retrieval_book_index",
                ready_qos,
            )
        )


        # ==========================================
        # コンテナ累積offset
        #
        # manager側で保持する。
        # 統合側が再起動しても最後の値を取得できる。
        # ==========================================
        self.container_offset_pub = (
            self.create_publisher(
                Float32,
                "/retrieval_container_offset_mm",
                ready_qos,
            )
        )

        self.container_offset_update_sub = (
            self.create_subscription(
                Float32,
                "/retrieval_container_offset_update_mm",
                self.container_offset_update_callback,
                10,
            )
        )


        # ==========================================
        # 現在の出庫stage
        #
        # manager側で保持する。
        # 統合コードが途中再起動しても、
        # 最後に完了/開始したstageを取得できる。
        # ==========================================
        self.retrieval_stage_pub = (
            self.create_publisher(
                String,
                "/retrieval_stage",
                ready_qos,
            )
        )

        self.retrieval_stage_update_sub = (
            self.create_subscription(
                String,
                "/retrieval_stage_update",
                self.retrieval_stage_update_callback,
                10,
            )
        )


        # ==========================================
        # Subscriber
        # ==========================================
        self.retrieval_done_sub = (
            self.create_subscription(
                Bool,
                "/retrieval_done",
                self.retrieval_done_callback,
                10,
            )
        )

        self.system_ready_sub = (
            self.create_subscription(
                Bool,
                "/retrieval_system_ready",
                self.system_ready_callback,
                ready_qos,
            )
        )

        # ==========================================
        # State
        # ==========================================
        self.index = 0

        # 現在のコンテナ累積offset [mm]
        self.container_offset_mm = 0.0

        # 現在処理中のstage
        self.retrieval_stage = "IDLE"

        self.system_ready = False
        self.state = self.STATE_INITIAL_WAIT


        now = time.monotonic()

        self.start_time = now
        self.next_action_time = now
        self.last_wait_log_time = 0.0

        self.all_done_logged = False

        # 現在処理中の本情報
        self.current_book_name = ""
        self.current_shelf_id = ""

        # ==========================================
        # Timer
        # ==========================================
        self.timer = self.create_timer(
            0.1,
            self.timer_callback,
        )
        self.publish_container_offset()
        self.publish_retrieval_stage()
        # ==========================================
        # Startup log
        # ==========================================
        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "RetrievalListTriggerNode started"
        )
        self.get_logger().info(
            f"Config YAML: {self.config_path}"
        )
        self.get_logger().info(
            f"Master JSON: {self.master_path}"
        )
        self.get_logger().info(
            f"Loaded {len(self.books)} books"
        )
        self.get_logger().info(
            "Waiting for retrieval system READY..."
        )
        self.get_logger().info(
            "========================================"
        )

    # ==================================================
    # YAML
    # ==================================================
    def load_yaml(
        self,
        config_path: Path,
    ) -> dict:
        if not config_path.exists():
            raise FileNotFoundError(
                "YAMLファイルが見つかりません: "
                f"{config_path}"
            )

        try:
            with config_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                config = yaml.safe_load(
                    file
                )
        except yaml.YAMLError as exc:
            raise RuntimeError(
                "YAMLの読み込みに失敗しました: "
                f"{config_path}\n{exc}"
            ) from exc

        if not isinstance(
            config,
            dict,
        ):
            raise RuntimeError(
                "YAMLの最上位が辞書形式ではありません: "
                f"{config_path}"
            )

        return config

    # ==================================================
    # master JSON
    # ==================================================
    def load_master_json(
        self,
        master_path: Path,
    ) -> list:
        if not master_path.exists():
            raise FileNotFoundError(
                "master JSONが見つかりません: "
                f"{master_path}"
            )

        try:
            with master_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                books = json.load(
                    file
                )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "JSONの読み込みに失敗しました: "
                f"{master_path}\n{exc}"
            ) from exc

        if not isinstance(
            books,
            list,
        ):
            raise RuntimeError(
                "master JSONの最上位は"
                "リスト形式である必要があります。"
            )

        return books

    # ==================================================
    # READY callback
    # ==================================================
    def system_ready_callback(
        self,
        msg: Bool,
    ):
        new_state = bool(
            msg.data
        )

        previous_ready = self.system_ready

        self.system_ready = new_state

        # ==================================================
        # 統合プロセス再起動検出
        #
        # managerは現在の本の /retrieval_done を
        # 待っているのに、統合側からREADY=Trueが
        # もう一度届いた
        #
        # → 統合プロセスが再起動したと判断
        # → 現在の本を最初から再送する
        # ==================================================
        if (
            new_state
            and self.state
            == self.STATE_WAIT_RETRIEVAL_DONE
        ):
            self.get_logger().warn(
                "========================================"
            )
            self.get_logger().warn(
                "[RESUME] Retrieval integration "
                "READY received again."
            )
            self.get_logger().warn(
                "[RESUME] Integration process "
                "restart detected."
            )
            self.get_logger().warn(
                "[RESUME] Replay current request: "
                f"index={self.index}, "
                f"book={self.index + 1}/"
                f"{len(self.books)}, "
                f"name={self.current_book_name!r}"
            )
            self.get_logger().warn(
                "========================================"
            )

            # 現在の本番号は進めない。
            # shelf_idから要求シーケンスをやり直す。
            self.state = (
                self.STATE_WAIT_SHELF_SUBSCRIBER
            )

            self.last_wait_log_time = 0.0

            return

        # 通常時、同じREADY値なら何もしない
        if new_state == previous_ready:
            return

        if self.system_ready:
            self.get_logger().info(
                "========================================"
            )
            self.get_logger().info(
                "Received "
                "/retrieval_system_ready: true"
            )
            self.get_logger().info(
                "Retrieval system is READY."
            )
            self.get_logger().info(
                "========================================"
            )

        else:
            self.get_logger().warn(
                "========================================"
            )
            self.get_logger().warn(
                "Received "
                "/retrieval_system_ready: false"
            )
            self.get_logger().warn(
                "New retrieval requests will be paused."
            )
            self.get_logger().warn(
                "A request already in progress "
                "is not cancelled."
            )
            self.get_logger().warn(
                "========================================"
            )
    # ==================================================
    # Container offset
    # ==================================================
    def publish_container_offset(self):
        msg = Float32()
        msg.data = float(
            self.container_offset_mm
        )

        self.container_offset_pub.publish(
            msg
        )

        self.get_logger().info(
            "[CONTAINER OFFSET] published: "
            f"{self.container_offset_mm:.1f} mm"
        )

    def container_offset_update_callback(
        self,
        msg: Float32,
    ):
        value = float(
            msg.data
        )

        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            self.get_logger().error(
                "[CONTAINER OFFSET] "
                f"invalid update: {value}"
            )
            return

        self.container_offset_mm = value

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "[CONTAINER OFFSET] "
            f"updated: {value:.1f} mm"
        )
        self.get_logger().info(
            "========================================"
        )

        # TRANSIENT_LOCAL Publisherへ載せ直す
        self.publish_container_offset()


    # ==================================================
    # Retrieval stage
    # ==================================================
    def publish_retrieval_stage(self):
        msg = String()
        msg.data = str(
            self.retrieval_stage
        )

        self.retrieval_stage_pub.publish(
            msg
        )

        self.get_logger().info(
            "[RETRIEVAL STAGE] published: "
            f"{self.retrieval_stage}"
        )


    def set_retrieval_stage(
        self,
        stage: str,
    ):
        stage = str(stage).strip()

        if not stage:
            self.get_logger().warn(
                "[RETRIEVAL STAGE] "
                "empty stage ignored"
            )
            return

        if stage == self.retrieval_stage:
            return

        old_stage = self.retrieval_stage
        self.retrieval_stage = stage

        self.get_logger().info(
            "[RETRIEVAL STAGE] "
            f"{old_stage} -> {stage}"
        )

        self.publish_retrieval_stage()


    def retrieval_stage_update_callback(
        self,
        msg: String,
    ):
        stage = str(
            msg.data
        ).strip()

        if not stage:
            self.get_logger().warn(
                "[RETRIEVAL STAGE] "
                "empty update ignored"
            )
            return

        self.set_retrieval_stage(
            stage
        )


    # ==================================================
    # retrieval_done callback
    # ==================================================
    def retrieval_done_callback(
        self,
        msg: Bool,
    ):
        if not msg.data:
            return

        if (
            self.state
            != self.STATE_WAIT_RETRIEVAL_DONE
        ):
            self.get_logger().warn(
                "Received /retrieval_done=true, "
                "but no retrieval completion was expected. "
                f"Current state={self.state!r}. Ignored."
            )
            return

        if self.index >= len(
            self.books
        ):
            self.get_logger().warn(
                "Received /retrieval_done=true "
                "after all books completed. Ignored."
            )
            return

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "Retrieval completed: "
            f"{self.index + 1}/"
            f"{len(self.books)}, "
            f"book_name="
            f"{self.current_book_name!r}"
        )
        self.get_logger().info(
            "========================================"
        )


        self.set_retrieval_stage(
            "IDLE"
        )

        self.index += 1

        self.current_book_name = ""
        self.current_shelf_id = ""

        if self.index >= len(
            self.books
        ):
            self.state = (
                self.STATE_ALL_DONE
            )
        else:
            # READY が true のままなら
            # 次の timer callback でそのまま次の本へ。
            # false なら WAIT_SYSTEM_READY で待つ。
            self.state = (
                self.STATE_WAIT_SYSTEM_READY
            )

        self.last_wait_log_time = 0.0

    # ==================================================
    # Publish helper
    # ==================================================
    @staticmethod
    def make_bool_msg(
        value: bool,
    ) -> Bool:
        msg = Bool()
        msg.data = bool(
            value
        )
        return msg

    # ==================================================
    # Connection helper
    # ==================================================
    def subscriber_count(
        self,
        publisher,
    ) -> int:
        return int(
            publisher.get_subscription_count()
        )

    def log_wait_periodically(
        self,
        message: str,
    ):
        now = time.monotonic()

        if (
            self.last_wait_log_time == 0.0
            or (
                now - self.last_wait_log_time
                >= self.connection_log_interval_sec
            )
        ):
            self.get_logger().info(
                message
            )
            self.last_wait_log_time = now

    # ==================================================
    # Book data
    # ==================================================
    def prepare_current_book(
        self,
    ) -> bool:
        """
        現在 index の book を検証し、
        current_book_name/current_shelf_id に保存する。

        不正データはスキップして False を返す。
        """
        if self.index >= len(
            self.books
        ):
            return False

        book = self.books[
            self.index
        ]

        if not isinstance(
            book,
            dict,
        ):
            self.get_logger().error(
                "Book data is not an object. "
                f"index={self.index}, "
                f"data={book!r}. "
                "Skip this entry."
            )
            self.index += 1
            return False

        book_name = str(
            book.get(
                "book_name",
                "",
            )
        )

        shelf_id = str(
            book.get(
                "bookshelf_ID",
                "",
            )
        ).strip()

        if not shelf_id:
            self.get_logger().error(
                "bookshelf_ID is empty. "
                f"index={self.index}, "
                f"book_name={book_name!r}. "
                "Skip this entry."
            )
            self.index += 1
            return False

        self.current_book_name = (
            book_name
        )
        self.current_shelf_id = (
            shelf_id
        )

        return True

    # ==================================================
    # Start one retrieval request
    # ==================================================
    def start_current_book(
        self,
    ):
        if not self.prepare_current_book():
            if self.index >= len(
                self.books
            ):
                self.state = (
                    self.STATE_ALL_DONE
                )
            return

        # ==========================================
        # 新しい本のstage開始
        # ==========================================
        self.set_retrieval_stage(
            "BOOK_START"
        )

        # ==========================================
        # 現在処理する本indexを通知
        # ==========================================
        index_msg = Int32()
        index_msg.data = int(
            self.index
        )

        self.book_index_pub.publish(
            index_msg
        )

        self.get_logger().info(
            "Published /retrieval_book_index: "
            f"{self.index} "
            f"(book {self.index + 1}/"
            f"{len(self.books)})"
        )

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "Preparing retrieval request: "
            f"{self.index + 1}/"
            f"{len(self.books)}"
        )
        self.get_logger().info(
            f"book_name="
            f"{self.current_book_name!r}"
        )
        self.get_logger().info(
            f"bookshelf_ID="
            f"{self.current_shelf_id!r}"
        )
        self.get_logger().info(
            "Waiting for /shelf_id subscriber..."
        )
        self.get_logger().info(
            "========================================"
        )

        self.state = (
            self.STATE_WAIT_SHELF_SUBSCRIBER
        )
        self.last_wait_log_time = 0.0

    # ==================================================
    # Main state machine
    # ==================================================
    def timer_callback(
        self,
    ):
        now = time.monotonic()

        # ------------------------------------------
        # Initial wait
        # ------------------------------------------
        if (
            self.state
            == self.STATE_INITIAL_WAIT
        ):
            if (
                now - self.start_time
                < self.initial_wait_sec
            ):
                return

            self.state = (
                self.STATE_WAIT_SYSTEM_READY
            )
            self.last_wait_log_time = 0.0

        # ------------------------------------------
        # All done
        # ------------------------------------------
        if (
            self.index
            >= len(self.books)
        ):
            self.state = (
                self.STATE_ALL_DONE
            )

        if (
            self.state
            == self.STATE_ALL_DONE
        ):
            if not self.all_done_logged:
                self.get_logger().info(
                    "========================================"
                )
                self.get_logger().info(
                    "All retrieval requests completed."
                )
                self.get_logger().info(
                    f"Total books: "
                    f"{len(self.books)}"
                )
                self.get_logger().info(
                    "========================================"
                )
                self.all_done_logged = True

            return

        # ------------------------------------------
        # System READY
        # ------------------------------------------
        if (
            self.state
            == self.STATE_WAIT_SYSTEM_READY
        ):
            if not self.system_ready:
                self.log_wait_periodically(
                    "Waiting for "
                    "/retrieval_system_ready = true"
                )
                return

            self.start_current_book()
            return

        # ------------------------------------------
        # Wait /shelf_id subscriber
        # ------------------------------------------
        if (
            self.state
            == self.STATE_WAIT_SHELF_SUBSCRIBER
        ):
            # 新しい要求を開始する直前なので、
            # READY が false に戻った場合は送らない。
            if not self.system_ready:
                self.get_logger().warn(
                    "System became NOT READY before "
                    "/shelf_id was sent. "
                    "Returning to READY wait."
                )
                self.state = (
                    self.STATE_WAIT_SYSTEM_READY
                )
                self.last_wait_log_time = 0.0
                return

            count = self.subscriber_count(
                self.shelf_id_pub
            )

            if count <= 0:
                self.log_wait_periodically(
                    "Waiting for subscriber: "
                    f"/shelf_id (count={count})"
                )
                return

            self.get_logger().info(
                "Subscriber ready: "
                f"/shelf_id (count={count})"
            )

            shelf_msg = String()
            shelf_msg.data = (
                self.current_shelf_id
            )

            self.shelf_id_pub.publish(
                shelf_msg
            )

            self.get_logger().info(
                "Published /shelf_id: "
                f"{self.current_shelf_id}"
            )

            self.next_action_time = (
                now
                + self.after_shelf_id_wait_sec
            )

            self.state = (
                self.STATE_AFTER_SHELF_ID
            )
            self.last_wait_log_time = 0.0
            return

        # ------------------------------------------
        # Small delay after /shelf_id
        # ------------------------------------------
        if (
            self.state
            == self.STATE_AFTER_SHELF_ID
        ):
            if now < self.next_action_time:
                return

            self.state = (
                self.STATE_WAIT_NAV_SUBSCRIBERS
            )
            self.last_wait_log_time = 0.0
            return

        # ------------------------------------------
        # Wait navigation subscribers
        #
        # 出庫側は /shelf_id を受信した後に
        # BoolPulseWatcher / BoolLatchWatcher を生成するため、
        # ここで初めて navigation 系の Subscriber を待つ。
        # ------------------------------------------
        if (
            self.state
            == self.STATE_WAIT_NAV_SUBSCRIBERS
        ):
            nav_count = self.subscriber_count(
                self.navigation_goal_pub
            )

            final_count = self.subscriber_count(
                self.navigation_goal_final_pub
            )

            if (
                nav_count <= 0
                or final_count <= 0
            ):
                self.log_wait_periodically(
                    "Waiting for navigation subscribers: "
                    f"/navigation_goal={nav_count}, "
                    f"/navigation_goal_final={final_count}"
                )
                return

            self.get_logger().info(
                "Navigation subscribers ready: "
                f"/navigation_goal={nav_count}, "
                f"/navigation_goal_final={final_count}"
            )

            self.navigation_goal_pub.publish(
                self.make_bool_msg(
                    True
                )
            )

            self.get_logger().info(
                "Published /navigation_goal: true"
            )

            self.next_action_time = (
                now
                + self.after_navigation_goal_wait_sec
            )

            self.state = (
                self.STATE_AFTER_NAVIGATION_GOAL
            )
            self.last_wait_log_time = 0.0
            return

        # ------------------------------------------
        # Delay before final navigation goal
        # ------------------------------------------
        if (
            self.state
            == self.STATE_AFTER_NAVIGATION_GOAL
        ):
            if now < self.next_action_time:
                return

            # 先に WAIT_DONE 状態へ移してから publish する。
            # 非常に早く done が返る構成でも状態不整合を避ける。
            self.state = (
                self.STATE_WAIT_RETRIEVAL_DONE
            )

            self.navigation_goal_final_pub.publish(
                self.make_bool_msg(
                    True
                )
            )

            self.get_logger().info(
                "Published "
                "/navigation_goal_final: true"
            )
            self.get_logger().info(
                "Waiting for "
                "/retrieval_done = true ..."
            )
            self.get_logger().info(
                "========================================"
            )

            self.last_wait_log_time = 0.0
            return

        # ------------------------------------------
        # Wait retrieval done
        # ------------------------------------------
        if (
            self.state
            == self.STATE_WAIT_RETRIEVAL_DONE
        ):
            # callback が index/state を更新する。
            self.log_wait_periodically(
                "Still waiting for "
                "/retrieval_done = true "
                f"({self.index + 1}/"
                f"{len(self.books)}, "
                f"book={self.current_book_name!r})"
            )
            return


def main(
    args=None,
):
    rclpy.init(
        args=args
    )

    node = None

    try:
        node = (
            RetrievalListTriggerNode()
        )

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warn(
                "Interrupted by user"
            )

    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"{type(exc).__name__}: "
                f"{exc}"
            )
        else:
            print(
                "[FATAL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()