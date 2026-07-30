#!/usr/bin/env python3

import json
import time
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


DEFAULT_CONFIG_PATH = (
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "Retrieval_integration.yaml"
)


class RetrievalListTriggerNode(Node):
    def __init__(self):
        super().__init__("retrieval_list_trigger_node")

        # ==========================================
        # ROSパラメータ
        # ==========================================
        self.declare_parameter(
            "config_path",
            DEFAULT_CONFIG_PATH,
        )
        self.declare_parameter("initial_wait_sec", 2.0)
        self.declare_parameter("after_shelf_id_wait_sec", 0.5)
        self.declare_parameter(
            "after_navigation_goal_wait_sec",
            0.5,
        )

        self.config_path = Path(
            str(self.get_parameter("config_path").value)
        ).expanduser().resolve()

        self.initial_wait_sec = float(
            self.get_parameter("initial_wait_sec").value
        )
        self.after_shelf_id_wait_sec = float(
            self.get_parameter("after_shelf_id_wait_sec").value
        )
        self.after_navigation_goal_wait_sec = float(
            self.get_parameter(
                "after_navigation_goal_wait_sec"
            ).value
        )

        # ==========================================
        # Retrieval_integration.yaml読み込み
        # ==========================================
        self.config = self.load_yaml(self.config_path)

        master_file = (
            self.config
            .get("books", {})
            .get("master_file")
        )

        if not master_file:
            raise RuntimeError(
                "Retrieval_integration.yamlに"
                "books.master_fileが設定されていません。"
            )

        master_path = Path(str(master_file)).expanduser()

        # 相対パスなら、YAMLがあるディレクトリを基準にする
        if master_path.is_absolute():
            self.master_path = master_path.resolve()
        else:
            self.master_path = (
                self.config_path.parent / master_path
            ).resolve()

        # ==========================================
        # master JSON読み込み
        # ==========================================
        self.books = self.load_master_json(
            self.master_path
        )

        # ==========================================
        # Publisher
        # ==========================================
        self.shelf_id_pub = self.create_publisher(
            String,
            "/shelf_id",
            10,
        )

        self.navigation_goal_pub = self.create_publisher(
            Bool,
            "/navigation_goal",
            10,
        )

        self.navigation_goal_final_pub = (
            self.create_publisher(
                Bool,
                "/navigation_goal_final",
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

        # ==========================================
        # 状態変数
        # ==========================================
        self.index = 0
        self.waiting_done = False
        self.all_done_logged = False

        self.start_time = time.time()
        self.timer = self.create_timer(
            0.5,
            self.timer_callback,
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

    def load_yaml(self, config_path: Path) -> dict:
        if not config_path.exists():
            raise FileNotFoundError(
                f"YAMLファイルが見つかりません: "
                f"{config_path}"
            )

        try:
            with config_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                config = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise RuntimeError(
                f"YAMLの読み込みに失敗しました: "
                f"{config_path}\n{exc}"
            ) from exc

        if not isinstance(config, dict):
            raise RuntimeError(
                f"YAMLの最上位が辞書形式ではありません: "
                f"{config_path}"
            )

        return config

    def load_master_json(
        self,
        master_path: Path,
    ) -> list:
        if not master_path.exists():
            raise FileNotFoundError(
                f"master JSONが見つかりません: "
                f"{master_path}"
            )

        try:
            with master_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                books = json.load(file)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"JSONの読み込みに失敗しました: "
                f"{master_path}\n{exc}"
            ) from exc

        if not isinstance(books, list):
            raise RuntimeError(
                "master JSONの最上位は"
                "リスト形式である必要があります。"
            )

        return books

    def retrieval_done_callback(
        self,
        msg: Bool,
    ):
        if not msg.data:
            return

        if not self.waiting_done:
            self.get_logger().warn(
                "Received /retrieval_done, "
                "but node was not waiting. Ignored."
            )
            return

        if self.index >= len(self.books):
            self.get_logger().warn(
                "Received /retrieval_done after "
                "all books completed."
            )
            return

        book = self.books[self.index]
        book_name = book.get("book_name", "")

        self.get_logger().info(
            f"Retrieval completed: "
            f"{self.index + 1}/{len(self.books)}, "
            f"book_name={book_name!r}"
        )

        self.index += 1
        self.waiting_done = False

    def publish_bool(
        self,
        publisher,
        value: bool,
    ):
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def publish_current_book_request(self):
        if self.index >= len(self.books):
            return

        book = self.books[self.index]

        if not isinstance(book, dict):
            self.get_logger().error(
                f"Book data is not an object. "
                f"index={self.index}, "
                f"data={book!r}"
            )
            self.index += 1
            return

        book_name = str(
            book.get("book_name", "")
        )

        shelf_id = str(
            book.get("bookshelf_ID", "")
        ).strip()

        if not shelf_id:
            self.get_logger().error(
                f"bookshelf_ID is empty. "
                f"index={self.index}, "
                f"book_name={book_name!r}"
            )
            self.index += 1
            return

        self.get_logger().info(
            f"Start retrieval request: "
            f"{self.index + 1}/{len(self.books)}"
        )

        self.get_logger().info(
            f"book_name={book_name!r}"
        )

        # /shelf_idを送信
        shelf_msg = String()
        shelf_msg.data = shelf_id

        self.shelf_id_pub.publish(shelf_msg)

        self.get_logger().info(
            f"Published /shelf_id: {shelf_id}"
        )

        time.sleep(
            self.after_shelf_id_wait_sec
        )

        # /navigation_goalを送信
        self.publish_bool(
            self.navigation_goal_pub,
            True,
        )

        self.get_logger().info(
            "Published /navigation_goal: true"
        )

        time.sleep(
            self.after_navigation_goal_wait_sec
        )

        # /navigation_goal_finalを送信
        self.publish_bool(
            self.navigation_goal_final_pub,
            True,
        )

        self.get_logger().info(
            "Published /navigation_goal_final: true"
        )

        self.waiting_done = True

    def timer_callback(self):
        elapsed_sec = (
            time.time() - self.start_time
        )

        if elapsed_sec < self.initial_wait_sec:
            return

        if self.waiting_done:
            return

        if self.index >= len(self.books):
            if not self.all_done_logged:
                self.get_logger().info(
                    "All retrieval requests completed."
                )
                self.all_done_logged = True

            return

        self.publish_current_book_request()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = RetrievalListTriggerNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warn(
                "Interrupted by user"
            )

    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"[FATAL] "
                f"{type(exc).__name__}: {exc}"
            )

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

