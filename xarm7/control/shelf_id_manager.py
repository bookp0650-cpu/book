# shelf_id_manager.py

from std_msgs.msg import String
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)

class ShelfIDManager:
    """
    Shelf ID を受信して
    - side (left / right)
    - height (mm)
    を決定するロジッククラス

    Nodeは継承しない。
    Subscriptionは外部Nodeに作らせる。
    """

    # --------------------------
    # 野田図書館
    # --------------------------
    # HEIGHT_MAP= {
    #     1: 0.0,
    #     2: 0.0,
    #     3: 200.0,
    #     4: 550.0,
    #     5: 850.0,
    #     6: 1100.0
    # }


    # # TCPのZ微調整（moveLでやる）: C -> tcp_z_offset_mm
    # TCP_Z_OFFSET_MAP = {
    #     1: -380.0,
    #     2: -60.0,
    #     3: 0.0,
    #     4: 0.0,
    #     5: 0.0,
    #     6: 0.0,
    #}

    # --------------------------
    # 2号館  [-300,140,550,850,1250]
    # --------------------------
    HEIGHT_MAP = {
       1: 0.0,
       2: 440.0,
       3: 850.0,
       4: 1100.0,
       5: 1100.0
    }

    #TCPのZ微調整（moveLでやる）: C -> tcp_z_offset_mm
    TCP_Z_OFFSET_MAP = {
       1: 0.0,
       2: 0.0,
       3: 0.0,
       4: 50.0,
       5: 400.0,
    }

    # --------------------------
    # トーハン  固定
    # --------------------------
    # HEIGHT_MAP = {
    #    1: 0.0,
    #    2: 270.0,
    #    3: 600.0,
    #    4: 930.0,
    #    5: 1100.0
    # }

    # # #TCPのZ微調整（moveLでやる）: C -> tcp_z_offset_mm
    # TCP_Z_OFFSET_MAP = {
    #     1: 0.0,
    #     2: 0.0,
    #     3: 0.0,
    #     4: 0.0,
    #     5: 180,
    #     6: 0.0
    # }

    # --------------------------
    # トーハン  
    # --------------------------
    # HEIGHT_MAP = {
    #    1: 1100.0,
    #    2: 1100.0,
    #    3: 1100.0,
    #    4: 1000.0,
    #    5: 750.0,
    #    6: 500.0,
    #    7: 250.0,
    #    8: 0.0,
    # }

    # #TCPのZ微調整（moveLでやる）: C -> tcp_z_offset_mm
    # TCP_Z_OFFSET_MAP = {
    #     1: 0.0,
    #     2: 400.0,
    #     3: 150.0,
    #     4: 0.0,
    #     5: 0.0,
    #     6: 0.0,
    #     7: 0.0,
    #     8: 0.0,
    # }



    def __init__(self, node):
        """
        node: 親となるROS2ノード
        """
        self.node = node

        # ==========================================
        # 内部状態
        # ==========================================
        self.shelf_id = None
        self.B = None
        self.C = None
        self.side = None
        self.height = None
        self.tcp_z_offset = 0.0
        self.received = False

        # ==================================================
        # /shelf_id QoS
        #
        # manager側が最後にpublishした shelf_id を保持し、
        # 統合プロセスが途中再起動しても受信できるようにする。
        # ==================================================
        shelf_id_qos = QoSProfile(
            depth=1
        )

        shelf_id_qos.reliability = (
            ReliabilityPolicy.RELIABLE
        )

        shelf_id_qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscriptionを親ノードに作らせる
        self.subscription = (
            self.node.create_subscription(
                String,
                "/shelf_id",
                self.shelf_callback,
                shelf_id_qos,
            )
        )

    # ==========================
    # Callback
    # ==========================
    def shelf_callback(self, msg: String):
        try:
            data = msg.data.strip()
            parts = data.split('-')

            if len(parts) != 4:
                self.node.get_logger().error(
                    f"Invalid shelf_id format: {data}"
                )
                return

            A, B, C, D = parts
            self.shelf_id = data
            self.B = int(B)
            self.C = int(C)

            # ------------------------
            # B 偶奇 → side
            # ------------------------
            self.side = "left" if self.B % 2 == 0 else "right"

            # ------------------------
            # C → height
            # ------------------------
            if self.C in self.HEIGHT_MAP:
                self.height = self.HEIGHT_MAP[self.C]
                # TCP Z 微調整
                self.tcp_z_offset = self.TCP_Z_OFFSET_MAP.get(self.C, 0.0)
            else:
                self.node.get_logger().error(
                    f"Invalid C value: {self.C}"
                )
                return

            self.received = True

            self.node.get_logger().info(
                f"Shelf → B={self.B}, C={self.C}, "
                f"side={self.side}, "
                f"lift={self.height}mm, "
                f"tcp_offset={self.tcp_z_offset}mm"
            )


        except Exception as e:
            self.node.get_logger().error(
                f"Failed to parse shelf_id: {e}"
            )

    # ==========================
    # Getter
    # ==========================
    def get_side(self):
        return self.side

    def get_height(self):
        return self.height
    
    def get_shelf_id(self):
        return self.shelf_id

    def is_received(self):
        return self.received

    def reset(self):
        self.received = False
        self.shelf_id = None
        self.B = None
        self.C = None
        self.side = None
        self.height = None
        self.tcp_z_offset = 0.0
        
    def get_tcp_z_offset(self):    # TCP微調整
        return self.tcp_z_offset