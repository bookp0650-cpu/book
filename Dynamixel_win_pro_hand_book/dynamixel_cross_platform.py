# dynamixel_cross_platform.py

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
import ctypes
import time


class Dynamixel:
    """
    Cross-platform Dynamixel control class (Linux/Windows/macOS).

    Port names:
        Windows: "COM7", "COM3", ...
        Linux  : "/dev/ttyUSB0" or "/dev/ttyACM0"
        macOS  : "/dev/tty.usbserial-XXXX"
    """

    # ---- Control Table (Protocol 2.0, X-Series) ----
    __MY_DXL = "X_SERIES"

    __ADDR_TORQUE_ENABLE = 64
    __ADDR_OPERATION_MODE = 11

    __ADDR_VELOCITY_LIMIT = 44
    __ADDR_GOAL_VELOCITY = 104
    __ADDR_PRESENT_VELOCITY = 128

    __ADDR_MAXIMUM_POSITION = 48
    __ADDR_MINIMUM_POSITION = 52
    __ADDR_GOAL_POSITION = 116

    # XC430-W240 Present PWM
    __ADDR_PRESENT_PWM = 124

    __ADDR_PRESENT_POSITION = 132
    __ADDR_PRESENT_INPUT_VOLTAGE = 144

    __TORQUE_ENABLE = 1
    __TORQUE_DISABLE = 0

    __VELOCITY_MODE = 1
    __POSITION_MODE = 3
    __EX_POSITION_MODE = 4

    def __init__(
        self,
        port: str,
        baudrate: int = 57600,
        protocol: float = 2.0,
    ):
        self.__DEVICENAME = port
        self.__BAUDRATE = baudrate
        self.__PROTOCOL_VERSION = protocol

        self.__portHandler = PortHandler(
            self.__DEVICENAME
        )
        self.__packetHandler = PacketHandler(
            self.__PROTOCOL_VERSION
        )

        if not self.__portHandler.openPort():
            raise RuntimeError(
                f"Failed to open port: {self.__DEVICENAME}"
            )

        if not self.__portHandler.setBaudRate(
            self.__BAUDRATE
        ):
            try:
                self.__portHandler.closePort()
            except Exception:
                pass

            raise RuntimeError(
                f"Failed to set baudrate: {self.__BAUDRATE}"
            )

    # =====================================================
    # Helpers
    # =====================================================
    def _check_write(
        self,
        dxl_comm_result,
        dxl_error,
        context="",
    ):
        if dxl_comm_result != COMM_SUCCESS:
            msg = self.__packetHandler.getTxRxResult(
                dxl_comm_result
            )
            raise RuntimeError(
                f"{context} COMM ERROR: {msg}"
            )

        if dxl_error != 0:
            msg = self.__packetHandler.getRxPacketError(
                dxl_error
            )
            raise RuntimeError(
                f"{context} DXL ERROR: {msg}"
            )

    def _read4(
        self,
        dxl_id: int,
        addr: int,
        context="",
    ):
        val, comm, err = (
            self.__packetHandler.read4ByteTxRx(
                self.__portHandler,
                dxl_id,
                addr,
            )
        )

        self._check_write(
            comm,
            err,
            context=context,
        )

        # Extended Position / Velocityでは負値があり得る。
        return ctypes.c_int32(val).value

    def _read2(
        self,
        dxl_id: int,
        addr: int,
        context="",
    ):
        """
        2バイトの符号なしデータを読み取る。
        電圧値などに使用する。
        """

        val, comm, err = (
            self.__packetHandler.read2ByteTxRx(
                self.__portHandler,
                dxl_id,
                addr,
            )
        )

        self._check_write(
            comm,
            err,
            context=context,
        )

        return int(val)

    def _read2_signed(
        self,
        dxl_id: int,
        addr: int,
        context="",
    ):
        """
        2バイトの符号付きデータを読み取る。
        Present PWMなどに使用する。
        """

        val, comm, err = (
            self.__packetHandler.read2ByteTxRx(
                self.__portHandler,
                dxl_id,
                addr,
            )
        )

        self._check_write(
            comm,
            err,
            context=context,
        )

        return ctypes.c_int16(val).value

    def _write1(
        self,
        dxl_id: int,
        addr: int,
        data: int,
        context="",
    ):
        comm, err = (
            self.__packetHandler.write1ByteTxRx(
                self.__portHandler,
                dxl_id,
                addr,
                data,
            )
        )

        self._check_write(
            comm,
            err,
            context=context,
        )

    def _write4(
        self,
        dxl_id: int,
        addr: int,
        data: int,
        context="",
    ):
        comm, err = (
            self.__packetHandler.write4ByteTxRx(
                self.__portHandler,
                dxl_id,
                addr,
                data,
            )
        )

        self._check_write(
            comm,
            err,
            context=context,
        )

    # =====================================================
    # Basic
    # =====================================================
    def ping(self, dxl_id: int) -> int:
        model_number, comm, err = (
            self.__packetHandler.ping(
                self.__portHandler,
                dxl_id,
            )
        )

        self._check_write(
            comm,
            err,
            context=f"PING(ID={dxl_id})",
        )

        return model_number

    def enable_torque(self, dxl_id: int):
        self._write1(
            dxl_id,
            self.__ADDR_TORQUE_ENABLE,
            self.__TORQUE_ENABLE,
            "enable_torque",
        )

    def disable_torque(self, dxl_id: int):
        self._write1(
            dxl_id,
            self.__ADDR_TORQUE_ENABLE,
            self.__TORQUE_DISABLE,
            "disable_torque",
        )

    def close_port(self):
        self.__portHandler.closePort()

    def reopen_port(
        self,
        wait_sec: float = 0.2,
    ):
        """
        通信異常時にシリアルポートを閉じて再オープンする。
        PortHandler / PacketHandler オブジェクト自体は使い回す。
        """

        print("[DXL PORT RECOVERY] closing port...")

        try:
            self.__portHandler.closePort()
        except Exception as exc:
            print(
                "[DXL PORT RECOVERY] closePort warning: "
                f"{type(exc).__name__}: {exc}"
            )

        time.sleep(wait_sec)

        print(
            f"[DXL PORT RECOVERY] opening "
            f"{self.__DEVICENAME} ..."
        )

        if not self.__portHandler.openPort():
            raise RuntimeError(
                f"Failed to reopen port: {self.__DEVICENAME}"
            )

        if not self.__portHandler.setBaudRate(
            self.__BAUDRATE
        ):
            try:
                self.__portHandler.closePort()
            except Exception:
                pass

            raise RuntimeError(
                "Failed to restore baudrate after reopen: "
                f"{self.__BAUDRATE}"
            )

        print(
            "[DXL PORT RECOVERY] recovered: "
            f"port={self.__DEVICENAME}, "
            f"baudrate={self.__BAUDRATE}"
        )

    # =====================================================
    # Modes
    # =====================================================
    def set_mode_velocity(self, dxl_id: int):
        self._write1(
            dxl_id,
            self.__ADDR_OPERATION_MODE,
            self.__VELOCITY_MODE,
            "set_mode_velocity",
        )

        self.write_velocity(
            dxl_id,
            0,
        )

    def set_mode_position(self, dxl_id: int):
        self._write1(
            dxl_id,
            self.__ADDR_OPERATION_MODE,
            self.__POSITION_MODE,
            "set_mode_position",
        )

    def set_mode_ex_position(self, dxl_id: int):
        self._write1(
            dxl_id,
            self.__ADDR_OPERATION_MODE,
            self.__EX_POSITION_MODE,
            "set_mode_ex_position",
        )

    # =====================================================
    # Limits
    # =====================================================
    def set_max_velocity(
        self,
        dxl_id: int,
        max_velocity: int,
    ):
        self._write4(
            dxl_id,
            self.__ADDR_VELOCITY_LIMIT,
            max_velocity,
            "set_max_velocity",
        )

    def set_min_max_position(
        self,
        dxl_id: int,
        min_position: int,
        max_position: int,
    ):
        self._write4(
            dxl_id,
            self.__ADDR_MINIMUM_POSITION,
            min_position,
            "set_min_position",
        )

        self._write4(
            dxl_id,
            self.__ADDR_MAXIMUM_POSITION,
            max_position,
            "set_max_position",
        )

    # =====================================================
    # I/O
    # =====================================================
    def write_velocity(
        self,
        dxl_id: int,
        vel: int,
    ):
        self._write4(
            dxl_id,
            self.__ADDR_GOAL_VELOCITY,
            vel,
            "write_velocity",
        )

    def read_velocity(
        self,
        dxl_id: int,
    ) -> int:
        return self._read4(
            dxl_id,
            self.__ADDR_PRESENT_VELOCITY,
            "read_velocity",
        )

    def write_position(
        self,
        dxl_id: int,
        pos: int,
    ):
        self._write4(
            dxl_id,
            self.__ADDR_GOAL_POSITION,
            pos,
            "write_position",
        )

    def read_position(
        self,
        dxl_id: int,
    ) -> int:
        return self._read4(
            dxl_id,
            self.__ADDR_PRESENT_POSITION,
            "read_position",
        )

    def read_present_pwm(
        self,
        dxl_id: int,
    ) -> int:
        """
        Present PWM(Address 124)の生値を返す。

        XC430-W240では1 count ~= 0.113 %。
        符号はPWM出力方向を表す。
        """

        return self._read2_signed(
            dxl_id,
            self.__ADDR_PRESENT_PWM,
            "read_present_pwm",
        )

    def read_input_voltage(
        self,
        dxl_id: int,
    ) -> float:
        """
        Dynamixelへ供給されている現在の入力電圧[V]を返す。

        Present Input Voltage:
            Address: 144
            Size: 2 bytes
            Unit: 0.1 V
        """

        raw_voltage = self._read2(
            dxl_id,
            self.__ADDR_PRESENT_INPUT_VOLTAGE,
            "read_input_voltage",
        )

        return float(raw_voltage) * 0.1