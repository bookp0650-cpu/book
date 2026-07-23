#!/usr/bin/env python3
import serial
import minimalmodbus

PORT = "/dev/ttyIAI"

BAUDRATES = [
    9600,
    19200,
    38400,
    57600,
    115200,
]

PARITIES = [
    ("NONE", serial.PARITY_NONE),
    ("EVEN", serial.PARITY_EVEN),
]

for baud in BAUDRATES:
    for parity_name, parity in PARITIES:
        print(f"\n=== baud={baud}, parity={parity_name} ===")

        for slave in range(1, 17):
            inst = None

            try:
                inst = minimalmodbus.Instrument(
                    PORT,
                    slave,
                    mode=minimalmodbus.MODE_RTU
                )

                inst.serial.baudrate = baud
                inst.serial.bytesize = 8
                inst.serial.parity = parity
                inst.serial.stopbits = 1
                inst.serial.timeout = 0.2

                inst.clear_buffers_before_each_transaction = True
                inst.close_port_after_each_call = True

                value = inst.read_long(
                    0x9000,
                    functioncode=3,
                    signed=True,
                    byteorder=minimalmodbus.BYTEORDER_BIG
                )

                print("================================")
                print("応答あり")
                print(f"baud   = {baud}")
                print(f"parity = {parity_name}")
                print(f"slave  = {slave}")
                print(f"PNOW   = {value * 0.01:.2f} mm")
                print("================================")

                raise SystemExit(0)

            except minimalmodbus.NoResponseError:
                print(f"応答なし: slave={slave}")

            except minimalmodbus.ModbusException as e:
                # Modbus例外でも応答が返ったなら通信条件は合っている
                print("================================")
                print("Modbus応答あり")
                print(f"baud   = {baud}")
                print(f"parity = {parity_name}")
                print(f"slave  = {slave}")
                print(f"内容   = {e}")
                print("================================")

                raise SystemExit(0)

            except Exception as e:
                print(f"slave={slave}: {type(e).__name__}: {e}")

            finally:
                if inst is not None:
                    try:
                        inst.serial.close()
                    except Exception:
                        pass

print("\n全条件で応答がありませんでした。")
print("配線・変換器・IAI側通信設定を確認してください。")