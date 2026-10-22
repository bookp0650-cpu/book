from __future__ import annotations
import time
import traceback
import sys
import threading

# ROS 2 関連のインポート
import rclpy
from rclpy.executors import MultiThreadedExecutor

# パスが通っていない場合のための保険（不要であれば削除可）
sys.path.append("/home/book/pro_book_SAM3/pro_hand_book_python")

from xarm7.control.xarm7 import XArm7

BOOK_BARCODE_1 = [36.69, -48.12, 179.98, 53.91, 102.09, -0.54, -18.31]
BOOK_BARCODE_2 = [-36.9, -75.2, 159.3, 80.5, 83.8, 18.8, -34.9]
def test_arm_motion_only(arm: XArm7):
    """
    カメラやバーコード認識を省き、アームの軌道と姿勢制御のみをテストする関数
    """
    print("\n--- アーム動作テスト開始 ---")
    try:
        # 1. 撮影用の基本ポジションへ移動
        print("[1/5] 右側の撮影基本位置へ移動します (moveJ_to_capture_right)")
        arm.moveJ_to_capture_right()
        time.sleep(1.0) # 動作確認用に少しだけ待機

        # 2. 最初の撮影姿勢へ移動
        print("[2/5] バーコード撮影姿勢1へ移動します (BOOK_BARCODE_1)")
        arm.switch_gripper_pose(BOOK_BARCODE_1)
        time.sleep(1.0)

        # 3. 二番目の撮影姿勢へ移動
        print("[3/5] バーコード撮影姿勢2へ移動します (BOOK_BARCODE_2)")
        arm.switch_gripper_pose(BOOK_BARCODE_2)

        # 4. 本来カメラで撮影・解析を行っている時間の待機
        print("[4/5] 撮影フェーズを想定した2秒間の待機...")
        time.sleep(2.0)

    except Exception:
        print("\n[エラー] アームの移動中にエラーが発生しました:")
        print(traceback.format_exc())

    #finally:
        # 5. 終了処理（エラーが起きても必ず実行される）
        print("[5/5] 初期姿勢(姿勢1)へ戻します (finallyブロック)")
        try:
            arm.switch_gripper_pose(BOOK_BARCODE_1)
            # arm.moveJ_to_capture_right() 
        except Exception as e:
            print(f"[警告] 初期姿勢への復帰に失敗しました: {repr(e)}")
            
    print("--- アーム動作テスト終了 ---\n")


if __name__ == "__main__":
    print("[Main] ROS 2 の初期化を行います...")
    rclpy.init()
    
    # テスト専用のノードを作成
    node = rclpy.create_node("book_barcode_test_node")
    
    # 統合コードにあったIPアドレスを設定
    XARM_HOST = "192.168.2.197"
    
    print(f"[Main] XArm7 ({XARM_HOST}) へ接続します...")
    try:
        arm = XArm7(node=node, host=XARM_HOST)
        
        # ROS 2 の通信を裏側で処理し続けるためのエグゼキュータ設定
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        # アームを動かしながら裏で通信処理(spin)をするため、別スレッドで実行
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        
        # ここでメインのアーム動作テストを実行
        test_arm_motion_only(arm)
        
    except KeyboardInterrupt:
        print("\n[Main] ユーザーによって中断されました。")
    except Exception as e:
        print(f"\n[Main] エラーが発生しました: {e}")
        print(traceback.format_exc())
    finally:
        print("[Main] ROS 2 の終了処理を行います...")
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass