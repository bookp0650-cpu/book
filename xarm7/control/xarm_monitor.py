import threading
import time
import os
import signal


class XArmMonitor:

    def __init__(
        self,
        arm,
        check_period=0.5,
        auto_stop=True,
        on_abnormal=None,
        on_emergency=None,
    ):
        self.arm = arm
        self.check_period = check_period
        self.auto_stop = auto_stop

        # ==============================
        # 通信例外監視
        # ==============================
        self.exception_count = 0
        self.EXCEPTION_THRESHOLD = 5

        # ==============================
        # xArm状態異常監視
        # ==============================
        self.abnormal_count = 0
        self.ABNORMAL_THRESHOLD = 3

        self.state = None
        self.err = None
        self.warn = None

        # ==============================
        # コールバック
        # ==============================

        # ログ保存など
        self.on_abnormal = on_abnormal

        # 緊急停止処理
        # 主に上下機構の非常停止用
        self.on_emergency = on_emergency

        self.abnormal = False

        self._lock = threading.Lock()

        self._running = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self._thread.start()

    # ============================================================
    # メイン監視ループ
    # ============================================================
    def _loop(self):

        while self._running:

            try:
                state = self.arm.get_state()
                err, warn = self.arm.get_err_warn()

                with self._lock:
                    self.state = state
                    self.err = err
                    self.warn = warn

                # ==============================================
                # 通信成功したら通信例外カウンタリセット
                # ==============================================
                self.exception_count = 0

                # ==============================================
                # xArm状態異常
                # ==============================================
                if state in (4, 5) or err != 0:

                    self.abnormal_count += 1

                    print(
                        "[MONITOR] "
                        f"abnormal_count="
                        f"{self.abnormal_count}/"
                        f"{self.ABNORMAL_THRESHOLD}"
                    )

                else:
                    self.abnormal_count = 0

                # ==============================================
                # 連続異常
                # ==============================================
                if (
                    self.abnormal_count
                    >= self.ABNORMAL_THRESHOLD
                ):
                    self._handle_abnormal(
                        f"state={state}, "
                        f"err={err}, "
                        f"warn={warn}"
                    )

            except Exception as e:

                self.exception_count += 1

                print(
                    "[MONITOR] transient exception "
                    f"({self.exception_count}/"
                    f"{self.EXCEPTION_THRESHOLD}): "
                    f"{type(e).__name__}: {e}"
                )

                # ==============================================
                # 通信異常が規定回数続いた
                # ==============================================
                if (
                    self.exception_count
                    >= self.EXCEPTION_THRESHOLD
                ):
                    self._handle_abnormal(
                        "Monitor exception: "
                        f"{type(e).__name__}: {e}"
                    )

            time.sleep(
                self.check_period
            )

    # ============================================================
    # 異常処理
    # ============================================================
    def _handle_abnormal(self, msg):

        # ==============================================
        # 1回だけ実行
        # ==============================================
        with self._lock:

            if self.abnormal:
                return

            self.abnormal = True

        print("")
        print("========================================")
        print("[MONITOR] ABNORMAL DETECTED")
        print(f"[MONITOR] {msg}")
        print("========================================")

        # ========================================================
        # 1. 上下機構など外部装置を最優先で非常停止
        # ========================================================
        if self.on_emergency is not None:

            try:
                print(
                    "[MONITOR] "
                    "external emergency stop..."
                )

                self.on_emergency(msg)

                print(
                    "[MONITOR] "
                    "external emergency stop done"
                )

            except Exception as e:

                print(
                    "[MONITOR] "
                    "external emergency stop failed: "
                    f"{type(e).__name__}: {e}"
                )

        # ========================================================
        # 2. xArm非常停止
        # ========================================================
        try:

            print(
                "[MONITOR] "
                "xArm emergency stop..."
            )

            self.arm.emergency_stop()

            print(
                "[MONITOR] "
                "xArm emergency stop done"
            )

        except Exception as e:

            print(
                "[MONITOR] "
                "xArm emergency stop failed: "
                f"{type(e).__name__}: {e}"
            )

        # ========================================================
        # 3. ログ保存
        #
        # 停止を先に行う。
        # ODS保存などが遅れてもロボット停止を遅らせない。
        # ========================================================
        if self.on_abnormal is not None:

            try:
                self.on_abnormal(msg)

            except Exception as e:

                print(
                    "[MONITOR] "
                    "abnormal log failed: "
                    f"{type(e).__name__}: {e}"
                )

        # ========================================================
        # 4. システム終了
        # ========================================================
        if self.auto_stop:

            print(
                "[MONITOR] "
                "SYSTEM TERMINATION REQUESTED"
            )

            # ----------------------------------------------------
            # os._exit() は使わない
            #
            # SIGINTを送ることで、
            # Retrieval_integration側のsigint_handler()
            # を必ず通す。
            # ----------------------------------------------------
            os.kill(
                os.getpid(),
                signal.SIGINT,
            )

    # ============================================================
    # 外部用
    # ============================================================
    def is_abnormal(self):

        with self._lock:
            return self.abnormal

    def get_status(self):

        with self._lock:
            return (
                self.state,
                self.err,
                self.warn,
            )

    def stop(self):

        self._running = False

        # 自分自身のスレッドからjoinしない
        if (
            threading.current_thread()
            is not self._thread
        ):
            self._thread.join()


# ================================================================
# 安全なモーション実行
# ================================================================
def safe_motion(
    func,
    monitor,
    where="",
):

    # ============================================================
    # すでに異常状態
    # ============================================================
    if monitor.is_abnormal():

        print(
            "[SAFE_MOTION] "
            f"Already abnormal. "
            f"Skip motion: {where}"
        )

        return

    state, err, warn = (
        monitor.get_status()
    )

    # ============================================================
    # 起動直後
    # ============================================================
    if state is None:

        print(
            "[SAFE_MOTION] "
            f"Monitor state not ready: {where}"
        )

        return

    # ============================================================
    # 動作前チェック
    # ============================================================
    if state in (4, 5) or err != 0:

        print(
            "[SAFE_MOTION] "
            f"Pre-motion abnormal at {where}"
        )

        monitor._handle_abnormal(
            f"Pre-motion abnormal at {where}: "
            f"state={state}, "
            f"err={err}, "
            f"warn={warn}"
        )

        return

    # ============================================================
    # Motion
    # ============================================================
    try:

        ret = func()

        # ========================================================
        # False
        #
        # 「APIエラー」ではなく、
        # モーション関数独自の失敗判定として扱う。
        # システム停止はしない。
        # ========================================================
        if ret is False:

            print(
                "[SAFE_MOTION] "
                f"Non-critical failure at {where}"
            )

            return

        # ========================================================
        # intのAPIエラーコード
        #
        # Pythonではboolもintの派生クラスなので、
        # boolは必ず除外する。
        # ========================================================
        if (
            isinstance(ret, int)
            and not isinstance(ret, bool)
            and ret != 0
        ):

            print(
                "[SAFE_MOTION] "
                f"API error at {where}, "
                f"code={ret}"
            )

            monitor._handle_abnormal(
                f"API error at {where}: "
                f"code={ret}"
            )

            return

        # ========================================================
        # Motion後チェック
        # ========================================================
        state, err, warn = (
            monitor.get_status()
        )

        if state in (4, 5) or err != 0:

            print(
                "[SAFE_MOTION] "
                f"Post-motion abnormal at {where}"
            )

            monitor._handle_abnormal(
                f"Post-motion abnormal at {where}: "
                f"state={state}, "
                f"err={err}, "
                f"warn={warn}"
            )
        # ========================================================
        # 正常終了
        # func() の戻り値を呼び出し元へ返す
        # ========================================================
        return ret
    except Exception as e:

        print(
            "[SAFE_MOTION] "
            f"Motion exception at {where}: "
            f"{type(e).__name__}: {e}"
        )

        monitor._handle_abnormal(
            f"Motion exception at {where}: "
            f"{type(e).__name__}: {e}"
        )