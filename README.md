# pro_hand_book_python

## Mobile Manipulator System for Automated Book Retrieval  
## 書籍出納システムの構築

本リポジトリは，移動型ロボットによる書籍出納システムにおいて，  
マニピュレータ側の書籍認識，把持，取り出し，収納動作を実現するためのプログラム群です。

RGB-Dカメラ，xArm7，ロボットハンド，昇降機構，バーコードリーダを統合し，  
既存の書架環境に後付け可能な移動マニピュレータシステムの構築を目的としています。

既存の自動書庫は，設備全体の改変が必要となる場合が多く，導入コストが高いという課題があります。  
本研究では，既存の書架や通路環境を活用しながら，段階的に導入可能なロボットシステムの実現を目指しています。

---

## Overview

本システムでは，AMRが対象書架前まで自律移動した後，  
マニピュレータ側のシステムが書籍の認識，把持，バーコード照合，収納・取り出し動作を行います。

主な処理の流れは以下の通りです。

1. AMRが対象書架前へ移動
2. RGB-Dカメラにより書架内の書籍を撮影
3. 画像処理・点群処理により対象書籍の位置と姿勢を推定
4. カメラ座標系からロボット座標系へ変換
5. xArm7により対象書籍へアプローチ
6. ロボットハンドにより書籍を把持
7. バーコードを読み取り，対象書籍かを判定
8. 昇降機構・収納機構と連携し，出庫または収納動作を実行

---

## Features

- Intel RealSense D435iを用いたRGB-D画像取得
- 書籍領域の検出と位置推定
- 点群処理による書籍姿勢の推定
- カメラ座標系からロボット座標系への変換
- xArm7による書籍把持動作
- MoveIt / xArm Python SDKを用いたロボットアーム制御
- Dynamixelを用いたロボットハンド制御
- バーコード読み取りによる書籍照合
- IAI直動シリンダーによる昇降機構制御
- AMR側システムとの通信による統合動作
- 実機環境での統合実験・誤差補正

---

## System Configuration

本リポジトリは，主にマニピュレータ側の処理を担当します。

```text
Book Retrieval System
├── AMR side
│   ├── SLAM / Navigation
│   ├── LiDAR-based localization
│   ├── Shelf-front positioning
│   └── Communication with manipulator PC
│
└── Manipulator side
    ├── RGB-D perception
    ├── Book pose estimation
    ├── Coordinate transformation
    ├── xArm7 control
    ├── Robot hand control
    ├── Barcode verification
    ├── IAI linear actuator control
    └── Storage / retrieval motion
```

AMR側の自律移動システムとはROS 2トピックおよびUDP通信により連携します。  
AMRが書架前に到達した後，本リポジトリのプログラムによって書籍の認識・把持・収納・取り出し動作を実行します。

---

## Hardware

本システムでは，主に以下のハードウェアを使用しています。

- xArm7
- Intel RealSense D435i
- Robot hand
- Dynamixel actuator
- IAI linear actuator
- Mobile robot platform
- LiDAR sensor
- Barcode reader

---

## Software / Libraries

主な使用技術は以下の通りです。

- Python
- ROS 2 Humble
- MoveIt 2
- OpenCV
- NumPy
- RealSense SDK
- xArm Python SDK
- Dynamixel SDK
- Point cloud processing
- Barcode recognition
- UDP communication

---

## GitHubを使った別PCへの環境移行

### 前提環境

動作確認に使用している主な環境は以下です。

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- NVIDIA GPU / CUDA（SAM3をGPUで実行する場合）
- Intel RealSense SDK 2.x

仮想環境、CUDAインストーラ、学習済みモデル、撮影データ、ROS 2の
`build/`、`install/`、`log/` はGitHubには保存しません。別PC側で再作成、
または別のストレージからコピーしてください。

### 1. リポジトリを取得

SSH鍵をGitHubへ登録済みの場合:

```bash
cd ~
git clone git@github.com:bookp0650-cpu/book.git pro_book_SAM3
cd ~/pro_book_SAM3
```

HTTPSを使用する場合:

```bash
cd ~
git clone https://github.com/bookp0650-cpu/book.git pro_book_SAM3
cd ~/pro_book_SAM3
```

### 2. ROS 2 Humbleを準備

ROS 2 Humbleを公式手順でインストールした後、シェルへ読み込みます。

```bash
source /opt/ros/humble/setup.bash
```

ROS依存パッケージを解決し、ワークスペースをbuildします。

```bash
cd ~/pro_book_SAM3
sudo rosdep init  # 初回のみ。既に初期化済みなら不要
rosdep update
rosdep install --from-paths ros2_ws/src --ignore-src -r -y
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

毎回読み込む場合は、次の2行を `~/.bashrc` に追加します。

```bash
source /opt/ros/humble/setup.bash
source ~/pro_book_SAM3/ros2_ws/install/setup.bash
```

### 3. Python仮想環境を作成

仮想環境そのものはPC間でコピーせず、移行先で作り直します。

```bash
cd ~/pro_book_SAM3
python3 -m venv .pro_hand_book_fixed --system-site-packages
source .pro_hand_book_fixed/bin/activate
python -m pip install --upgrade pip
```

SAM3推論環境の固定依存関係は次のファイルにあります。

```bash
python -m pip install -r \
  detection/pro_handbook/sam3_runtime/configs/requirements-inference-direct.txt
python -m pip install -e detection/pro_handbook/sam3_runtime/sam3_source
```

RealSense、xArm、Dynamixel、OCRなど、実機構成に応じたPythonパッケージも
必要です。現在使用中の環境一覧は `env_info.txt` を参照してください。
`env_info.txt` にはROS 2側のパッケージも含まれるため、ファイル全体を
そのまま `pip install -r` する用途ではありません。

### 4. GitHubに含まれないファイルを配置

次のファイルは容量が大きいため、GitHub以外のストレージで移行します。

- `models/` 以下のSAM/SAM3モデル
- CUDAおよび各種ドライバのインストーラ
- `captures/`、`barcode_test_captures/` などの撮影・実験データ
- PC固有のキャリブレーション結果やログ

モデルをコピーした後、YAMLやPythonコードで指定しているモデルパスが
移行先の配置と一致することを確認してください。

### 5. 実機固有設定を確認

実行前に以下を移行先PCの構成へ合わせます。

- xArm7、AMR、IAI機器のIPアドレス
- RealSenseの接続とデバイス権限
- Dynamixelのシリアルポート名と権限
- ハンドアイキャリブレーション行列
- `Retrieval_integration.yaml` 内のパス、棚、動作パラメータ
- CUDA、PyTorch、GPUドライバの互換性

### 6. 起動前の確認

```bash
cd ~/pro_book_SAM3
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
source .pro_hand_book_fixed/bin/activate
python3 -m py_compile Retrieval_integration_SAM3.py
python3 -c "import rclpy; import Retrieval_integration_SAM3; print('IMPORT_OK')"
```

`ModuleNotFoundError: No module named 'rclpy'` が出た場合は、ROS 2 Humbleの
setupと仮想環境が同じターミナルで有効になっているか確認してください。

### GitHubへ更新を送る手順

大容量ファイルや秘密情報が含まれていないことを確認してから送信します。

```bash
cd ~/pro_book_SAM3
git status
git diff --stat
git add README.md .gitignore
git commit -m "docs: add setup and PC migration guide"
git push origin main
```

プロジェクト全体を送る場合も、最初から `git add .` を実行せず、
`git status` で対象を確認してから必要なファイルを追加してください。

---

## My Role

本リポジトリでは，主に以下の実装を担当しました。

- RGB-Dカメラによる書籍認識処理
- 書籍位置・姿勢推定アルゴリズムの実装
- カメラ座標系からロボット座標系への変換処理
- xArm7による書籍把持動作の制御
- ロボットハンドの開閉・把持制御
- バーコード照合処理
- IAI直動シリンダーとの連携
- AMR側システムとの通信処理
- 出庫・収納動作の統合
- 実機環境での動作検証と誤差改善

---

## Technical Challenges

### 1. RGB-D認識の誤差

RealSense D435iによる深度情報には，距離，照明条件，書籍の傾き，表面状態などによって数mm単位の誤差が発生します。  
そのため，深度画像，カメラ内部パラメータ，点群処理結果を用いて，ロボットが把持可能な位置へ変換する処理を実装しました。

### 2. 座標変換の精度

カメラ座標系で得られた書籍位置を，xArm7のロボット座標系へ変換する必要があります。  
ハンドアイキャリブレーション，TCP設定，カメラ取り付け位置の誤差が把持位置に影響するため，実機での動作確認を通じて補正値の調整を行いました。

### 3. 書籍の傾きへの対応

書籍が垂直に並んでいる場合だけでなく，左右に傾いた状態でも把持できるように，  
書籍の姿勢推定結果に基づいてロボットアームの手先姿勢を調整する処理を実装しました。

### 4. 実機統合

認識，アーム制御，ハンド制御，昇降機構，バーコード照合，AMRとの通信を組み合わせ，  
単体動作ではなく，システム全体として動作するように統合しました。

---

## Main Programs

代表的な実行ファイルは以下の通りです。

```text
pro_book_SAM3/
├── Retrieval_integration_SAM3.py
├── Retrieval_integration_input_SAM3.py
├── xarm7/
├── Dynamixel_win_pro_hand_book/
├── ros2_ws/
└── README.md
```

### Retrieval

書籍の取り出し動作を行うメインプログラムです。

```bash
python3 Retrieval_integration_SAM3.py
```

入力確認を含む取り出しプログラムです。

```bash
python3 Retrieval_integration_input_SAM3.py
```

ROS2 launchによる20冊用自動取り出しプログラムです。上記とセットで使います。

```bash
ros2 launch retrieval_manager retrieval_auto.launch.py
```

### Storage

RGB-D認識結果に基づいて，書籍の収納動作を行います。

```bash
python3 Storage_by_Detection2.py
```

---

## Operation Commands

以下のコマンドは，実機実験および統合動作確認で使用したものです。  
ロボット本体，ROS 2環境，ネットワーク設定が完了していることを前提としています。


### 1. Communication Between AMR and Manipulator PC

AMR側PCとマニピュレータ側PCの通信にはUDPブリッジを使用します。

```bash
ros2 launch udp_bridge_manip udp_bridge_manip.launch.py
```

ネットワーク設定の確認例です。

```bash
ip a | grep 172.20
```

IPアドレスを削除する場合の例です。

```bash
sudo ip addr del 172.20.10.2/28 dev wlp132s0f0
```


---

### 2. Activate Python Virtual Environment

Pythonスクリプトを実行する前に，仮想環境を有効化します。

```bash
cd ~/pro_book_SAM3
source .pro_hand_book_fixed/bin/activate
```

---

### 2. Retrieval Operation

書籍取り出し動作用のプログラムを実行します。

```bash
cd ~/pro_book_SAM3

python3 Retrieval_integration_SAM3.py
```

ROS 2 launchファイルを用いた自動取り出し動作は以下のコマンドで実行します。

```bash
ros2 launch retrieval_manager retrieval_auto.launch.py
```

---

### 2. Storage Operation

書籍収納動作を実行します。

```bash
cd ~/pro_book_SAM3
python3 Storage_by_Detection2.py
```

---

### 単体動作用コマンド


### IAI Linear Actuator Control

IAI直動シリンダーをROS 2ノードとして起動します。

```bash
cd ~/pro_book_SAM3/ros2_ws
ros2 run iai_cylinder height_controller
```

目標高さをROS 2トピックで送信します。

```bash
ros2 topic pub --once /target_mm std_msgs/msg/Float32 "{data: 140.0}"
```

---

---

### Shelf ID Input

対象書架IDをROS 2トピックで送信します。

```bash
ros2 topic pub --once /shelf_id std_msgs/String "{data: '2-5-1-1'}"
```

`shelf_id` は，対象書籍の書架位置を表します。

---

### AMR Completion Signal

AMRが目標位置へ到達したことを通知するトピックです。  
単体テスト時には，以下のコマンドで到達信号を手動送信できます。

```bash
ros2 topic pub --once /navigation_goal std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /navigation_goal_final std_msgs/msg/Bool "{data: true}"
```

---

### Manual Velocity Command

`/cmd_vel` に速度指令を送信する例です。

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.03, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

### Robot Hand Adjustment

Dynamixelを用いたロボットハンドの各アクチュエータ位置を確認・調整します。

```bash
cd ~/pro_book_SAM3
source .pro_hand_book_fixed/bin/activate
```

グリッパの位置確認です。

```bash
python3 -m Dynamixel_win_pro_hand_book.id_1_gripper_pos_check
```

回転機構の位置確認です。

```bash
python3 -m Dynamixel_win_pro_hand_book.id_2_sp_rot_pos_check
```

直動機構の位置確認です。

```bash
python3 -m Dynamixel_win_pro_hand_book.id_3_sp_lin_pos_check
```

---

### xArm7 Motion Between Initial Pose and Capture Pose

xArm7を初期姿勢と撮影姿勢の間で移動させます。

初期姿勢から撮影姿勢へ移動します。

```bash
cd ~/pro_book_SAM3
python3 -m xarm7.control.xarm_init_to_capture
```

左側撮影姿勢へ移動します。

```bash
cd ~/pro_book_SAM3
python3 -m xarm7.control.xarm_init_to_capture_left
```

撮影姿勢から初期姿勢へ戻します。

```bash
cd ~/pro_book_SAM3
python3 -m xarm7.control.xarm_capture_to_init
```

---

## ROS 2 Topics

本システムで主に使用するROS 2トピックは以下の通りです。

| Topic | Type | Description |
|---|---|---|
| `/shelf_id` | `std_msgs/String` | 対象書架ID |
| `/navigation_goal` | `std_msgs/msg/Bool` | AMRの目標到達通知 |
| `/navigation_goal_final` | `std_msgs/msg/Bool` | AMRの最終位置到達通知 |
| `/target_mm` | `std_msgs/msg/Float32` | IAI直動シリンダーの目標高さ |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 速度指令 |

---

## Related Repository

AMR側の自律移動システムはこちらです。

- [book_AMR](https://github.com/bookp0650-cpu/book_AMR)

---

## Future Work

今後の改善点は以下の通りです。

- 書籍認識精度の向上
- ハンドアイキャリブレーション精度の改善
- 傾いた書籍に対する把持安定性の向上
- 書籍サイズの違いに対する汎用性向上
- 連続出庫・収納動作の安定化
- AMRとの統合動作の安定化
- 実環境での長時間連続動作検証

---

## Author

Shota Shimazaki  
Intelligent Robotics Laboratory  
Tokyo University of Science
