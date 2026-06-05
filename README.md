# pro_hand_book_python

## 書籍出納システムの構築

本リポジトリは，移動型ロボットによる書籍出納システムにおいて，  
ロボットアーム，ロボットハンド，RGB-Dカメラ，昇降機構を統合し，書籍の認識・把持・収納・取り出し動作を実現するためのプログラム群です。

既存の自動書庫のように設備全体を大きく改変するのではなく，既存の書架環境に後付け可能な移動マニピュレータシステムを目指して開発しています。

---

## Overview

本システムでは，RGB-Dカメラによって書架内の書籍位置を認識し，  
ロボットアームとハンドを用いて対象書籍の把持・取り出し・収納を行います。

主な処理の流れは以下の通りです。

1. 書架前でRGB-Dカメラにより書籍を撮影
2. 画像処理・点群処理により対象書籍の位置と姿勢を推定
3. カメラ座標系からロボット座標系へ変換
4. xArm7により対象位置へアプローチ
5. ロボットハンドにより書籍を把持
6. バーコードを読み取り，対象書籍かを判定
7. 昇降機構や収納機構と連携し，出庫・収納動作を実行

---

## Features

- RealSense D435iを用いたRGB-D画像取得
- 書籍領域の認識と位置推定
- 点群処理による書籍姿勢の推定
- カメラ座標系からロボット座標系への変換
- xArm7による書籍把持動作
- MoveIt/pythonSDKを用いた軌道生成
- ロボットハンドによる把持・開閉制御
- バーコード読み取りによる書籍照合
- 昇降機構との連携
- AMR側システムとの通信による統合動作

---

## System Configuration

本リポジトリは，主にマニピュレータ側の処理を担当します。

```text
Book Retrieval System
├── AMR side
│   ├── SLAM / Navigation
│   ├── LiDAR-based localization
│   └── Shelf-front positioning
│
└── Manipulator side
    ├── RGB-D perception
    ├── Book pose estimation
    ├── xArm7 control
    ├── Robot hand control
    ├── Barcode verification
    └── Storage / retrieval motion
