[English](README.md) | **日本語**

# ALICE-Camera

[Project A.L.I.C.E.](https://github.com/anthropics/alice) のカメラキャプチャ・ISPライブラリ

## 概要

`alice-camera` は純RustによるImage Signal Processing（ISP）パイプラインです。生のBayerセンサーデータから最終補正出力までの完全なチェーンをカバーします。

## 機能

- **ホワイトバランス** — 色温度補正のためのチャンネル別ゲイン調整
- **デモザイク** — Bayerパターン（RGGB）からフルRGBへの復元
- **露出制御** — 自動露出調整
- **オートフォーカスメトリクス** — ラプラシアンベースのフォーカススコアリング
- **レンズ歪み補正** — 放射状・接線方向の歪み除去
- **ヒストグラム均一化** — 累積分布によるコントラスト強調
- **ノイズリダクション** — 空間デノイズフィルタ
- **HDRマージ** — 多重露出合成による高ダイナミックレンジ化
- **ガンマ補正** — 知覚的輝度マッピング

## クイックスタート

```rust
use alice_camera::{Rgb, Image};

let mut img = Image::new(640, 480);
img.pixels[0] = Rgb::new(0.8, 0.5, 0.3);

let lum = img.pixels[0].luminance();
let (r, g, b) = img.pixels[0].to_u8();
```

## アーキテクチャ

```
alice-camera
├── Rgb              — f32 RGBピクセル型（輝度計算・変換付き）
├── Image            — 行優先イメージバッファ
├── white_balance    — チャンネル別ゲイン補正
├── demosaic         — Bayer RGGB補間
├── exposure         — 自動露出制御
├── autofocus        — ラプラシアンフォーカスメトリック
├── distortion       — 放射状/接線方向レンズ補正
├── histogram        — ヒストグラム均一化
├── denoise          — 空間ノイズリダクション
├── hdr              — 多重露出HDRマージ
└── gamma            — ガンマカーブ補正
```

## ライセンス

MIT OR Apache-2.0
