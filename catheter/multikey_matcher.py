#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多段query + 全体最適割当（ハンガリー法）による対象マスク選択。

本番 OCR/only_one_tilted.match_text_to_mask_main のドロップイン置換。
既存ファイルは一切変更せず、呼び出し側の import を差し替えるだけで使える。

    # 変更前
    from .OCR.only_one_tilted import match_text_to_mask_main
    # 変更後
    from catheter.scripts.multikey_matcher import match_text_to_mask_main

■ 何を変えるのか
  現行: query（REF）1本で、各マスクを独立にスコアリングして argmax を取る。
        → REFの印字が小さすぎて読めない品目（例 MC1715000）は選定に失敗する。
        → 似た品目（Excelsior XT-17 / XT-27 / SL-10）で取り違えが起きる。

  本実装:
    1. 多段query
       マスタから query(REF) に対応する display_name と有効期限を引き、
       REF / display_name / 期限 の3キーで採点して最大値を採用する。
       実測: 確信ありが 56/80 → 78/80 に改善（catheter-100・4画像・20品目）。
    2. 全体最適割当
       「全マスク × マスタ全品目」のスコア行列を作りハンガリー法で1対1に割り当てる。
       「他の品目がより強く欲しがっているマスク」は取られるため、
       似た品目どうしの取り違えが構造的に減る。
       棚とマスタが1対1に閉じている場合は、読めない品目も消去法で確定できる。

■ 返り値
  本番と同一形式（score降順）:
      [{"name": "mask_3", "score": 87, "box": {"x1":..,"y1":..,"x2":..,"y2":..},
        "forced_angle": 90}, ...]
  name の末尾数字が1始まりのマスク番号。呼び出し側 merge_ocr_and_masks は
  re.search(r"(\\d+)$", name) でこれを取り出すため、命名規則を厳守している。

■ 保存されるデバッグ出力（shot_dir配下）
  multikey_match_debug.json : キー別スコア・割当・採用理由
  ocr_overlay_perkey_top1.png : 選択マスク(query=target_jの割当先)に帰属する
      OCR検出のうち、5キー(ref/display_name/date/spec_1/spec_2)それぞれで
      最もスコアが高かった1件だけを切り出した可視化（値が空のキーは対象外）
  ocr_overlay_all_assignments.png : 全マスクの矩形とHungarian割当(1回の認識で
      マスタ全品目×全マスクを一括で最適割当している)の結果を1枚にまとめて
      可視化したもの(queryは赤枠太線、他品目は色分け、割当の無いマスクは
      灰色枠)。「query以外の品目も正しく別マスクへ選べているか」を一望で
      目視確認するための画像(2026-08-26試験導入、match_text_to_mask_main(...,
      save_all_assignments=True)のときのみ生成。既定Falseで、数値情報のみ
      multikey_match_debug.jsonのall_assignmentsに常に出力される)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidfuzz import fuzz

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # scipy が無い環境では貪欲法にフォールバック
    _HAS_SCIPY = False


# ===== 設定 =====

# マスタJSON。呼び出し時に master_json= で上書きできる。
# 2026-10-22: 実機取り出しの本番既定を、今週の試験・改良で検証済みの0911版に統一
# (以前の既定master_catheter_20260216.json→master_catheter_reco.jsonは30件で、
# 今週の検証(33件、0911版)とは別物だった)。
DEFAULT_MASTER_JSON = (
    Path(__file__).resolve().parents[1] / "master_json" / "master_catheter_0911.json"
)

# 低信頼のOCR断片はゴミ文字になりやすいので落とす
MIN_REC_SCORE = 0.5

# 従来手法（比較実験用）: 環境変数 MULTIKEY_LEGACY=1 で有効になる。
# query(book_name = REF) 1本だけで採点し、領域ごとに独立して最尤候補を選ぶ
# （多段クエリもハンガリー法も使わない）。2026-08-30、ポスターの従来手法/
# 提案手法の比較のために追加。呼び出し側で use_multikey / use_hungarian を
# 明示指定した場合はそちらが優先される。
LEGACY_MODE = os.environ.get("MULTIKEY_LEGACY", "") == "1"

# 根拠ゼロの割当を棄却する（比較実験用）: 環境変数 MULTIKEY_DROP_ZERO=1 で有効。
# ハンガリー法は「マスタの全品目を必ずどこかのマスクへ割り当てる」ため、棚に写って
# いない品目まで空いたマスクへ押し込まれる。その過程で、実在する品目の正解マスクが
# 横取りされる事例が確認された（2026-09-01、書籍100test の test_index=82:
# 目標書籍が mask_17 で88.9点だったにもかかわらず、他に行き場のない「演習機械振動学」
# (50.0点)へ mask_17 を譲り、目標は次点の mask_11(80.0点)に回されて誤選択となった）。
# 有効時は「全キーの生スコアが0の組み合わせ」を割当候補から外し、根拠が無い品目を
# 押し込まないようにする。既定は無効で、無指定なら挙動は一切変わらない。
DROP_ZERO_ASSIGN = os.environ.get("MULTIKEY_DROP_ZERO", "") == "1"

# しきい値未満の対応を「該当なし」として棄却する:
# 2026-10-19、ユーザー要望により既定を有効(ON)に変更。無効にしたい場合のみ
# 環境変数 MULTIKEY_REJECT_LOW_SCORE=0 を指定する。
# ハンガリー法は行列の形状で挙動が変わる: マスク数がマスタ品目数より多いと、
# マスタの全品目が必ずどれかのマスクへ割り当てられる。逆にマスタの方が多いと、
# 割当から漏れた品目は「割当」節の独立argmaxフォールバックがスコアを問わず
# 強制的に拾ってしまう。どちらの場合も「マスタに載っているが棚に無い品目」や
# 「棚にはあるがマスタに載っていない品目」を無理やり対応付けてしまう
# (2026-09-24、指導教員からの指摘: 一対一対応は「マスタの全品目が画像に
# 写っている」前提でないと成立しない)。有効時は、assignの構築直後と
# 独立argmaxフォールバックの両方で、しきい値(呼び出し時のthreshold引数)未満の
# ペアを除外し、該当する対応が無ければ「見つからない」を返せるようにする。
# 注意: スコアSは raw_sum なら5キー合計(最大500点)で、既定のthreshold=40は
# 0〜100点尺度の名残のためほぼ効かない(reco/0911で選択マスクS<40は201件中2件、
# どちらもS=0)。実際に使うときは呼び出し側でthresholdを尺度に合わせて渡すこと。
REJECT_LOW_SCORE = os.environ.get("MULTIKEY_REJECT_LOW_SCORE", "1") != "0"

# 足切りに使うスコアの種類: 環境変数 MULTIKEY_REJECT_RULE (既定"sum")。
# 2026-09-26にreco/0911で検証した結果、5キー合計(S)は「映っていない品目」を見分けられない
# ことが分かった(該当なしにしたい18件の合計S中央値271に対し、映っている正解106件の中央値
# 324で大差が無い。原因はSPEC_1/SPEC_2が「単位が読めない数値」へ80点を与える等でどの箱にも
# 点が出やすいこと)。識別力があるのはref/display_nameの2キーだけで、この2つの合計だけで
# 判定する方が、正解を落とす数に対して該当なし/誤選択を除ける数がはるかに多かった
# (reco/0911原因調査/しきい値足切り分析_20260926/状況1と状況2_結果まとめ.md 参照)。
#   "sum"             : 5キー合計S(既存、raw_sumなら最大500点)で判定
#   "ref_display_name": ref+display_nameの合計(最大200点)で判定(2026-09-29追加)
REJECT_RULE = os.environ.get("MULTIKEY_REJECT_RULE", "sum").strip()
if REJECT_RULE not in ("sum", "ref_display_name"):
    raise ValueError(f"MULTIKEY_REJECT_RULEはsum/ref_display_nameのいずれかにしてください: {REJECT_RULE!r}")

# 該当なし足切りのしきい値を、コードを変えずに指定する: 環境変数 MULTIKEY_REJECT_THRESHOLD=200 等
# (2026-09-26追加、ユーザー承認)。REJECT_LOW_SCORE=1のときだけ意味を持つ。未指定なら、
# ルールごとの既定値を使う: "sum"は呼び出し側から渡されるthreshold引数(既定40、0〜100点
# 尺度の名残でほぼ効かない)、"ref_display_name"は100.0(上記の検証で選んだ値)。
_REJECT_THRESHOLD_ENV = os.environ.get("MULTIKEY_REJECT_THRESHOLD", "").strip()
REJECT_THRESHOLD = float(_REJECT_THRESHOLD_ENV) if _REJECT_THRESHOLD_ENV else None
REJECT_RULE_DEFAULT_THRESHOLD = {"sum": None, "ref_display_name": 100.0}  # sum は呼び出し時のthreshold引数を使う

# 「確信あり」の判定。
# 2026-08-25まではscore>=70 and margin>=20(margin=最高スコアと2位の差)だけで
# 判定していたが、REFコード同士が非常に似通っている(INC-11814-125 / -146等)ため、
# marginは正しいマッチでも小さくなりがちで、確信度の指標として頼りにならないと
# ユーザーから指摘を受けた。そこで2段構成に変更する:
#   1) REFスコアが100(完全一致)なら、他キーの結果に関わらず確信ありとする。
#      同一タイトルの品目が大量にあってもREFは品目ごとに一意なので、
#      完全一致は最も強い証拠になる。
#   2) REFスコアが低い場合は、display_name/date/spec_1/spec_2という4本の
#      補助キーのうち何本が高スコアかで総合判断する(marginは使わない)。
#      display_nameだけでは同一タイトル品目を区別できないため、
#      複数の補助キーが揃って高スコアであることを求める。
REF_EXACT_SCORE = 100.0
SUPPORT_KEY_SCORE = 70.0     # 補助キー1本が「効いている」とみなす閾値
SUPPORT_KEY_MIN_COUNT = 2    # 確信ありとみなすために必要な補助キーの本数

# 旧方式(score/marginベース)の名残。デバッグ出力の参考値としてのみ使う。
CONFIDENT_SCORE = 70.0
CONFIDENT_MARGIN = 20.0

# combined(マスクに帰属したOCRテキスト)がこれより短い場合はスコアを信用しない。
# fuzz.partial_ratio は短い文字列ほど「たまたま部分一致」しやすく、2文字の
# OCR断片("00"等)が長いREFコード("M00345100950"等)に含まれるだけで満点(100)に
# なってしまう実例が確認された(2026-08-21)。過去に軸検出側で見つかった同種の
# バグ(HANDOFF_20260731.md、1文字断片が満点になっていた件)と同根の問題。
MIN_KEY_TEXT_LEN_FOR_MATCH = 3

# キースコアの合成方式。
# 2026-08-28、ユーザー要望: OPTI0152CSS10/OPTI0153CSS10のように、5キー中
# display_name/spec_1が完全に同一で、spec_1のような共有キーが複数マスクで
# 同時に満点(100)になると、旧既定のcentered_max(中央値補正+MAX)ではその
# 同点キーだけで決着してしまい、本来決め手になるべき他キー(spec_2等)の差が
# 完全に無視される問題が判明した(実データで検証: ROOB-3のOPTIMA系45件中
# 6件が同点による誤選択)。raw_sumは中央値補正をせず5キーの生スコアを単純
# 合計する方式で、ROOB-1(近縁品目中心・96件、目視レビュー済み)で95.8%正解
# (centered_max由来の同点誤選択が解消)を確認した。
# 2026-09-01、ユーザー承認によりraw_sumを本番の既定値に変更(ポスターの
# 提案手法の数値もraw_sum実行分を採用)。centered_maxはstand-100のような
# 多品目混在データセットでの影響が未検証(2026-08-28時点、実データ50/127件で
# 選択結果が変化、大半が目視未レビュー)なため、比較用に環境変数
# MULTIKEY_SCORE_COMBINE=centered_max で明示的に戻せるようにしてある。
SCORE_COMBINE_METHOD = os.environ.get("MULTIKEY_SCORE_COMBINE", "raw_sum").strip().lower()

FORCED_ANGLE = 90


# ===== 文字列正規化 =====

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in ("®", "™", "©"):
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip().upper()


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """_normalizeと同じ正規化をしつつ、出力の各文字が元の文字列sの何文字目から
    来たかを対応表(idx_map)として返す(idx_map[i] = 正規化後i文字目の、s内の位置)。

    2026-10-16追加: REF/商品名の照合(fuzz.partial_ratio)で実際に一致した範囲を
    combined(正規化前)側の位置として求めるために必要(_consumed_span_for_ref_or_name
    参照)。NFKCは文字によって複数文字に展開されることがあるため、1文字ずつ
    NFKC正規化してから結合することで、展開後の各文字がsのどの位置由来かを追える
    ようにしている(文字列全体を一度にNFKC正規化する_normalizeとは展開結果が
    理論上ズレうるが、本プロジェクトのOCRテキスト(英数字主体)では実質的に
    差が出ない)。
    """
    if not s:
        return "", []
    chars: list[str] = []
    idx_map: list[int] = []
    special = {"®", "™", "©"}
    for i, ch in enumerate(s):
        for nc in unicodedata.normalize("NFKC", ch):
            if nc in special:
                continue
            chars.append(nc)
            idx_map.append(i)

    out_chars: list[str] = []
    out_idx: list[int] = []
    prev_space = False
    for ch, oi in zip(chars, idx_map):
        if ch.isspace():
            if prev_space or not out_chars:
                prev_space = True
                continue
            out_chars.append(" ")
            out_idx.append(oi)
            prev_space = True
        else:
            out_chars.append(ch.upper())
            out_idx.append(oi)
            prev_space = False
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_idx.pop()
    return "".join(out_chars), out_idx


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (s or ""))


# 選択マスクのOCRテキストが「識別の根拠として意味を持ちそうか」の簡易判定。
# 2026-08-21、他セッションとの共同分析(88件の目視レビュー)で、選択マスクの
# OCRテキスト長がT(正解)/F(誤り)を分ける最も強いシグナルだと判明した
# (T群 text_len 中央値12、F群中央値4〜7.5)。ただし文字数だけでは
# sensitivity/specificityが頭打ちになる(F群にも誤読で長くなったガーベジ文字列が
# 混じる)ため、長さに加えて「REFコードや日付らしい形式か」も見る2段構成にする。
MIN_TEXT_LEN_FOR_PLAUSIBLE = 3


def _looks_like_plausible_identifier(text: str) -> bool:
    """selected_maskのcombinedテキストが、REF/日付として意味を持ちそうかを判定する。

    根拠にならない短い断片・でたらめなOCR誤読を、スコアが高くても弾くための
    2段目のガード(1段目は_key_scoreの最小文字数ガード)。
    """
    t = (text or "").strip()
    if len(_normalize(t)) < MIN_TEXT_LEN_FOR_PLAUSIBLE:
        return False
    digits = _digits_only(t)
    has_long_digit_run = len(digits) >= 4
    has_alnum_mix = bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"\d", t))
    looks_like_date = bool(re.search(r"\d[\d\-/.]{5,}", t))
    return has_long_digit_run or has_alnum_mix or looks_like_date


def _extract_value_unit_pairs(s: str) -> list[tuple[float, str]]:
    """"1.5 mm"、"2cm"のような(数値, 単位)のペアを文字列から全て抜き出す。
    単位が読めない/無い場合は unit="" になる。"""
    pairs = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)", s):
        pairs.append((float(m.group(1)), m.group(2).lower()))
    return pairs


# 2026-10-16: spec(size)の数値抽出から、日付・REFコードらしい並びに含まれる数字を
# 除外するためのパターン。実例2件(いずれも2026-10-12のユーザーとの原因分析で発覚):
#   1. 日付「2029-03-31」の月(03=3.0)が、SHIDENのSPEC_1「3.0 mm」と誤って一致
#   2. REFコード「SS2-40-070H」の40が、SHIDENのSPEC_2「40 mm」と誤って一致
# 詳細はHANDOFF_20261012_spec_numeric_false_match_todo.md参照。
# 2026-10-16: 当初は_key_scoreのis_date分岐と同じ緩い正規表現(\d[\d\-/.]{5,})を
# 流用していたが、"3.2/3.4Fr."(KMDA044178自身の正しいspec値)まで日付っぽい並びと
# 誤認識して消してしまう回帰が見つかった(実データでの回帰確認により判明)。
# 西暦4桁で始まる並びだけに絞ることで、このような短い数値表記を誤って巻き込まない
# ようにする(この項目の実例の日付「2029-03-31」「2029-01」はどちらも対象のまま)。
_DATE_LIKE_RE = re.compile(r"\d{4}[\-/.]\d{1,2}(?:[\-/.]\d{1,2})?")
# ハイフンで挟まれた英数字コードらしい並び(REFコード等)。
# 「2-3」のような範囲表記のspec値(例: 393-020310のSPEC_1="2-3 mm")を誤って除外
# しないよう、アルファベットを1文字も含まない(=純粋な数字の範囲)場合は対象外にする。
_HYPHENATED_CODE_RE = re.compile(r"[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def _mask_date_and_refcode_spans(s: str) -> str:
    """specの数値抽出対象から、日付らしい並び・REFコードらしい並びをスペースに
    置き換えて除外する(文字数・位置は変えないので、他の処理への影響は無い)。"""
    out = list(s)
    for m in _DATE_LIKE_RE.finditer(s):
        for i in range(m.start(), m.end()):
            out[i] = " "
    for m in _HYPHENATED_CODE_RE.finditer(s):
        if not re.search(r"[A-Za-z]", m.group(0)):
            continue
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out)


# 2026-10-16: REF/商品名/使用期限のスコア計算で実際に使われた範囲を、sizeの
# 採点前にcombinedから除外する(ユーザー提案)。_mask_date_and_refcode_spansが
# 「日付・REFコードっぽい形」を機械的に検出するのに対し、こちらは「このマスク×
# このマスタ品目の組み合わせで、実際にREF/商品名/期限として一致した場所」を
# 直接特定する。両方を併用する。
# 低スコアの偶然の一致まで除外すると実害があるため、スコアがこの値以上の
# ときだけ「実際に使われた」とみなす。
# 2026-10-16、実データでの回帰確認により50点では緩すぎると判明: REFの
# fuzz.partial_ratioがバーコードの無関係な数字列に対してたまたま50点を
# 出し、その範囲をマスクしたことで残った断片("06970908212826"の一部等)が
# 別の品目のspec値と偶然一致する新しい誤マッチが58件発生した。90点まで
# 上げたところ、残った数件は逆に「REFコード直後の数字が、REFの文字列を
# 単位と誤認識して0点になっていたのが、REFをマスクすることで正しく
# 救済される」という正しい修正だった(例: XT275081自身の箱で、"150"の
# 直後の"XT275081"が単位"xt"として誤って拾われていたのが解消された)。
MASK_CONSUME_MIN_SCORE = 90.0


def _consumed_span_for_ref_or_name(
    key: str, combined_norm: str, idx_map: list[int],
) -> tuple[int, int] | None:
    """REF/商品名のfuzz.partial_ratioで実際に一致した範囲を、combined(正規化前)
    での位置として返す。一致が無い/短すぎる場合はNone。"""
    key_norm = _normalize(key)
    if min(len(key_norm), len(combined_norm)) < MIN_KEY_TEXT_LEN_FOR_MATCH:
        return None
    align = fuzz.partial_ratio_alignment(key_norm, combined_norm)
    if align is None or align.dest_end <= align.dest_start:
        return None
    orig_start = idx_map[align.dest_start]
    orig_end = idx_map[align.dest_end - 1] + 1
    return orig_start, orig_end


def _consumed_span_for_date(key: str, combined: str) -> tuple[int, int] | None:
    """日付の採点で実際に最も高いスコアを出した候補の、combined内の位置を返す
    (_key_scoreのis_date分岐と同じロジックで候補を探す)。"""
    k = _digits_only(key)
    if len(k) < 6:
        return None
    combined_for_date = re.sub(r"\s*[年月日]\s*", "-", combined)
    if len(combined_for_date) != len(combined):
        # 年月日の置換で長さが変わると位置が合わなくなるため、その場合は
        # 置換前の文字列で候補を探す(年月日混在の日付はここでは拾えなくなるが、
        # 安全側に倒す)。
        combined_for_date = combined
    best_span, best_score = None, -1.0
    for m in re.finditer(r"\d[\d\-/.]{5,}", combined_for_date):
        c = _digits_only(m.group(0))
        if len(c) < 6:
            continue
        score = fuzz.ratio(k, c)
        if score > best_score:
            best_score, best_span = score, (m.start(), m.end())
    return best_span


def _mask_spans(s: str, spans: list[tuple[int, int]]) -> str:
    """spansで指定された範囲をスペースに置き換える(文字数・位置は変えない)。"""
    if not spans:
        return s
    out = list(s)
    n = len(out)
    for start, end in spans:
        for i in range(max(0, start), min(end, n)):
            out[i] = " "
    return "".join(out)


def _score_keys(keys_of_entry: list[tuple[str, str, bool, bool]], combined: str) -> list[float]:
    """1つの(マスク, マスタ品目)ペアについて、_keys_of()の5項目を順番に採点する。

    REF→商品名→使用期限→spec_1→spec_2 の順(_keys_of()の並びそのもの)で処理し、
    REF/商品名/使用期限で実際に使われた範囲(スコアがMASK_CONSUME_MIN_SCORE以上の
    場合のみ)を、それより後のspec採点のcombinedから除外していく。
    """
    combined_norm, idx_map = _normalize_with_map(combined)
    consumed_spans: list[tuple[int, int]] = []
    scores: list[float] = []
    for _, key, is_date, is_numeric in keys_of_entry:
        if is_numeric:
            masked = _mask_spans(combined, consumed_spans)
            scores.append(_key_score(key, masked, is_numeric=True))
            continue

        score = _key_score(key, combined, is_date=is_date, is_numeric=is_numeric)
        scores.append(score)

        if score >= MASK_CONSUME_MIN_SCORE:
            span = (
                _consumed_span_for_date(key, combined)
                if is_date
                else _consumed_span_for_ref_or_name(key, combined_norm, idx_map)
            )
            if span is not None:
                consumed_spans.append(span)
    return scores


def _key_score(key: str, combined: str, *, is_date: bool = False, is_numeric: bool = False) -> float:
    """
    1キーに対するスコア（0〜100）。

    日付は素朴に数字だけ比較すると、型番・寸法・LOTの数字に当たって
    ほぼ何にでも一致してしまう（検証時に全件が閾値を超えた）。
    そのため combined 側を「日付らしいトークン」に絞ってから比較する。
    """
    if not key:
        return 0.0

    if is_date:
        k = _digits_only(key)
        if len(k) < 6:  # '2029-02' のような不完全な日付はキーにしない
            return 0.0
        # 2026-09-11: 「2028年 10月」のような漢字区切り(空白混在)の日付は、
        # 年/月/日で数字の連続が途切れてしまい、そのままではこの下の正規表現が
        # 候補として拾えなかった(常に0点になっていた)。日付候補抽出専用の
        # 一時変換として、年/月/日とその前後の空白だけをまとめてハイフンに
        # 置換してから抽出する(combined自体やref/display_name/spec側のロジックには
        # 影響しない)。空白を無差別に消すと隣接する別トークン(REF末尾の数字等)が
        # 日付候補に混入してしまうため、年/月/日に隣接する空白だけを対象にする。
        combined_for_date = re.sub(r"\s*[年月日]\s*", "-", combined)
        cands = [_digits_only(t) for t in re.findall(r"\d[\d\-/.]{5,}", combined_for_date)]
        cands = [c for c in cands if len(c) >= 6]
        if not cands:
            return 0.0
        return float(max(fuzz.ratio(k, c) for c in cands))

    if is_numeric:
        # 2026-08-25試験導入: SPEC_1/SPEC_2("1.5 mm"、"2 cm"等)は数値そのものに
        # 意味があり、fuzz.partial_ratioでは"2cm"と"3cm"のような1文字違いの短い
        # 文字列を区別できない(実データで確認、OPTIMA系列で誤認識の主因になった)。
        # 数値部分を抜き出して一致判定する。単位まで一致すれば満点、片側の単位が
        # 読めていない場合は減点、単位が明確に食い違う場合は不一致とみなす。
        key_pairs = _extract_value_unit_pairs(key)
        if not key_pairs:
            return 0.0
        # 2026-10-16: 日付・REFコードらしい並びに含まれる数字を候補から除外してから
        # 抽出する(_mask_date_and_refcode_spans参照)。keyはマスタの信頼できる値
        # なのでマスクしない。
        combined_pairs = _extract_value_unit_pairs(_mask_date_and_refcode_spans(combined))
        if not combined_pairs:
            return 0.0
        best = 0.0
        for kv, ku in key_pairs:
            for cv, cu in combined_pairs:
                if kv != cv:
                    continue
                if ku and cu:
                    if ku == cu:
                        best = max(best, 100.0)
                    # 単位が両方読めていて食い違う場合は別の数値とみなし加点しない
                else:
                    best = max(best, 80.0)  # 片側の単位が読めていない場合はやや減点
        return best

    key_norm = _normalize(key)
    combined_norm = _normalize(combined)
    # partial_ratioは短い方の文字列が長い方のどこかに偶然含まれるだけで満点になり
    # うるため、key/combinedどちらが短くても最低文字数を満たさなければ信用しない。
    if min(len(key_norm), len(combined_norm)) < MIN_KEY_TEXT_LEN_FOR_MATCH:
        return 0.0
    return float(fuzz.partial_ratio(key_norm, combined_norm))


# ===== マスク・OCRの前処理 =====

def _mask_to_binary(mask, h: int, w: int) -> np.ndarray:
    """本番 only_one_tilted._mask_to_binary と同じ意図の正規化。"""
    b = (np.asarray(mask) > 0).astype(np.uint8)
    if b.ndim > 2:
        b = np.squeeze(b)
    if b.ndim != 2:
        raise ValueError(f"mask の次元が不正です: {b.shape}")
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST)
    return b


def _mask_bbox(mask_bin: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return None
    return {
        "x1": float(xs.min()), "y1": float(ys.min()),
        "x2": float(xs.max()), "y2": float(ys.max()),
    }


def _bbox_overlap(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    """2つの矩形(x1,y1,x2,y2)が重なるか。重ならなければ実ピクセルの重なりも必ず0。"""
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def _poly_mask_overlap_ratio(
    poly_pts: np.ndarray, mask_bin: np.ndarray, h: int, w: int,
    poly_bbox: tuple[float, float, float, float] | None = None,
) -> float:
    """OCR文字ポリゴンの面積のうち、実際のマスク輪郭(ピクセル単位)と重なる割合。

    2026-08-21: 従来は矩形バウンディングボックス同士の重なりで判定していたが、
    棚の箱は微妙に傾いて立っているため隣接マスクの矩形が重なり合い、隣の箱の
    文字が誤って割り当てられる実例が確認された(query=ESC0305で隣のAXS Vecta 46
    DACのテキストバケットに"ESC0305"が混入し、confident=Trueで誤選択)。矩形では
    なく実際のマスク輪郭との重なりで判定することで、傾いた隣接マスク同士が
    矩形上は重なっていても実体(輪郭)は重ならない場合に正しく区別できる。

    2026-10-16: 画像全体(h×w)ではなく、文字ポリゴンの外接矩形の範囲だけで計算する
    (性能改善、pro-book-aa提案)。ポリゴンはその外接矩形の外には絶対に出ないため、
    範囲を切り出しても結果は変わらない(整数だけの平行移動なのでnp.roundの結果も
    不変)。呼び出し側で外接矩形を既に計算済みならpoly_bboxで渡すと再計算を省ける。
    """
    if poly_bbox is not None:
        x1, y1, x2, y2 = poly_bbox
    else:
        x1, y1 = float(poly_pts[:, 0].min()), float(poly_pts[:, 1].min())
        x2, y2 = float(poly_pts[:, 0].max()), float(poly_pts[:, 1].max())
    cx1, cy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    cx2, cy2 = min(w, int(np.ceil(x2)) + 1), min(h, int(np.ceil(y2)) + 1)
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    canvas = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
    pts_shifted = poly_pts.copy()
    pts_shifted[:, 0] -= cx1
    pts_shifted[:, 1] -= cy1
    pts_int = np.round(pts_shifted).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts_int], 1)
    poly_area = int(canvas.sum())
    if poly_area <= 0:
        return 0.0
    mask_crop = mask_bin[cy1:cy2, cx1:cx2]
    inter = int(np.count_nonzero((canvas > 0) & (mask_crop > 0)))
    return inter / poly_area


def _unrotate_poly(poly, angle: int, w: int, h: int) -> np.ndarray:
    """本番 unrotate_poly_to_original と同一の変換。"""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    a = int(angle) % 360
    if a == 90:
        xo, yo = y, (h - 1) - x
    elif a == 180:
        xo, yo = (w - 1) - x, (h - 1) - y
    elif a == 270:
        xo, yo = (w - 1) - y, x
    else:
        xo, yo = x, y
    return np.stack([xo, yo], axis=1)


# 2026-10-16: 数値の直後に来るべき単位として扱う、短い文字列のホワイトリスト。
# マスタのSPEC_1/SPEC_2に実際に出てくる単位に絞る(例: "0.046 in"、"3.2/3.4Fr."等)。
# 範囲を広げすぎると("TM"のような無関係な短い英字も該当しうる)誤爆のリスクが
# 増えるため、意図的に狭く絞ってある。
_UNIT_WHITELIST = {"mm", "cm", "in", "fr", "f"}


def _looks_like_bare_value(text: str) -> bool:
    """単位を伴わない、数字(整数/小数)だけの文字列か。"""
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", text.strip()))


def _looks_like_bare_unit(text: str) -> bool:
    """_UNIT_WHITELISTに含まれる、単位だけの短い文字列か。"""
    return text.strip().rstrip(".").lower() in _UNIT_WHITELIST


def _order_bucket_texts(items: list[tuple[float, float, float, str]]) -> str:
    """1つのマスクに帰属したOCR文字を、読み順に並べて結合する。

    2026-10-16追加: 数値と単位が別々のOCR文字列として離れて検出され、読み順
    (Y座標)がたまたま遠く離れてしまう問題への対応
    (詳細はHANDOFF_20261012_ocr_two_row_order_todo.md)。

    当初は「2段ラベル全体をX座標で行分けする」案を試したが、実データ
    (10CSW01515)で検証したところ、背表紙が細いため本体の品名文字列も
    spec表の数字もほぼ同じX範囲に収まってしまい、行として切り分けられなかった
    (2026-10-16、実データでの検証により判明)。そのため、2段表全体を解析する
    のではなく、**数値のすぐ近く(Y座標がとても近い)に単位らしい短い文字列が
    あれば、その2つだけをペアにして隣接させる**という、局所的な対策に絞った。

    手順:
    1. 通常通りY座標降順で全文字列を並べる(既存の挙動そのまま)
    2. 数字だけの文字列(値)と、単位らしい短い文字列(_UNIT_WHITELIST)を洗い出す
    3. Y座標が、両者の文字の高さから見て明らかに近い(他の単語同士の間隔より
       ずっと近い)もの同士を、距離が近い順に貪欲に1対1でペアにする
    4. ペアになった単位は元の位置から外し、対応する値の直後に挿入し直す

    1行のみの通常ケースや、対象となる値・単位の組が無いケースでは、
    これまでと全く同じ挙動になる。
    """
    if len(items) <= 1:
        return " ".join(t for *_, t in items)

    # 旧実装は(-Y, text)のタプルをsortedに渡していたため、Yが同値の場合は
    # textの辞書順でタイブレークされていた。挙動を完全に保つため、ここでも
    # 同じタイブレーク規則にする(単にkey=...,reverse=Trueにすると、Y同値時に
    # OCR検出順でタイブレークされてしまい、挙動が変わる)。
    ordered = sorted(items, key=lambda it: (-it[2], it[3]))

    value_idxs = [i for i, it in enumerate(ordered) if _looks_like_bare_value(it[3])]
    unit_idxs = [i for i, it in enumerate(ordered) if _looks_like_bare_unit(it[3])]

    candidates = []
    for vi in value_idxs:
        v_y, v_h = ordered[vi][2], ordered[vi][1]
        for ui in unit_idxs:
            u_y, u_h = ordered[ui][2], ordered[ui][1]
            dist = abs(v_y - u_y)
            threshold = (v_h + u_h) / 2.0 * 0.5
            if dist < threshold:
                candidates.append((dist, vi, ui))
    candidates.sort(key=lambda c: c[0])

    pair_for_value: dict[int, int] = {}
    used_values: set[int] = set()
    used_units: set[int] = set()
    for dist, vi, ui in candidates:
        if vi in used_values or ui in used_units:
            continue
        pair_for_value[vi] = ui
        used_values.add(vi)
        used_units.add(ui)

    skip_idxs = set(pair_for_value.values())
    parts: list[str] = []
    for i, it in enumerate(ordered):
        if i in skip_idxs:
            continue
        parts.append(it[3])
        if i in pair_for_value:
            parts.append(ordered[pair_for_value[i]][3])
    return " ".join(parts)


def _collect_mask_texts(
    ocr_json_path: Path,
    masks: list,
    rgb_path: Path,
    forced_angle: int = FORCED_ANGLE,
) -> tuple[list[str], list[dict | None], list[dict]]:
    """
    OCR文字を各マスクへ割り当て、マスクごとの結合テキストを作る。

    重要: 文字は「読み順」に並べて結合する。
    fuzz.partial_ratio は連続部分列を見るため、並びが崩れると
    'pNOVUS17' と '150' が離れただけでスコアが落ちる（実測 80.0 → 66.7）。
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polys = data.get("dt_polys", [])
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [1.0] * len(texts))

    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(f"画像が読めませんでした: {rgb_path}")
    h, w = img.shape[:2]

    mask_bins = [_mask_to_binary(m, h, w) for m in masks]
    boxes = [_mask_bbox(mb) for mb in mask_bins]

    buckets: list[list[tuple[float, float, float, str]]] = [[] for _ in masks]
    debug: list[dict] = []

    for idx, (poly, text, sc) in enumerate(zip(polys, texts, scores), start=1):
        text = (text or "").strip()
        if not text or float(sc) < MIN_REC_SCORE:
            continue

        p = _unrotate_poly(poly, forced_angle, w=w, h=h)
        x1, y1 = float(p[:, 0].min()), float(p[:, 1].min())
        x2, y2 = float(p[:, 0].max()), float(p[:, 1].max())
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue

        # 文字ポリゴンの面積のうち、どのマスクの実際の輪郭(ピクセル単位)に
        # 最も多く重なるかで帰属を決める(矩形バウンディングボックスではない。
        # 理由は_poly_mask_overlap_ratioのdocstring参照)。
        # 2026-10-16: マスクの外接矩形(boxes、384行目で計算済み)と文字の外接矩形が
        # そもそも重ならない組は、実ピクセルの重なりも必ず0なので計算自体を省く
        # (性能改善、pro-book-aa提案)。
        best_i, best_ratio = None, 0.0
        for i, mb in enumerate(mask_bins):
            box = boxes[i]
            if box is None:
                continue
            if not _bbox_overlap(x1, y1, x2, y2, box["x1"], box["y1"], box["x2"], box["y2"]):
                continue
            ratio = _poly_mask_overlap_ratio(p, mb, h, w, poly_bbox=(x1, y1, x2, y2))
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i

        if best_i is not None and best_ratio > 0:
            # 読み順。
            # 本番のOCR入力は rotate90CW(after_init_rgb) なので base_x = h-1-y。
            # つまり base 上の左→右は、本番画像座標では y の降順にあたる(行内の順)。
            # 元画像のX座標は「どの行か」に対応する(_order_bucket_texts参照)。
            x_center = (x1 + x2) / 2.0
            x_height = x2 - x1
            y_avg = (y1 + y2) / 2.0
            buckets[best_i].append((x_center, x_height, y_avg, text))
            debug.append({
                "ocr_index": idx, "text": text,
                "matched": f"mask_{best_i + 1}", "overlap_ratio": round(best_ratio, 3),
            })
        else:
            debug.append({"ocr_index": idx, "text": text, "matched": None, "overlap_ratio": 0.0})

    combined = [_order_bucket_texts(b) for b in buckets]
    return combined, boxes, debug


# ===== マスタ =====

def _load_master(master_json: Path) -> list[dict]:
    with open(master_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"マスタはリスト形式である必要があります: {master_json}")
    return data


def _keys_of(entry: dict) -> list[tuple[str, str, bool, bool]]:
    """(キー種別, 文字列, 日付か, 数値専用比較か) の一覧。

    spec_1/spec_2(2026-08-25試験導入、2026-08-25に数値専用比較へ変更): 箱に印字された
    数値+単位のスペック値(例 "10 mm"、"150 cm")。当初はREF/display_nameと同じ
    「combined全体へのあいまい一致(fuzz.partial_ratio)」に乗せていたが、"2cm"と"3cm"の
    ような1文字違いの短い文字列を区別できず、似た品目同士(OPTIMA系列等)で誤認識の
    主因になったため、数値部分を抜き出して比較する専用ロジック(_key_scoreのis_numeric)
    に切り替えた。スペックが1つしかない品目はspec_2が空文字になり、_key_scoreが0点を
    返すだけで他キーの邪魔をしない(dateキーと同じ後方互換の考え方)。
    """
    return [
        ("ref", entry.get("book_name", "") or "", False, False),
        ("display_name", entry.get("display_name", "") or "", False, False),
        ("date", entry.get("expiration date", "") or "", True, False),
        ("spec_1", entry.get("SPEC_1", "") or "", False, True),
        ("spec_2", entry.get("SPEC_2", "") or "", False, True),
    ]


# ===== 割当 =====

def _greedy_assign(S: np.ndarray) -> dict[int, int]:
    """scipy が無い場合のフォールバック（スコアの高い順に確定させる）。"""
    n_mask, n_master = S.shape
    order = np.argsort(-S, axis=None)
    used_m, used_j, out = set(), set(), {}
    for flat in order:
        i, j = divmod(int(flat), n_master)
        if i in used_m or j in used_j:
            continue
        out[j] = i
        used_m.add(i)
        used_j.add(j)
        if len(out) >= min(n_mask, n_master):
            break
    return out


# ===== 可視化 =====

def _save_assignment_overview(
    shot_dir: Path,
    rgb_path: Path,
    masks: list,
    boxes: list[dict | None],
    mask_to_item: dict[int, dict],
    out_filename: str = "ocr_overlay_all_assignments.png",
) -> None:
    """全マスク x 全品目の割当結果を、1枚の画像にまとめて可視化する
    (2026-08-26試験導入)。個別品目ごとの画像(ocr_overlay_assign_*.png)だと
    枚数が多く全体像が見えないとの要望に対応し、1枚で「どのマスクがどの品目に
    紐付いたか」を一望できるようにした。

    ocr_overlay_perkey_top1.pngと同じく、画像はOCRの処理向き(rotate90CW、
    箱の背表紙文字が横書きで読める向き)に回転して表示し、右側に余白パネルを
    作ってそこへ品目名の一覧を載せる(画像に直接ラベルを重ねると、隣接する
    棚の箱同士でラベルが重なって読めなくなるため)。

    2026-09-29修正(ユーザー要望): 矩形の枠線ではなく、sam3_all_masks_overlay.png
    と同じく実際のマスク形状(画素単位)を半透明で塗りつぶす方式に変更した。
    軸並行bbox(boxes)だと、斜めや不整形なマスクの実際の輪郭が分からないため。
    色分け(クエリ=赤、各品目=パレット色)は維持し、割当の無いマスクは塗らずに
    輪郭線だけ描く(従来のbbox枠と同じ「未割当=薄い印」という役割だが、実マスクの
    輪郭に変えた)。

    masks: match_text_to_mask_main に渡された生のマスク一覧(boxesと同じ並び)。
    mask_to_item: {mask_index(0始まり): {"book_name", "is_query", "score"}}
    """
    try:
        img = cv2.imread(str(rgb_path))
        if img is None:
            return
        h, w = img.shape[:2]
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        overlay = rotated.copy()

        # (0,0,255)の赤はQUERY専用色として予約するため、通常品目のパレットには含めない。
        palette = [
            (255, 60, 0), (0, 160, 0), (200, 0, 200), (0, 160, 200),
            (0, 128, 255), (128, 0, 255), (0, 200, 120), (255, 0, 120), (120, 120, 0),
            (0, 100, 200),
        ]

        def to_rotated(x: float, y: float) -> tuple[int, int]:
            # _unrotate_poly(angle=90)の逆変換(元画像座標->OCR処理向き座標)。
            return int((h - 1) - y), int(x)

        rows = []  # (i, item, rx1, ry1, rx2, ry2) 番号ラベルの位置・一覧の並び順に使う
        for i, box in enumerate(boxes):
            if box is None:
                continue
            corners = [
                to_rotated(box["x1"], box["y1"]), to_rotated(box["x2"], box["y1"]),
                to_rotated(box["x1"], box["y2"]), to_rotated(box["x2"], box["y2"]),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)

            # 実マスクを、背景画像と同じ向き(rotate90CW)に揃えてから使う。
            mask_bin = _mask_to_binary(masks[i], h, w)
            mask_rot = cv2.rotate(mask_bin, cv2.ROTATE_90_CLOCKWISE).astype(bool)

            item = mask_to_item.get(i)
            if item is None:
                # 未割当: 塗りつぶさず、マスクの輪郭線だけ描く。
                contours, _ = cv2.findContours(
                    mask_rot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (120, 120, 120), 1)
                continue
            color = (0, 0, 255) if item["is_query"] else palette[i % len(palette)]
            alpha = 0.5
            overlay[mask_rot] = (
                (1 - alpha) * overlay[mask_rot].astype(np.float32)
                + alpha * np.asarray(color, dtype=np.float32)
            ).astype(np.uint8)
            # マスク番号だけは画像上にも小さく振っておく(余白側の一覧と対応付けるため)。
            # 位置はbboxの左上を使う(塗りは実マスク形状、ラベル位置だけbbox基準で従来通り)。
            cv2.putText(overlay, str(i + 1), (rx1 + 4, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
            rows.append((i, item, rx1, ry1, rx2, ry2))
        rotated = overlay

        # 余白パネル(右側)に品目一覧を書く。画像上の並び順(x座標=元画像の上下方向)に揃える。
        rows.sort(key=lambda t: t[2])
        panel_w = 480
        line_h = 26
        panel_h = max(rotated.shape[0], line_h * (len(rows) + 2) + 20)
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "mask -> assigned item", (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y0 = 60
        for i, item, *_ in rows:
            color = (0, 0, 255) if item["is_query"] else palette[i % len(palette)]
            cv2.circle(panel, (22, y0 - 6), 10, color, -1)
            cv2.putText(panel, str(i + 1), (17, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            score_txt = "" if item["score"] is None else f"  score={item['score']:.1f}"
            line = f"mask_{i + 1}: {item['book_name']}{score_txt}"
            if item["is_query"]:
                line += "  [QUERY]"
            cv2.putText(panel, line, (42, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += line_h

        if rotated.shape[0] < panel_h:
            pad = np.zeros((panel_h - rotated.shape[0], rotated.shape[1], 3), dtype=np.uint8)
            rotated = np.vstack([rotated, pad])
        out = np.hstack([rotated, panel])

        shot_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(shot_dir / out_filename), out)
    except Exception as e:
        print(f"[multikey] 割当一覧画像の保存に失敗（処理は継続）: {e}")


def _save_perkey_overlay(
    shot_dir: Path,
    rgb_path: Path,
    ocr_json_path: Path,
    ocr_debug: list[dict],
    sel_mask_name: str,
    keys_of_target: list[tuple[str, str, bool]],
    out_filename: str = "ocr_overlay_perkey_top1.png",
    panel_label: str = "selected(top1)",
    mask=None,
) -> None:
    """指定マスク(sel_mask_name)に帰属するOCR検出のうち、キーごとに最もスコアが高かった
    1件だけを切り出して可視化する（multikey_match_debug.jsonの数値だけでは
    「勝った検出」が視覚的に分かりにくいため）。失敗しても認識処理は継続する。

    out_filenameを変えることで、採用マスク(top1)だけでなく次点マスク(top2)にも
    流用できる(2026-08-26試験導入: 全体最適割当(ハンガリー法)が正しく機能しているか
    ―採用されなかったマスクのOCRが本当に別品目のものかを目視で確認したいとの要望)。

    mask: sel_mask_nameの実マスク(画素配列、maskはimgと同じ元画像の向き)。渡すと、
    検出文字列のボックスだけでなく、その文字列が帰属しているマスクの輪郭も黄色線で
    重ねて描く(2026-09-29追加、ユーザー要望: 「文字列boxだけでなく、文字列が帰属される
    マスクも載せてほしい」。マスクが複数の箱にまたがって融合していないか等の確認に使う)。
    Noneなら従来通りマスクの輪郭は描かない。
    """
    try:
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        polys = ocr_data.get("dt_polys", [])

        img = cv2.imread(str(rgb_path))
        if img is None:
            return
        h, w = img.shape[:2]

        matched = [a for a in ocr_debug if a["matched"] == sel_mask_name]
        if not matched:
            return

        palette = [(0, 0, 255), (255, 60, 0), (0, 160, 0), (200, 0, 200), (0, 160, 200)]
        winners: list[tuple[str, float, dict]] = []
        for kname, ktext, is_date, is_numeric in keys_of_target:
            if not ktext:
                continue
            scored = sorted(
                ((_key_score(ktext, a["text"], is_date=is_date, is_numeric=is_numeric), a) for a in matched),
                key=lambda t: -t[0],
            )
            score, a = scored[0]
            winners.append((kname, score, a))
        if not winners:
            return

        boxes_px: dict[str, np.ndarray] = {}
        for kname, _score, a in winners:
            idx = a["ocr_index"]
            if idx - 1 >= len(polys):
                continue
            boxes_px[kname] = _unrotate_poly(polys[idx - 1], FORCED_ANGLE, w=w, h=h)
        if not boxes_px:
            return

        all_pts = np.concatenate(list(boxes_px.values()), axis=0)
        x1, y1 = all_pts[:, 0].min(), all_pts[:, 1].min()
        x2, y2 = all_pts[:, 0].max(), all_pts[:, 1].max()
        margin = 130
        cx1, cy1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
        cx2, cy2 = min(w, int(x2 + margin)), min(h, int(y2 + margin))

        crop = img[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return
        scale = 3
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if mask is not None:
            mask_bin = _mask_to_binary(mask, h, w)
            mask_crop = mask_bin[cy1:cy2, cx1:cx2]
            if mask_crop.size:
                mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(
                    mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(crop, contours, -1, (0, 255, 255), 4)  # 黄色=帰属先マスクの輪郭

        for i, (kname, _score, _a) in enumerate(winners, start=1):
            if kname not in boxes_px:
                continue
            color = palette[(i - 1) % len(palette)]
            pp = ((boxes_px[kname] - [cx1, cy1]) * scale).astype(np.int32)
            cv2.polylines(crop, [pp], isClosed=True, color=color, thickness=3)
            cx, cy = int(pp[:, 0].mean()), int(pp[:, 1].min())
            cv2.circle(crop, (cx, cy - 18), 16, color, -1)
            cv2.putText(crop, str(i), (cx - 8, cy - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        legend_h = 34 * (len(winners) + 1) + 20
        panel = np.zeros((legend_h, crop.shape[1], 3), dtype=np.uint8)
        title = f"mask={sel_mask_name} [{panel_label}]  (per-key top-1 OCR match)"
        if mask is not None:
            title += "  / yellow outline = mask shape"
        cv2.putText(panel, title, (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y0 = 60
        for i, (kname, score, a) in enumerate(winners, start=1):
            color = palette[(i - 1) % len(palette)]
            cv2.circle(panel, (24, y0 - 7), 14, color, -1)
            cv2.putText(panel, str(i), (18, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            line = f"{kname}: \"{a['text']}\"  score={score:.1f}"
            cv2.putText(panel, line, (52, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += 34

        out = np.vstack([crop, panel])
        shot_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(shot_dir / out_filename), out)
    except Exception as e:
        print(f"[multikey] per-keyオーバーレイの保存に失敗（処理は継続）: {e}")


def _save_black_placeholder(path: Path, size: tuple[int, int] = (900, 500)) -> None:
    """対応付けされなかった品目のうち、足切りしなければ選ばれていたはずのマスクが
    分からない場合の最終手段としての真っ暗な画像(2026-09-29追加。2026-09-30、
    通常は_save_would_be_overlayに置き換えたので、それも失敗した場合のみ使う)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((size[1], size[0], 3), dtype=np.uint8))


def _save_would_be_overlay(path: Path, rgb_path: Path, mask, would_be_mask_name: str,
                            would_be_score: float | None) -> bool:
    """該当なし(足切り)になった品目について、足切りしなければ選ばれていたはずの
    マスクを灰色で塗りつぶした画像を書き出す(2026-09-30追加、ユーザー要望:
    「該当なしとされても、足切りしなければ選ぶはずのマスクがあるはず。それが分かる
    ようにしてほしい」)。_save_perkey_overlayと同じくマスク周辺を拡大して見せる。
    成功したらTrue、失敗(マスクの面積が0等)したらFalseを返す(呼び出し側で
    _save_black_placeholderにフォールバックするため)。
    """
    try:
        img = cv2.imread(str(rgb_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        mask_bin = _mask_to_binary(mask, h, w)
        box = _mask_bbox(mask_bin)
        if box is None:
            return False

        margin = 60
        cx1, cy1 = max(0, int(box["x1"] - margin)), max(0, int(box["y1"] - margin))
        cx2, cy2 = min(w, int(box["x2"] + margin)), min(h, int(box["y2"] + margin))
        crop = img[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return False
        scale = 3
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        mask_crop = mask_bin[cy1:cy2, cx1:cx2]
        mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = np.array([150, 150, 150], dtype=np.float32)
        alpha = 0.55
        crop[mask_crop] = (
            (1 - alpha) * crop[mask_crop].astype(np.float32) + alpha * gray
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (0, 0, 255), 4)

        panel = np.zeros((70, crop.shape[1], 3), dtype=np.uint8)
        score_txt = f"score={would_be_score:.1f}(しきい値未満で該当なし)" if would_be_score is not None else "score=?"
        cv2.putText(panel, f"該当なし - 足切りしなければ選ばれていたマスク: {would_be_mask_name}  {score_txt}",
                    (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        out = np.vstack([crop, panel])

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), out)
        return True
    except Exception as e:
        print(f"[multikey] would-beオーバーレイの保存に失敗（処理は継続）: {e}")
        return False


def _save_all_perkey_overlays(
    shot_dir: Path,
    rgb_path: Path,
    ocr_json_path: Path,
    ocr_debug: list[dict],
    all_assignments: list[dict],
    master: list[dict],
    masks: list,
    images_dirname: str = "images",
) -> None:
    """マスタの全品目について、選択マスクを強調した画像を images/<品目名>.png として
    書き出す(2026-09-29追加、ユーザー要望)。

    ocr_overlay_all_assignments.png(全品目を1枚に詰め込む図)は「どれがどれだか分からない」
    との指摘を受け、_save_perkey_overlayと同じ形式(該当箇所を拡大＋キーごとの凡例)の画像を
    品目ごとに1枚ずつ作れるようにした。masksを渡すことで、検出文字列のボックスに加え、
    帰属先マスクの輪郭(黄色)も重ねて描く。

    対応付けされなかった品目は、足切りしなければ選ばれていたはずのマスク(all_assignmentsの
    would_be_mask)を灰色で塗りつぶした画像にする(2026-09-30追加、ユーザー要望)。
    would_be_maskが無い(そもそも候補マスクが無い等)場合のみ、従来通り真っ暗な画像にする。
    """
    images_dir = shot_dir / images_dirname
    images_dir.mkdir(parents=True, exist_ok=True)
    for a, m in zip(all_assignments, master):
        name = a["book_name"] or a["display_name"] or "item"
        filename = f"{_safe_name(name)}.png"
        out_path = images_dir / filename
        if a["assigned_mask"] is None:
            ok = False
            would_name = a.get("would_be_mask")
            if would_name:
                would_idx = int(would_name.rsplit("_", 1)[1]) - 1
                if 0 <= would_idx < len(masks):
                    ok = _save_would_be_overlay(
                        out_path, rgb_path, masks[would_idx], would_name, a.get("would_be_score")
                    )
            if not ok:
                _save_black_placeholder(out_path)
            continue
        mask_idx = int(a["assigned_mask"].rsplit("_", 1)[1]) - 1
        _save_perkey_overlay(
            shot_dir=images_dir,
            rgb_path=rgb_path,
            ocr_json_path=ocr_json_path,
            ocr_debug=ocr_debug,
            sel_mask_name=a["assigned_mask"],
            keys_of_target=_keys_of(m),
            out_filename=filename,
            panel_label=name,
            mask=masks[mask_idx] if 0 <= mask_idx < len(masks) else None,
        )
        if not out_path.exists():
            # 該当OCRが見つからない等でこの関数が黙って書き出さないことがある。
            # 品目ごとに必ず1枚出すため、その場合も真っ暗な画像で埋める。
            _save_black_placeholder(out_path)


# ===== 本番互換の入口 =====

def match_text_to_mask_main(
    query: str,
    masks,
    shot_dir,
    threshold: int = 40,
    *,
    master_json: str | Path | None = None,
    use_hungarian: bool | None = None,
    use_multikey: bool | None = None,
    save_all_assignments: bool = False,
    save_all_perkey_overlays: bool = False,
) -> list[dict[str, Any]]:
    """
    本番 only_one_tilted.match_text_to_mask_main と同じ入出力。

    query はマスタの book_name（REF）を想定する。
    マスタに query が無い場合は、多段queryが使えないため
    query 1本のみで採点する（現行と同じ挙動）。

    save_all_assignments: Trueにすると、全マスクの矩形とHungarian割当の結果を
    1枚にまとめた画像(ocr_overlay_all_assignments.png)を書き出す
    (2026-08-26試験導入)。Hungarian割当は実際には1回の認識でマスタ全品目×
    全マスクの最適割当を一括で解いており、queryの品目だけでなく他の品目にも
    それぞれマスクが割り当たっている。「query以外の品目も正しく別マスクへ
    選べているか」を一望で目視確認したいとの要望に対応。既定はFalse
    (数値情報はmultikey_match_debug.jsonのall_assignmentsに常に出力するので、
    画像無しでも確認可能)。

    save_all_perkey_overlays: Trueにすると、マスタの全品目について、
    shot_dir/images/<品目名>.png を1枚ずつ書き出す(2026-09-29追加)。
    ocr_overlay_all_assignments.pngは1枚に全品目を詰め込むため「どれがどれだか
    分からない」との指摘を受け、_save_perkey_overlayと同じ形式(選択箇所の拡大＋
    キーごとの凡例)を品目ごとに分けた。対応付けされなかった品目は真っ暗な画像になる。
    既定はFalse。
    """
    shot_dir = Path(shot_dir)
    ocr_json_path = shot_dir / "ocr_result.json"
    rgb_path = shot_dir / "after_init_rgb.png"

    combined, boxes, ocr_debug = _collect_mask_texts(ocr_json_path, masks, rgb_path)
    # 明示指定が無ければ、環境変数 MULTIKEY_LEGACY による既定を使う
    if use_multikey is None:
        use_multikey = not LEGACY_MODE
    if use_hungarian is None:
        use_hungarian = not LEGACY_MODE
    if LEGACY_MODE:
        print("[multikey] 従来手法モード: query(REF)1本のみ・独立argmax")

    n_mask = len(masks)

    master_path = Path(master_json) if master_json else DEFAULT_MASTER_JSON
    if not use_multikey:
        # 現行方式の再現用: query(REF) 1本だけで採点する
        master_path = Path("(disabled)")
        master = []
    else:
        try:
            master = _load_master(master_path)
        except Exception as e:
            print(f"[multikey] マスタを読めませんでした（query単独にフォールバック）: {e}")
            master = []

    # query に対応するマスタ行を探す
    q_norm = _normalize(query)
    target_j = next(
        (j for j, m in enumerate(master) if _normalize(m.get("book_name", "")) == q_norm),
        None,
    )
    if target_j is None:
        master = [{"book_name": query, "display_name": "", "expiration date": ""}]
        target_j = 0
        print(f"[multikey] query '{query}' はマスタに無いため query 単独で採点します")

    n_master = len(master)

    # ===== スコア行列 =====
    # 素朴に max(REF, display_name, 期限) を取ると、識別力の無いキーが勝ってしまう。
    # 実例: query=MC1715000（期限2028-11-27）に対し、別の箱の '2028-11-15' が
    #       fuzz.ratio=75 を出し、正解の箱の display_name=66.7 を上回った。
    # そこで各キーの列から中央値を引いた「その品目らしさの突出度」で比較する。
    # 期限のように全マスクへ一様に高い値を出すキーは、中央値を引くとほぼ0になり、
    # 本当に効いているキーだけが残る。
    n_keys = 5  # ref, display_name, date, spec_1, spec_2(2026-08-25)
    per_key = np.zeros((n_mask, n_master, n_keys), dtype=np.float64)
    for i in range(n_mask):
        c = combined[i]
        for j, m in enumerate(master):
            per_key[i, j, :] = _score_keys(_keys_of(m), c)

    centered = np.zeros_like(per_key)
    for j in range(n_master):
        for k in range(n_keys):
            col = per_key[:, j, k]
            centered[:, j, k] = col - (np.median(col) if col.size else 0.0)

    # win_key/margin/confident等の報告用にcenteredは常に計算しておく(SCORE_COMBINE_METHODに
    # 依らず、採用マスクに対する「どのキーが効いたか」の説明には引き続き使う)。
    if SCORE_COMBINE_METHOD == "raw_sum":
        S = per_key.sum(axis=2)
    else:
        S = centered.max(axis=2)

    # 根拠ゼロ（全キーの生スコアが0）の組み合わせを割当候補から外す。
    # 大きな負値を置くことで、ハンガリー法が他に選択肢を持つ限り選ばなくなる。
    if DROP_ZERO_ASSIGN and n_mask and n_master:
        no_evidence = per_key.max(axis=2) <= 0.0
        if no_evidence.any():
            S = np.where(no_evidence, -1.0e6, S)
            print(f"[multikey] 根拠ゼロの組み合わせ {int(no_evidence.sum())} 件を割当候補から除外")

    # 該当なし足切りに使うスコア行列(REJECT_RULE定義部のコメント参照)。
    # 割当(assign)自体はSで解くが、「該当なしにするかどうか」の判定だけは別の
    # 指標を使えるようにする。既定の"sum"はSそのもの。
    if REJECT_RULE == "ref_display_name":
        reject_S = per_key[:, :, 0] + per_key[:, :, 1]
    else:
        reject_S = S

    # ===== 割当 =====
    if use_hungarian and n_mask > 0 and n_master > 0:
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(-S)
            assign = {int(j): int(i) for i, j in zip(rows, cols)}
        else:
            assign = _greedy_assign(S)
        method = "hungarian" if _HAS_SCIPY else "greedy"
    else:
        assign = {target_j: int(S[:, target_j].argmax())} if n_mask else {}
        method = "independent"

    # 2026-09-24試験導入: しきい値未満の対応を「該当なし」として除外する。
    # REJECT_LOW_SCORE定義部のコメント参照。all_assignments(全品目一括対応表)も
    # このassignを直接参照しているので、ここで除外すれば両方に反映される。
    # 除外前に「どの品目が割当を受けていたか」を控えておく(下のフォールバック判定用)。
    # 値(マスク番号)も含めて丸ごと控える(pre_reject_assign)。該当なしになった品目について
    # 「足切りしなければ選ばれていたはずのマスク」をall_assignmentsに残すのに使う
    # (2026-09-30追加、ユーザー要望)。
    assigned_before_reject = set(assign)
    pre_reject_assign = dict(assign)
    # 足切りに使う値: 環境変数MULTIKEY_REJECT_THRESHOLDがあればそれ、無ければ
    # REJECT_RULEごとの既定値(sumはthreshold引数、ref_display_nameは100.0)。
    reject_th = REJECT_THRESHOLD
    if reject_th is None:
        reject_th = REJECT_RULE_DEFAULT_THRESHOLD[REJECT_RULE]
        if reject_th is None:
            reject_th = threshold
    if REJECT_LOW_SCORE and n_mask and n_master:
        before = len(assign)
        assign = {j: i for j, i in assign.items() if float(reject_S[i, j]) >= reject_th}
        dropped = before - len(assign)
        if dropped:
            print(f"[multikey] {REJECT_RULE}のしきい値{reject_th}未満の対応 {dropped} 件を除外(該当なし扱い)")

    sel_i = assign.get(target_j)
    if sel_i is None and n_mask:
        # 2026-09-25修正: 「割当は受けたがしきい値未満で除外された」品目は、独立argmaxへ
        # 切り替えず該当なしのままにする。argmaxのマスクは、ハンガリー法が他の品目へ
        # 意図的に譲ったマスクなので、そこへ切り替えると、reco/0911の実データ
        # (正解136件)でしきい値200のとき9件が別マスクへ差し替わり誤選択になり得る
        # ことが分かった(切り替えず該当なしにすれば該当なしは10件、差し替えは0件)。
        # 独立argmaxを使うのは、ハンガリー法がそもそも割当を与えなかった品目
        # (マスタ品目数>マスク数のとき、割当から漏れる)だけに限る。
        rejected_by_threshold = REJECT_LOW_SCORE and target_j in assigned_before_reject
        if not rejected_by_threshold:
            candidate = int(S[:, target_j].argmax())
            # REJECT_LOW_SCORE有効時は、このフォールバックにも同じしきい値を適用する。
            # 満たさなければsel_iはNoneのまま(=「この商品は棚に見当たらない」)。
            # resultsが空になり、既存のTargetMaskSelectionErrorへ落ちる。
            if not REJECT_LOW_SCORE or float(reject_S[candidate, target_j]) >= reject_th:
                sel_i = candidate

    # 全体最適割当(ハンガリー法)の効果を検証するため、他品目との競合を無視して
    # このqueryだけで見た場合にどのマスクが最高スコアだったか(independent)も
    # 常に記録しておく(2026-08-28試験導入)。independent != sel_iの場合、
    # 一対一対応によって選択が変わったことを意味する。実際にそれで正解に
    # なったか(目視確認Tか)は、multikey_match_debug.json単体では分からず
    # 別途レビュー結果と突き合わせる必要がある。
    independent_i = int(S[:, target_j].argmax()) if n_mask else None
    hungarian_changed_selection = (
        independent_i is not None and sel_i is not None and independent_i != sel_i
    )

    # ===== 信頼度 =====
    # 報告するスコア/marginは「実際に効いたキー」の生スコアで出す。
    # 中央値を引いた値は比較用の内部量で、そのまま出すと意味が読み取れないため。
    key_names = ["ref", "display_name", "date", "spec_1", "spec_2"]
    col = S[:, target_j]
    if sel_i is not None:
        win_k = int(np.argmax(centered[sel_i, target_j]))
        raw_col = per_key[:, target_j, win_k]
        # 採用マスクが、そのキーで実際に最上位かどうかも見る(参考値。confidentの判定には使わない)
        top = float(raw_col[sel_i])
        others = np.delete(raw_col, sel_i)
        margin = top - (float(others.max()) if others.size else 0.0)
        win_key = key_names[win_k]

        ref_score = float(per_key[sel_i, target_j, 0])
        support_scores = {
            "display_name": float(per_key[sel_i, target_j, 1]),
            "date": float(per_key[sel_i, target_j, 2]),
            "spec_1": float(per_key[sel_i, target_j, 3]),
            "spec_2": float(per_key[sel_i, target_j, 4]),
        }
        support_count = sum(1 for v in support_scores.values() if v >= SUPPORT_KEY_SCORE)
    else:
        top, margin, win_key = 0.0, 0.0, "none"
        ref_score = 0.0
        support_scores = {"display_name": 0.0, "date": 0.0, "spec_1": 0.0, "spec_2": 0.0}
        support_count = 0
        # 2026-09-29修正: sel_i=None(該当なし)でも、per_mask debug出力はcolを使えるように
        # raw_colを定義しておく。未定義のままだと下のデバッグ保存がNameErrorで丸ごと失敗し、
        # 「なぜ該当なしになったか」を確認したい該当なしのケースに限ってdebug jsonが
        # 保存されないという、一番困る形の欠陥になっていた(REJECT_RULE=ref_display_nameの
        # 動作確認中に発見)。win_key="none"なのでキー別の生スコアではなく、col(=Sの列、
        # 割当に使った合成スコア)をそのまま流用する。
        raw_col = col
    selected_text = combined[sel_i] if sel_i is not None else ""
    text_plausible = _looks_like_plausible_identifier(selected_text)

    # REFが完全一致(100点)なら、他キーの結果に関わらず確信ありとする。
    # そうでなければmargin基準ではなく、補助キー(display_name/date/spec_1/spec_2)が
    # 何本高スコアで揃っているかで判断する(理由は REF_EXACT_SCORE 定義部のコメント参照)。
    if ref_score >= REF_EXACT_SCORE:
        confident = True
        confident_reason = "ref_exact_match"
    elif support_count >= SUPPORT_KEY_MIN_COUNT and text_plausible:
        confident = True
        confident_reason = f"support_keys>={SUPPORT_KEY_MIN_COUNT}"
    else:
        confident = False
        confident_reason = "insufficient_evidence"

    # ===== 本番形式へ =====
    results: list[dict[str, Any]] = []
    if sel_i is not None:
        order = [sel_i] + [i for i in np.argsort(-raw_col).tolist() if i != sel_i]
        for i in order:
            sc = float(raw_col[i])
            if i != sel_i and sc <= threshold:
                continue
            results.append({
                "name": f"mask_{i + 1}",
                "score": int(round(sc)),
                "box": boxes[i],
                "forced_angle": FORCED_ANGLE,
            })

    # ===== 全品目の割当結果(2026-08-26試験導入) =====
    # assign は実際には1回の認識でマスタ全品目 x 全マスクの最適割当を一括で
    # 解いた結果であり、target_j(query)だけでなく他の品目にもそれぞれ
    # マスクが割り当たっている。「query以外の品目も正しく別マスクへ選べて
    # いるか」を確認したいとの要望に対応し、全品目分の割当結果を出力する。
    all_assignments = []
    for j, m in enumerate(master):
        i = assign.get(j)
        if i is None:
            # 2026-09-30追加(ユーザー要望): 該当なしでも「足切りしなければ選ばれていた
            # はずのマスク」が分かるようにする。まずHungarian法が実際に割り当てていた
            # マスク(足切りで除外されただけ)を優先し、それも無ければ(マスタ品目数>
            # マスク数で最初から割当が無かった品目)独立argmaxを使う。sel_i算出のロジック
            # (922〜938行目付近)からしきい値チェックだけを除いたもの。
            would_i = pre_reject_assign.get(j)
            if would_i is None and n_mask:
                would_i = int(S[:, j].argmax())
            would_be_mask = f"mask_{would_i + 1}" if would_i is not None else None
            would_be_score = round(float(S[would_i, j]), 1) if would_i is not None else None
            # 2026-09-30追加(ユーザー要望): 該当なしの原因を2種類に区別する。
            # "threshold" = Hungarian法は実際にこの品目へマスクを割り当てていたが、
            #   しきい値未満だったため除外した(=スコアが低いための足切り)。
            # "no_assignment" = Hungarian法が一度も割り当てなかった(マスタ品目数>
            #   マスク数で競合に敗れた、または最初からそのマスクを他品目に取られた)。
            #   この場合、その品目の実物がこの撮影に写っていない可能性が高い。
            reject_reason = "threshold" if j in assigned_before_reject else "no_assignment"
            # 2026-09-30追加(ユーザー要望): 「合計スコアだけじゃ判断できない」ので、
            # would_be_mask(足切りしなければ選ばれていたはずのマスク)についても
            # 5キー内訳を出す。対応付け済みの品目と同じ per_key[i, j, k] を、
            # i=would_iで引くだけ(would_iはsel_i算出ロジックと同じ経路で求めた値)。
            would_be_by_key = (
                {key_names[k]: round(float(per_key[would_i, j, k]), 1) for k in range(n_keys)}
                if would_i is not None else {k: None for k in key_names}
            )
            all_assignments.append({
                "book_name": m.get("book_name", ""),
                "display_name": m.get("display_name", ""),
                "assigned_mask": None,
                "winning_key": None,
                "score": None,
                # 2026-09-29追加: 該当なしの場合は割当マスクが無いため5キー内訳も無い。
                # 他品目と列数を揃えるため、値はNoneで埋める。
                "by_key": {k: None for k in key_names},
                # 2026-09-30追加: 足切りしなければ選ばれていたはずのマスクとそのスコア。
                # このスコアはしきい値未満(=該当なしになった理由)のはず。
                "would_be_mask": would_be_mask,
                "would_be_score": would_be_score,
                "would_be_by_key": would_be_by_key,
                "reject_reason": reject_reason,
                "is_query": (j == target_j),
            })
            continue
        wk_j = int(np.argmax(centered[i, j]))
        all_assignments.append({
            "book_name": m.get("book_name", ""),
            "display_name": m.get("display_name", ""),
            "assigned_mask": f"mask_{i + 1}",
            "winning_key": key_names[wk_j],
            "score": round(float(per_key[i, j, wk_j]), 1),
            # 2026-09-29追加(ユーザー要望): target_j(query)以外の品目もper_key内訳を
            # multikey_match_debug.jsonに残す。以前はtarget_j分のper_mask(全マスク×
            # target_jの5キー)しか無く、他品目の内訳は再現できなかった。
            "by_key": {key_names[k]: round(float(per_key[i, j, k]), 1) for k in range(n_keys)},
            # 対応付け済みなのでwould_be_mask/score/by_key/reject_reasonは対象外(列を揃えるためNoneで埋める)。
            "would_be_mask": None,
            "would_be_score": None,
            "would_be_by_key": {k: None for k in key_names},
            "reject_reason": None,
            "is_query": (j == target_j),
        })

    # 2026-10-01追加(ユーザー要望): 「全クエリ×全マスク」の完全な採点表が欲しいとのこと。
    # all_assignmentsは各品目につき対応付け(or 足切り/未割当)された1マスクの内訳しか
    # 持たず、「このクエリは他のマスクに対してはどんな点数だったか」が分からない。
    # per_maskは逆に1つの代表クエリ(target_j)に対する全マスクの採点表でしかなく、
    # クエリを跨いだ比較ができない。以前はクエリごとにフォルダ(debug json)を分けて
    # 生成しており、per_mask相当の情報がクエリの数だけ存在したが、現在は撮影1枚に
    # つき1ファイルにまとめているため、target_j以外の品目についてはこの情報が
    # 失われていた。full_score_matrixとして全品目×全マスクの採点をまるごと残す。
    full_score_matrix = [
        {
            "book_name": m.get("book_name", ""),
            "display_name": m.get("display_name", ""),
            "by_mask": {
                f"mask_{i + 1}": {
                    "total": round(float(S[i, j]), 1),
                    "by_key": {
                        key_names[k]: round(float(per_key[i, j, k]), 1)
                        for k in range(n_keys)
                    },
                    # 2026-10-01追加(ユーザー要望): マスク名だけでは画像上のどの箱を
                    # 指しているか分からないので、per_maskと同じくそのマスクに帰属した
                    # OCR文字列(combined[i])も添える。
                    "text": combined[i][:200],
                }
                for i in range(n_mask)
            },
        }
        for j, m in enumerate(master)
    ]

    # ===== デバッグ保存 =====
    try:
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "multikey_match_debug.json").write_text(
            json.dumps({
                "query": query,
                "master_json": str(master_path),
                "master_row": master[target_j],
                "assign_method": method,
                "score_combine_method": SCORE_COMBINE_METHOD,
                "threshold": threshold,
                "selected_mask": None if sel_i is None else f"mask_{sel_i + 1}",
                "independent_selected_mask": None if independent_i is None else f"mask_{independent_i + 1}",
                "hungarian_changed_selection": bool(hungarian_changed_selection),
                "selected_score": round(top, 1),
                "winning_key": win_key,
                "margin": round(margin, 1),
                "selected_text_len": len(_normalize(selected_text)),
                "text_plausible": bool(text_plausible),
                "ref_score": round(ref_score, 1),
                "support_scores": {k: round(v, 1) for k, v in support_scores.items()},
                "support_count": support_count,
                "confident": bool(confident),
                "confident_reason": confident_reason,
                "confident_rule": (
                    f"ref_score>={REF_EXACT_SCORE} で無条件に確信あり。"
                    f"それ以外は display_name/date/spec_1/spec_2 のうち"
                    f"{SUPPORT_KEY_MIN_COUNT}本以上が{SUPPORT_KEY_SCORE}点以上、"
                    f"かつtext_plausible(len>={MIN_TEXT_LEN_FOR_PLAUSIBLE} and looks like REF/date)"
                    "で確信あり(marginは判定に使わない、参考値としてのみ保存)"
                ),
                "per_mask": [
                    {
                        "mask": f"mask_{i + 1}",
                        "score": round(float(raw_col[i]), 1),
                        "by_key": {
                            key_names[k]: round(float(per_key[i, target_j, k]), 1)
                            for k in range(n_keys)
                        },
                        "text": combined[i][:200],
                    }
                    for i in range(n_mask)
                ],
                "all_assignments": all_assignments,
                "ocr_assignments": ocr_debug,
                "full_score_matrix": full_score_matrix,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[multikey] デバッグ保存に失敗（処理は継続）: {e}")

    if sel_i is not None:
        _save_perkey_overlay(
            shot_dir=shot_dir,
            rgb_path=rgb_path,
            ocr_json_path=ocr_json_path,
            ocr_debug=ocr_debug,
            sel_mask_name=f"mask_{sel_i + 1}",
            keys_of_target=_keys_of(master[target_j]),
            mask=masks[sel_i],
        )
    if save_all_assignments:
        # 全マスク x 全品目の割当結果を1枚にまとめて可視化する
        # (2026-08-26試験導入)。「1回の認識でマスタ全品目に対しマスクが
        # 割り当たっているはずなので、query以外も正しく別マスクへ選べて
        # いるか一望で確認したい」との要望に対応。
        mask_to_item = {}
        for a in all_assignments:
            if a["assigned_mask"] is None:
                continue
            mi = int(a["assigned_mask"].rsplit("_", 1)[1]) - 1
            mask_to_item[mi] = {
                "book_name": a["book_name"] or a["display_name"] or "?",
                "is_query": a["is_query"],
                "score": a["score"],
            }
        _save_assignment_overview(
            shot_dir=shot_dir,
            rgb_path=rgb_path,
            masks=masks,
            boxes=boxes,
            mask_to_item=mask_to_item,
        )
    if save_all_perkey_overlays:
        _save_all_perkey_overlays(
            shot_dir=shot_dir,
            rgb_path=rgb_path,
            ocr_json_path=ocr_json_path,
            ocr_debug=ocr_debug,
            all_assignments=all_assignments,
            master=master,
            masks=masks,
        )

    if not confident:
        support_str = ", ".join(f"{k}={v:.1f}" for k, v in support_scores.items())
        print(f"[multikey] 警告: query='{query}' は確信度が低いです "
              f"(ref_score={ref_score:.1f}, support_count={support_count}/{SUPPORT_KEY_MIN_COUNT}, "
              f"{support_str})")

    return results


# 呼び出し側が find_similar_books を使っている場合のための別名
find_similar_books = match_text_to_mask_main
