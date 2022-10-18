#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 設定ファイル
import configparser

# ログ出力用ライブラリのインポート
from logging import basicConfig, getLogger, StreamHandler, DEBUG, INFO, ERROR

# 終了シグナルをキャッチするライブラリのインポート
import signal

from typing import Sequence
from absl import app

# ロガーのインスタンス作成
logger = getLogger(__name__)
stream_handler = StreamHandler()

# ログレベルを設定
level = DEBUG
logger.setLevel(level)
stream_handler.setLevel(level)

# ロガーにハンドラーをセット
# logger.addHandler(stream_handler)

import cv2
import os
import shutil
import numpy as np


# --------------------------------------------------
# configparserの宣言とiniファイルの読み込み
# --------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

TARGET_FILE = '202210010000.jpg'
# IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/img/'

# 元素材
IMG_DIR = config['RemoveUnnecessaryImages']['IMG_DIR']

# 出力先
OUT_DIR = config['RemoveUnnecessaryImages']['OUT_DIR']

# 解析時のイメージサイズ
IMG_SIZE = (
  int(config['RemoveUnnecessaryImages']['ANALYSIS_IMG_SIZE']),
  int(config['RemoveUnnecessaryImages']['ANALYSIS_IMG_SIZE'])
)

# 解析時のイメージサイズ
HIST_THRESHOLD = float(config['RemoveUnnecessaryImages']['HIST_THRESHOLD'])


def run() -> None:
  # logger.debug("test")
  calcHist()

  # img = cv2.imread(IMG_DIR + TARGET_FILE)
  # img = cv2.resize(img, IMG_SIZE)
  # calcCenter(img)


# センターがずれたものを判定する関数
def calcCenter(out_path, img):

  # # 画像が存在するかを確認
  # if not os.path.exists(path):
  #   print("画像が存在しません。")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  gauss = cv2.GaussianBlur(gray, (5, 5), 0)


  ret, thresh = cv2.threshold(gauss, 50, 255, cv2.THRESH_BINARY)

  minDist = int(IMG_SIZE[0]/10)
  minW = int(IMG_SIZE[0] * 0.38)
  maxW = int(IMG_SIZE[0] * 0.43)
  centerPosError = 0.02

  circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, param1=30, param2=10, minRadius=minW, maxRadius=maxW)

  # logger.debug(circles)

  insideOfRange = False

  if circles.any():

    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:

      isWithInRangeX = IMG_SIZE[0] * (0.5 - centerPosError) < circle[0] and circle[0] < IMG_SIZE[0] * (0.5 + centerPosError)
      isWithInRangeY = IMG_SIZE[0] * (0.5 - centerPosError) < circle[1] and circle[1] < IMG_SIZE[0] * (0.5 + centerPosError)

      if (isWithInRangeX and isWithInRangeY):

        # 円周を描画する
        cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 165, 255), 5)
        # 中心点を描画する
        cv2.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        insideOfRange = True

      # # 平行移動の変換行列を作成
      # afin_matrix = np.float32([[1,0,dx/2],[0,1,dy/2]])
      #
      # # アファイン変換適用
      # img = cv2.warpAffine(
      #   img,           # 入力画像
      #   afin_matrix,   # 行列
      #   (W,W)  # 解像度
      # )


    # cv2.imwrite(out_path, img)
  else:
    logger.debug("検出なし")

  return insideOfRange



# 壊れた画像を除くために特徴点差分を取って類似度が低いものを除去
def calcHist() -> None:
  target_img_path = IMG_DIR + TARGET_FILE

  target_img = cv2.imread(target_img_path)
  target_img = cv2.resize(target_img, IMG_SIZE)
  target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

  logger.debug('TARGET_FILE: %s' % (TARGET_FILE))

  files = os.listdir(IMG_DIR)
  files.sort()
  for file in files:

    if file == '.DS_Store' or file == TARGET_FILE or file[0] == '.':
      continue
    # print(file)

    comparing_img_path = IMG_DIR + file
    comparing_img = cv2.imread(comparing_img_path)
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

    ret = cv2.compareHist(target_hist, comparing_hist, 0)


    if ret > HIST_THRESHOLD:

      # 中心ずれの検証
      insideOfRange = calcCenter(OUT_DIR + file, comparing_img)

      if insideOfRange:
        logger.debug('ズレ検証OK: %s' % (file))
        shutil.copyfile(comparing_img_path, OUT_DIR + file)
      else:
        logger.debug('ズレ検証NG: %s' % (file))

      # shutil.copyfile(comparing_img_path, OUT_DIR + file)
      target_hist = comparing_hist



      # print(file, ret)
      # os.remove(comparing_img_path)
    else:
      logger.debug("REMOVE!!  %s  %s", file, ret)
      # os.remove(comparing_img_path)


class TerminatedExecption(Exception):
  pass

def raise_exception(*_):
  raise TerminatedExecption()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # メインの実行関数
  run()


if __name__ == '__main__':

  # execute
  try:
    # Ctrl + C (SIGINT) で終了
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # set signal handler to detect to be stopped by systemd
    signal.signal(signal.SIGTERM, raise_exception)

    app.run(main)


  # (1) if ctrl-C is pushed, stop program nomally
  except KeyboardInterrupt:
    print("KeyboardInterrupt: stopped by keyboard input (ctrl-C)")

  # (2) if stopped by systemd, stop program nomally
  except TerminatedExecption:
    print("TerminatedExecption: stopped by systemd")

  # (3) if error is caused with network, restart program by systemd
  except OSError as e:
    import traceback
    traceback.print_exc()

    print("NETWORK_ERROR")

    # program will be restarted automatically by systemd (Restart on-failure)
    raise e

  # (4) if other error, restart program by systemd
  except Exception as e:
    import traceback
    traceback.print_exc()

    print("UNKNOWN_ERROR")

    # program will be restarted automatically by systemd (Restart on-failure)
    raise e

  # 終了
  exit()

