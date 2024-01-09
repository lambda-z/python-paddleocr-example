import logging

from paddleocr import PaddleOCR, draw_ocr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ocr = PaddleOCR(use_angle_cls=True, lang="ch")

print("ocr model loaded.")


def main():

    img_path = r'./imgs/tz.png'
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
            print("图片坐标: ", line[0])
            print("文字: ", line[1][0], "置信度: ", (line[1][1] * 100).__round__(4), "%")
            print("--------------")


if __name__ == '__main__':
    main()
