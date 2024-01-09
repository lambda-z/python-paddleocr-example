import logging

from paddleocr import PaddleOCR, draw_ocr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ocr = PaddleOCR(use_angle_cls=True, lang="ch")

print("ocr model loaded.")


def main():

    """
        [2024/01/09 23:22:57] ppocr DEBUG: Namespace(help='==SUPPRESS==', use_gpu=False, use_xpu=False, use_npu=False,
        ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, image_dir=None,
        page_num=0, det_algorithm='DB', det_model_dir='C:\\Users\\kishi/.paddleocr/whl\\det\\ch\\ch_PP-OCRv4_det_infer',
        det_limit_side_len=960, det_limit_type='max', det_box_type='quad', det_db_thresh=0.3, det_db_box_thresh=0.6,
         det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8,
         det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2,
         det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0,
         beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir='C:\\Users\\kishi/.paddleocr/whl\\rec\\
         ch\\ch_PP-OCRv4_rec_infer', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_len
         gth=25, rec_char_dict_path='D:\\root\\venvs\\python-paddleocr-example\\lib\\site-packages\\paddleocr\\ppocr\\ut
         ils\\ppocr_keys_v1.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algo
         rithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5,
         e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast',
         use_angle_cls=True, cls_model_dir='C:\\Users\\kishi/.paddleocr/whl\\cls\\ch_ppocr_mobile_v2.0_cls_infer',
         cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False,
         cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', s
         r_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output',
         use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True,
         use_onnx=False, output='./output', table_max_len=488, table_algorithm='TableAttn', table_model_dir=None,
         merge_no_span_structure=True, table_char_dict_path=None, layout_model_dir=None, layout_dict_path=None,
         layout_score_threshold=0.5, layout_nms_threshold=0.5, kie_algorithm='LayoutXLM', ser_model_dir=None,
         re_model_dir=None, use_visual_backbone=True, ser_dict_path='../train_data/XFUND/class_list_xfun.txt',
         ocr_order_method=None, mode='structure', image_orientation=False, layout=True, table=True, ocr=True,
         recovery=False, use_pdf2docx_api=False, invert=False, binarize=False, alphacolor=(255, 255, 255),
         lang='ch', det=True, rec=True, type='ocr', ocr_version='PP-OCRv4', structure_version='PP-StructureV2')

        ocr model loaded.
        [2024/01/09 23:22:59] ppocr DEBUG: dt_boxes num : 1, elapsed : 0.6124627590179443
        [2024/01/09 23:22:59] ppocr DEBUG: cls num  : 1, elapsed : 0.017464160919189453
        [2024/01/09 23:22:59] ppocr DEBUG: rec_res num  : 1, elapsed : 0.14176249504089355
        [[[325.0, 252.0], [945.0, 252.0], [945.0, 572.0], [325.0, 572.0]], ('特征', 0.9942247867584229)]
        图片坐标:  [[325.0, 252.0], [945.0, 252.0], [945.0, 572.0], [325.0, 572.0]]
        文字:  特征 置信度:  99.4225 %
    """

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
