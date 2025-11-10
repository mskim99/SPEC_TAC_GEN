import torch
import torchlens as tl
from model_complex import UNet
from os.path import join as opj

# --- 1. 시각화할 모델 준비 ---
model = UNet(in_ch=2, out_ch=2, base_ch=16, ch_mult=[1,2,4], conditional=False)
model.eval() # 모델을 추론 모드로 설정

# --- 2. 모델에 입력할 더미 텐서 생성 ---
# 모델이 입력을 추적(trace)할 수 있도록 더미 입력이 필요합니다.
# (배치 크기 1, 채널 3, 높이 224, 너비 224)
dummy_input = torch.randn(1, 2, 1024, 1024)
output_filename = "cm_vis"
output_format = "png"

# --- 3. torchlens (v0.1.36)로 모델 분석 ---
# log_model 함수를 사용해 모델과 더미 입력을 전달합니다.
# 이 함수가 ModelAnalysis 객체를 반환합니다.
# model_history = tl.log_forward_pass(model, dummy_input, layers_to_save='all', vis_opt='unrolled', vis_path=output_filename)
tl.show_model_graph(model, dummy_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "simple_ff"),)
# tl.show_model_graph(model_history, dummy_input)