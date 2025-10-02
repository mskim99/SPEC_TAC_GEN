import os
import glob
import json

def save_conditional_inputs(input_dir, out_path="conditional_inputs.json"):
    """
    지정한 폴더 내의 .npy 파일들을 읽어서
    첫 번째 '_' 앞 문자열을 condition으로 추출 후 JSON 저장
    """
    files = glob.glob(os.path.join(input_dir, "*.npy"))

    data_list = []
    for file_path in files:
        file_name = os.path.basename(file_path)  # 파일명만 추출
        base_name = os.path.splitext(file_name)[0]  # 확장자 제거
        condition = base_name.split("_")[0]  # 첫 번째 '_' 앞 부분

        data_list.append({
            "file_name": file_name,
            "condition": condition
        })

    with open(out_path, "w") as f:
        json.dump(data_list, f, indent=4)

    print(f"✅ Saved {len(data_list)} items to {out_path}")


# 실행 예시
if __name__ == "__main__":
    input_dir = "./data/phase"  # .npy 파일들이 들어 있는 폴더
    save_conditional_inputs(input_dir, "phase_cond.json")