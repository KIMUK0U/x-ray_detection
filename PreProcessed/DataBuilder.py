import os
import glob
import random
import shutil
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple, Dict, List, Any

try:
    from PreProcessed.LabelProcessor import LabelProcessor
except ImportError:
    from LabelProcessor import LabelProcessor
try:
    from PreProcessed.XRayPreprocessor import XRayPreprocessor 
except ImportError:
    from XRayPreprocessor import XRayPreprocessor

class DatasetBuilder:
    def __init__(self, root_dir: str, output_base_dir: str, preprocessor, label_processor):
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.label_processor = label_processor
        self.splits = ['train', 'val', 'test']
        
        # モードに基づいて出力ディレクトリを決定
        mode_name = self.label_processor.mode.replace('_', '-').upper()
        self.final_output_dir = os.path.join(output_base_dir, mode_name, 'Data') 

    def create_directory_structure(self):
        """出力先ディレクトリの作成"""
        if os.path.exists(self.final_output_dir):
            shutil.rmtree(self.final_output_dir)
            
        for split in self.splits:
            os.makedirs(os.path.join(self.final_output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.final_output_dir, split, 'labels'), exist_ok=True)

    def get_data_pairs(self) -> List[Dict[str, Any]]:
        """全データのリストを作成"""
        data_pairs = []
        
        mode_config = {
            'unet': {'folder': 'U-Net', 'suffix': '_mask.png'},   
            'yolo': {'folder': 'YOLO', 'suffix': '.txt'},        
            'faster_rcnn': {'folder': 'F-RCNN', 'suffix': '.txt'}, 
        }
        
        current_mode = self.label_processor.mode
        if current_mode not in mode_config:
            raise ValueError(f"Unknown label mode: {current_mode}")
            
        config = mode_config[current_mode]
        
        # --- 1. 結節あり (Nodule) ---
        nodule_path = os.path.join(self.root_dir, 'nodule')
        dicom_dir = os.path.join(nodule_path, 'Dicom')
        label_root_dir = os.path.join(nodule_path, 'Labels', config['folder']) 
        dcm_files = glob.glob(os.path.join(dicom_dir, '*.dcm'))
        
        for dcm_file in dcm_files:
            base_name = os.path.splitext(os.path.basename(dcm_file))[0]
            label_file = os.path.join(label_root_dir, base_name + config['suffix'])
            
            # ラベルファイルが存在しない場合は結節なしとして扱う
            if not os.path.exists(label_file):
                label_file = None
                has_nodule = False 
            else:
                has_nodule = True
            
            data_pairs.append({'img': dcm_file, 'lbl': label_file, 'has_nodule': has_nodule})
        # --- 2. 結節なし (Non-Nodule) ---
        non_nodule_path = os.path.join(self.root_dir, 'non_nodule')
        
        # 🚨 修正: 再帰的な検索を使用して、サブディレクトリ内のすべての.dcmファイルを収集する 🚨
        non_dcm_files = glob.glob(os.path.join(non_nodule_path, '**', '*.dcm'), recursive=True)
        
        for dcm_file in non_dcm_files:
            data_pairs.append({'img': dcm_file, 'lbl': None, 'has_nodule': False})
            
        return data_pairs

    def build(self, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """データセット構築処理"""
        print(f"Building dataset for mode: {self.label_processor.mode}")
        self.create_directory_structure()
        all_data = self.get_data_pairs()
        
        random.seed(42)
        
        # --- データを結節有無で分離 ---
        nodule_data = [d for d in all_data if d['has_nodule']]
        non_nodule_data = [d for d in all_data if not d['has_nodule']]
        
        random.shuffle(nodule_data)
        random.shuffle(non_nodule_data)
        
        # --- 分割関数 ---
        def split_list(data_list, ratios):
            count = len(data_list)
            train_end = int(count * ratios[0])
            val_end = int(count * (ratios[0] + ratios[1]))
            return data_list[:train_end], data_list[train_end:val_end], data_list[val_end:]

        # 結節ありデータの分割
        nod_train, nod_val, nod_test = split_list(nodule_data, split_ratios)
        
        # 結節なしデータの分割
        non_train, non_val, non_test = split_list(non_nodule_data, split_ratios)
        
        current_mode = self.label_processor.mode
        
        datasets = {
            'train': nod_train + non_train,
            'val': nod_val + non_val,
            'test': nod_test + non_test
        }
        
        # シャッフル (結節ありとなしが混ざるように)
        for split in datasets:
            random.shuffle(datasets[split])

        # ログ出力
        print(f"Total Nodule: {len(nodule_data)}, Total Non-Nodule: {len(non_nodule_data)}")
        for split, data in datasets.items():
            print(f"  [{split.upper()}] Total: {len(data)}")

        # --- 処理と保存 ---
        for split, data_list in datasets.items():
            print(f"Processing {split} data...")
            for i, item in tqdm(enumerate(data_list), total=len(data_list)):
                
                # 1. 画像処理
                processed_img = self.preprocessor.run(item['img'])
                if processed_img is None: continue
                
                # 2. ラベル処理
                processed_lbl = self.label_processor.process(item['lbl'], item['has_nodule'])
                
                base_filename = f"{i:05d}"
                
                if current_mode == 'yolo':
                    # --- YOLO: PNG & TXT ---
                    img_uint8 = (processed_img * 255).astype(np.uint8)
                    if img_uint8.shape[-1] == 1:
                        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    elif img_uint8.ndim == 2:
                        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    
                    save_img_path = os.path.join(self.final_output_dir, split, 'images', f"{base_filename}.png")
                    cv2.imwrite(save_img_path, img_uint8)
                    
                    save_lbl_path = os.path.join(self.final_output_dir, split, 'labels', f"{base_filename}.txt")
                    with open(save_lbl_path, 'w') as f:
                        # 結節なし(processed_lblが空)の場合、空のtxtファイルが生成される (正しい挙動)
                        if processed_lbl is not None and processed_lbl.shape[0] > 0:
                            for row in processed_lbl:
                                f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")
                                
                else:
                    # --- 他モデル: NPY ---
                    save_img_path = os.path.join(self.final_output_dir, split, 'images', f"{base_filename}.npy")
                    save_lbl_path = os.path.join(self.final_output_dir, split, 'labels', f"{base_filename}.npy")
                    np.save(save_img_path, processed_img)
                    np.save(save_lbl_path, processed_lbl)

        if current_mode == 'yolo':
            self._create_yolo_yaml()
            
        print("Dataset construction completed!")

    def _create_yolo_yaml(self):
        yaml_content = {
            'path': os.path.abspath(self.final_output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'nodule'}
        }
        yaml_path = os.path.join(self.final_output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"Created YOLO config: {yaml_path}")

if __name__ == "__main__":
    INPUT_ROOT = "./Data"
    OUTPUT_BASE_DIR = "."
    IMAGE_SIZE = (512, 512) # 学習画像サイズ

    LABEL_MODE = 'unet'  # 'unet', 'yolo', 'faster_rcnn' から選択可能
    
    preprocessor = XRayPreprocessor(target_size=IMAGE_SIZE)
    label_processor = LabelProcessor(target_size=IMAGE_SIZE, mode=LABEL_MODE)
    
    builder = DatasetBuilder(
        root_dir=INPUT_ROOT, 
        output_base_dir=OUTPUT_BASE_DIR, 
        preprocessor=preprocessor,
        label_processor=label_processor
    )
    
    builder.build()