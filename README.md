# The Laboratory-to-Field Gap
**Cross-domain evaluation of deep learning models for plant disease and fruit defect detection, with real-time camera detection**

## What is in this project
| File | Purpose |
|---|---|
| `run_all.py` | **One command: trains everything, makes all graphs, opens the results dashboard** |
| `make_dashboard.py` | Builds graphs + summary table + `results/dashboard.html` from `runs/` |
| `check_data.py` | Shows which classes match between lab and field folders (run first) |
| `merge_folders.py` | Merges train/test class folders into one folder |
| `train.py` | Trains on lab images; tests on lab (in-domain) and field (cross-domain); few-shot field fine-tuning; saves model, metrics, confusion matrices, curves |
| `evaluate.py` | Evaluates one model or an **ensemble** of models on any folder |
| `baseline_ml.py` | Classical ML baseline (colour histograms + Random Forest / SVM) for comparison with deep learning |
| `realtime.py` | **Real-time webcam detection** (plant and fruit modes, confidence, FPS, screenshots) |
| `app.py` | Streamlit web demo (upload or camera photo) |
| `class_map.json` | Maps differing class names between datasets |

Models: any `timm` model (EfficientNet-B0, ResNet50, MobileNetV3, ViT-B/16, ConvNeXt...), plus classical ML baselines and ensembling.

## 1. Install (Python 3.10 or 3.11 recommended)
```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
```
Internet is needed once, to download pretrained weights (ImageNet) the first time you train.

## 2. Arrange the datasets
```
data/plant/lab/<class>/*.jpg      PlantVillage  (point at the folder that directly contains the class folders)
data/plant/field/<class>/*.jpg    Cropped-PlantDoc (class folders; merge its train+test with merge_folders.py)
data/fruit/lab/<class>/*.jpg      Kaggle "Fruits Fresh and Rotten for Classification"
data/fruit/field/<class>/*.jpg    Mendeley "Fresh and Rotten Fruits" (original images, not augmented)
```
Then check the class matching and fix mismatches in `class_map.json`:
```
python check_data.py --lab_dir data/plant/lab --field_dir data/plant/field
python check_data.py --lab_dir data/fruit/lab --field_dir data/fruit/field
```
The matching works because training uses all lab classes, and the field set is evaluated only on classes that exist in both.

## 3A. EASIEST WAY: one command that produces all outputs and graphs
```
python run_all.py --plant_lab data/plant/lab --plant_field data/plant/field --fruit_lab data/fruit/lab --fruit_field data/fruit/field --cap_lab 300 --epochs 5
```
This trains plant and fruit models, runs the classical ML baseline, then **opens `results/dashboard.html` in your browser** with the summary table, lab-vs-field accuracy graph, few-shot curve, ML-vs-deep-learning graph, training curves and confusion matrices. The same images and `summary.csv` are saved in `results/`. Add `--models efficientnet_b0 resnet50 mobilenetv3_large_100` to compare several models. Add `--quick` first for a few-minute test.
Nothing appears until this has finished running: the graphs are made from YOUR training results. Run it on Colab GPU (see below) if your laptop has no NVIDIA GPU, then download the `results/` and `models/` folders.

## 3. Train step by step (do this on Google Colab GPU if your laptop has no NVIDIA GPU)
Quick pipeline test (minutes, any laptop):
```
python train.py --task plant --lab_dir data/plant/lab --field_dir data/plant/field --quick
```
Real run (use a GPU):
```
python train.py --task plant --lab_dir data/plant/lab --field_dir data/plant/field --model efficientnet_b0 --epochs 8
python train.py --task fruit --lab_dir data/fruit/lab --field_dir data/fruit/field --model efficientnet_b0 --epochs 8
```
If time is short, cap images per class: `--cap_lab 300 --epochs 5`. Models are saved to `models/plant.pt` and `models/fruit.pt`; metrics, confusion matrices and curves go to `runs/`.
Try more models for your comparison table: `--model resnet50`, `--model mobilenetv3_large_100`, `--model vit_base_patch16_224` (use `--ckpt models/plant_resnet50.pt` so you do not overwrite).

**Colab:** upload this folder to Google Drive, open a notebook with a GPU runtime, then:
```
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/lab2field_project
!pip -q install timm
!python train.py --task plant --lab_dir data/plant/lab --field_dir data/plant/field --epochs 8
```
Download `models/*.pt` to the same folder on your laptop.

## 4. Real-time camera
```
python realtime.py --plant models/plant.pt --fruit models/fruit.pt
```
Keys: `1` plant, `2` fruit, `3` auto, `s` screenshot, `q` quit. Hold a single leaf or fruit in the green box, close up, in good light. If the camera does not open, try `--cam 1`. No camera? Use a video: `--cam demo.mp4`.
Web version: `streamlit run app.py`

## 5. Comparison, ensemble and baseline
```
python baseline_ml.py --lab_dir data/plant/lab --field_dir data/plant/field
python evaluate.py --ckpts models/plant.pt --data_dir data/plant/field
python evaluate.py --ckpts models/plant.pt models/plant_resnet50.pt --data_dir data/plant/field     (ensemble)
```

## 6. What accuracies to expect (honest guidance)
- **Lab accuracy:** fine-tuned pretrained models usually reach about 97-99%+ on a random split of PlantVillage and similar clean datasets. Part of this is inflated because lab datasets contain many near-identical photos of the same leaf, which is exactly the problem this project highlights.
- **Field accuracy:** usually much lower, often far below lab accuracy, because of the lab-to-field gap. This is the finding of the project, not a bug. Few-shot fine-tuning and strong augmentation raise it.
- Report **whatever your run produces**. Never edit numbers: results are checked by re-running, and the viva will ask about them.

## 7. Demo plan for your professor (5 minutes)
1. Explain the problem: models look perfect on lab data but fail on real photos.
2. Show `runs/*/metrics.json`, `cm_lab.png`, `cm_field.png`, `kshot.png`.
3. Run `realtime.py`: show a leaf (plant mode) and a fruit (fruit mode), and point out low-confidence behaviour on unseen objects.
4. Show the table: lab acc vs field acc vs gap, ML baseline vs deep learning, ensemble.

## 8. Likely viva questions
- *Why transfer learning?* Pretrained ImageNet features need less data and train faster.
- *Why macro-F1?* Classes are imbalanced, so accuracy can hide poor classes.
- *Why is field accuracy lower?* Domain shift: background, lighting, occlusion, camera, scale.
- *How did you avoid leakage?* The field test set is never used for training or model selection; selection uses lab validation only. (Be ready to mention near-duplicate images within PlantVillage.)
- *What would improve it?* Domain adaptation, more field data, detection of the leaf before classification.
- *Limitations of the real-time demo?* The model only knows its training classes, so it can mislabel non-plants. Confidence thresholding reduces, but does not remove, this.

## Troubleshooting
- `No module named ...`: `pip install -r requirements.txt` inside the activated venv.
- Out of memory: `--bs 16` or `--img_size 160`.
- Very slow on laptop CPU: use Colab, or `--cap_lab 100 --model mobilenetv3_large_100 --img_size 160`.
- Matching classes shows few results: edit `class_map.json` using the `check_data.py` output.
