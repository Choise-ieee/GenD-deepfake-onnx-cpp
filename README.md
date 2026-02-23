# GenD-deepfake-onnx-cpp
Official implementation for the WACV 2026 paper "Deepfake Detection that Generalizes Across Benchmarks" onnx and cpp version

The reference paper:Deepfake Detection that Generalizes Across Benchmarks (WACV 2026) [https://github.com/yermandy/GenD]

## 1.windows environment setting up
According to the guide(chapter "Minimal dependencies" in official repository) in cuda12.8 windows system.
<img width="1038" height="1406" alt="image" src="https://github.com/user-attachments/assets/217835d5-f06c-47cb-bf40-3fbc4203c28b" />
<img width="1022" height="1408" alt="image" src="https://github.com/user-attachments/assets/285f9aa3-d1f4-4fab-9d38-1e768886fbb4" />
<img width="738" height="1398" alt="image" src="https://github.com/user-attachments/assets/f360a9bf-a17d-42f9-b0ea-eacf44b8ee79" />

## 2.export to onnx
According to the export_to_onnx.py, And run it to tranlate to the onnx file(gend.onnx).
<img width="3096" height="1104" alt="45eb4a610371d20a68e9b7bb1e2ee968" src="https://github.com/user-attachments/assets/7a45517f-5e9b-49bc-b9e9-2076a05a285a" />

The onnx model also can be download as:
1) gend.onnx: https://drive.google.com/file/d/1SZtRKPxlk-cvN3rLwvYgfEh9I0xDrk4z/view?usp=sharing
2) det_10g.onnx:in this project file lists.

## 3.onnx model python evaculation 
Use the verify_onnx.py to model inference.And we also use detect_image.py(Modified from the original Inference with transformers) to compare inference result.
And we found the onnx model results almost the same as the author inference ones.

<img width="2964" height="1440" alt="image" src="https://github.com/user-attachments/assets/74582734-9606-49b6-87f1-e24ac97ac08e" />

## 4.onnx model CPP evaculation 
Use the main.cpp file in VS2019 environt, and we found the results are also the sames.(The little difference is generated from the picture scale algorithm between opencv and python PIL )
<img width="3114" height="1640" alt="image" src="https://github.com/user-attachments/assets/3814514c-1dbf-4f9d-91f9-4903d52bbfc4" />

## 5.onnx-cpp cuda accelerate
We can speed up the inference by using onnx cuda mode, for example demonstrated in GTX-1060 environment.
<img width="3104" height="1650" alt="image" src="https://github.com/user-attachments/assets/fdd8312f-1413-4b52-95c3-2194de6b800c" />

## Thanks
Thanks for their extremelly perfect work.

@InProceedings{GenD,
  author    = {Yermakov, Andrii and Cech, Jan and Matas, Jiri and Fritz, Mario},
  title     = {Deepfake Detection that Generalizes Across Benchmarks},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {March},
  year      = {2026},
}
