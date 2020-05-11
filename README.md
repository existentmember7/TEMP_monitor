# 體溫量測檢測系統, System for Identification of Action to Take Temperature
The system can be divided into three parts
## 使用者頭部追蹤 (Head detection+Multiple Object Tracking)
We refered to pre-trained face detector.
> https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

combine detections with DeepSORT(also feature extractor)
### TODO
* Extractor for face features
:::info
## 體溫計追蹤 (Single Object Tracking of thermometer)
:::
openCV built-in tracker
### TODO
* need refinement, performance do not meet the requirment
:::info
## 判斷有無進行體溫量測(Rule to identify the action)
:::
* Decided by the distance between users' faces and thermometer
