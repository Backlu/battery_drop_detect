# F45 Battery Drop Detect

## DS的部分
### Service: Battery Drop Detect
- folder: service_batteryDropDetect
- cmd: 
    - python3 service_batteryDropDetect/f45_inference_main_online.py {crontabEndTimeOfHour} {crontabEndTimeOfMin} {camera_mid}
- 系統上線執行方式
    - 排程執行schedule.sh
- 排程方式: crontab (root)

|  module   | function  |
|  ----  | ----  |
| f45_inference_main_online.py | 偵測battery脫落, 並將偵測結果寫入DB |
| f45movement.py | battery脫落偵測主要模組 |
| schedule.py | call pod-f45.sh 啟動inference |
| darknet.py | yolov4模組|
| mrcnn | mask rcnn模組 |
| Dev_batteryDropDetect_sample_code.ipynb | offline測試用sample code |

### Service: Hourly AI Report 
- folder: service_batteryDropDetect
- cmd: 
    - python3 service_AIReport/f45_hourly_check.py.py {is_online}
    - python3 service_AIReport/f45_get_mes_data_min.py
- 系統上線執行方式
    - 排程執行f45_get_mes_data_min.sh, f45_hourly_check.sh
- 排程方式: crontab

|  module   | function  |
|  ----  | ----  |
| f45_get_mes_data_min.py | 從MES database query SN並寫入AA database (sn_detail_min) |
| f45_hourly_check.py | 將MES SN透過時間對應到AI檢測到的battery drop record (f45_anomaly_info) |
| Dev_HourlyAIReport_sample_code.ipynb | hourly ai report sample code|


### Service: Camera Quality Check
- folder: service_cameraQualityCheck
- cmd: 
    - python3 service_cameraQualityCheck/cam_quality_check_main.py
    - python3 service_cameraQualityCheck/cam_image_reflash_main.py
- 系統上線執行方式
    - 排程執行f45_quality_check.sh
- 排程方式: crontab

|  module   | function  |
|  ----  | ----  |
| cam_quality_check_main.py | 檢查camera quality |
| f45_quality_check.py | F45 camera quality檢查模組 |
| f68_quality_check.py | F68 camera quality檢查模組 (待移回F68專案repo) |
| camera_image_plot.py | 把所有的camera畫面組成一張圖 |
| cam_image_reflash_main.py | 只更新camera畫面, 不檢查quality |
| Dev_camera_quality_check.ipynb | sample code |


### Service: Camera Quality Review UI
- folder: service_cameraQualityReviewUI
- cmd: streamlit run service_cameraQualityReviewUI/app.py --server.port 5009 --theme.base light
- 系統上線執行方式
    - 手動執行 streamlit cmd

|  module   | function  |
|  ----  | ----  |
| app.py | streamlit UI主程式  |
| config.toml | UI設定檔 |
| tmp | UI執行時的暫存資料夾 |
| lotti | UI執行時的等待動畫gif檔 |
| asset | UI畫面上的camera檢查SOP圖檔 |


### Service: Copy Images
- folder: service_copyImages
- cmd: 
    - python3 service_copyImages/move_video_file_to_db58.py
    - python3 service_copyImages/move_video_file_to_60.py
- 系統上線執行方式: 排程執行python cmd
- 排程方式: crontab
- doc: https://www.notion.so/jayschsu/F45-Refactoring-a75ac00ae6f74cda904fddd6d44c960b (附錄．image產生流程)

|  module   | function  |
|  ----  | ----  |
| create_video_folder.py | 預先產生存放image的資料夾 |
| move_video_file_to_db58.py | Prepare AI label recheck images  |
| move_video_file_to_60.py | Prepare AI label recheck images  |


### Service: Model Training
- folder: service_modelTraining
- model training執行方式: 手動執行

|  module   | function  |
|  ----  | ----  |
| coco2yolo.py | 將coco label轉換為yolo可用格式 |
| dev_coco2yolo.ipynb | coco2yolo sample code |
| coco2maskrcnn.py | 將coco label轉換為maskrcnn可用格式 |
| maskrcnn_training.py | maskrcnn training module |
| Dev_coco2maskrcnn.ipynb | coco2maskrcnn sample code|
| Dev_MaskRCNN_model_train.ipynb | maskrcnn training sample code |
| Dev_MaskRCNN_model_evaluate.ipynb| maskrcnn evaluate sample code |



### Unit Test
- folder: unit_test
- cmd: pytest unit_test/ -s --disable-pytest-warnings

|  module   | function  |
|  ----  | ----  |
| test_batteryDropDetect.py | battery drop detect testing |
| test_camera_quality_check.py | camera quality check (f45, f68) |
| test_hourly_sync_mes_sn.py | hourly ai report testing |


### Database tabel
- IP: 10.109.6.58
- MES連線資訊: https://hackmd.io/JYwqWLWkRMKBxtrs6VZ6rg?both=

|  table   | description  |
|  ----  | ----  |
| f45_anomaly_info | AI偵測結果 |
| OP info | 透過EMI API取得OP資訊 |
| inference_schedule_setting | camera quality check的檢查時間與檢查結果 |
| f45_anomaly_sn | AI report執行結果 |
| f45_camera_list_map | 紀錄camera資訊 |
| MES SN info | 從MES DB取得過站SN資訊 |
| sn_detail_min | 從MES DB取得過站SN資訊 |



## DE的部分

### shell script

|  module   | 功能  |
|  ----  | ----  | 
| schedule.sh | schedule.py 啟動inference |
| pod-f45.sh | call pod-f45.yaml啟動inference, call cronjob-f45.yaml設定排程 |
| create_video_folder.sh |  |
| f45_get_mes_data_min.sh | 從MES database query SN並寫入AA database |
| f45_hourly_check.sh | 將MES SN透過時間對應到AI檢測到的battery drop record |
| f45_quality_check.sh | 執行camera quality check |


### YAML
|  module   | 功能  | cmd |
|  ----  | ----  | ----  |
| pod-f45.yaml | 啟動inference |
| cronjob-f45.yaml | 設定inference排程 |


### 資料掛載:
- model folder: /mnt/hdd1/Data/f45movement
- data floder: /mnt/hdd1/Model/f45movement


### Environment
#### Production
- 10.142.3.62 (master/worker1)
- 10.142.3.63 (worker2)
- 10.142.3.63 (worker3)
- 10.142.3.58 (db server/worker4)
    - Deploy Directory: /mnt/hdd/f45movement
    - 請不要直接改這個folder內的檔案. 更新程式請透過CICD

