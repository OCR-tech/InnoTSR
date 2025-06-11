# InnoTSR: Real-Time Traffic Sign Recognition System with Voice Feedback

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.18%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

<!-- ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=OCR-tech.InnoTSR) -->
<!-- ![GitHub repo size](https://img.shields.io/github/repo-size/OCR-tech/InnoTSR) -->
<!-- ![GitHub commit activity](https://img.shields.io/github/commit-activity/m/OCR-tech/InnoTSR) -->
<!-- ![GitHub contributors](https://img.shields.io/github/contributors-anon/OCR-tech/InnoTSR) -->

**InnoTSR** project is a Python-based real-time Traffic Sign Recognition (TSR) system using deep learning. It provides voice feedback and real-time information to users for personalized alert systems.

<br/>
<p align="center">
<img src="docs/public/img/img1a.png" style="width:35%; height:auto;">&emsp;
</p>

## Key Features

- **High Accuracy**: Utilizes cutting-edge deep learning algorithms for accurate traffic sign detection and recognition.
- **Voice Command and Alerts**: Built-in instant voice command and alert to notify the user of detected traffic sign information in real-time.
- **Enhanced User Experience**: Provides a seamless and intuitive graphical user interface experience.

## Installation

Requirements:

- **Python** >= 3.11
- **TensorFlow** >= 2.18
- **OpenCV** (video capturing and processing)
- **SpeechRecognition** (voice command processing)
  <!-- - TensorFlow 2.18 or higher -->
  <!-- - SSD MobileNet V2 model -->

To install and run this project, please follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/OCR-tech/InnoTSR.git
   cd InnoTSR
   ```

2. Create a virtual environment:

   ```sh
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

3. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

<!-- # ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1.tar -->
<!-- 4. Download the [SSD MobileNet V2 TensorFlow 2 model](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1) and extract the files into `app/models/pretrained_model/`.
   - Ensure the directory contains files like `saved_model.pb` and the `saved_model` folder. -->

<!-- 4. Download the [SSD MobileNet V2 TensorFlow 2 model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz) and extract it into `app/models/pretrained_model/`.
   - The directory should contain `saved_model.pb` and a `saved_model` folder. -->

4. Run the application:

   ```sh
   python app/main.py
   ```

## Usage

Follow these instructions:

1. **Prepare your environment**: Ensure your system has a camera or input method to capture traffic signs.
2. **Start the application**: Run the main script to initiate the traffic sign recognition system.
3. **Voice Command Alerts**: The system will provide real-time voice alerts for detected traffic signs.

## Contributing

- See the [CONTRIBUTING](CONTRIBUTING.md) for detailed guidelines.

<!-- For contributions, please follow the steps below:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request. -->

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

<!-- If you have questions, suggestions, or would like to contribute, feel free to reach out: -->

- **Email**: ocrtech.mail@gmail.com
- **Website**: [https://ocr-tech.github.io](https://ocr-tech.github.io)
- **GitHub**: [https://github.com/OCR-tech](https://github.com/OCR-tech)
