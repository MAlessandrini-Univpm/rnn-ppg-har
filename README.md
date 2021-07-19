# rnn-ppg-har
Recurrent Neural Network for Human Activity Recognition in Embedded Systems using PPG and Accelerometer

This is the software used in the article:

Alessandrini, M.; Biagetti, G.; Crippa, P.; Falaschetti, L.; Turchetti, C. "Recurrent Neural Network for Human Activity Recognition in Embedded Systems Using PPG and Accelerometer Data". Electronics 2021, 10, 1715. https://doi.org/10.3390/electronics10141715


If you use this work, please cite the article:
```
@Article{electronics10141715,
AUTHOR = {Alessandrini, Michele and Biagetti, Giorgio and Crippa, Paolo and Falaschetti, Laura and Turchetti, Claudio},
TITLE = {Recurrent Neural Network for Human Activity Recognition in Embedded Systems Using PPG and Accelerometer Data},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {14},
ARTICLE-NUMBER = {1715},
URL = {https://www.mdpi.com/2079-9292/10/14/1715},
ISSN = {2079-9292},
ABSTRACT = {Photoplethysmography (PPG) is a common and practical technique to detect human activity and other physiological parameters and is commonly implemented in wearable devices. However, the PPG signal is often severely corrupted by motion artifacts. The aim of this paper is to address the human activity recognition (HAR) task directly on the device, implementing a recurrent neural network (RNN) in a low cost, low power microcontroller, ensuring the required performance in terms of accuracy and low complexity. To reach this goal, (i) we first develop an RNN, which integrates PPG and tri-axial accelerometer data, where these data can be used to compensate motion artifacts in PPG in order to accurately detect human activity; (ii) then, we port the RNN to an embedded device, Cloud-JAM L4, based on an STM32 microcontroller, optimizing it to maintain an accuracy of over 95% while requiring modest computational power and memory resources. The experimental results show that such a system can be effectively implemented on a constrained-resource system, allowing the design of a fully autonomous wearable embedded system for human activity recognition and logging.},
DOI = {10.3390/electronics10141715}
}
```
## Code description

* `rnn_ppg.ipynb` Google Colab file with training and testing algorithms
* `rnn_ppg.py` pure Python code exported from Colab notebook; used as an imported module by the following scripts when running tests on local computer
* `batch_test.py` script performing series of tests with different parameters
* `inspect_data.py` analyze original data with different preprocessing algorithms
* `create_csv_xcube.py` create data files with the right format for the STM32Cube.AI tool
* `parse_log.py` parse log files generated by tests, to compute mean and max values
* `batch_cross_6_1.py`, `batch_cross_6_1_test_only.py` perform leave-one-out cross-training tests
* `PPG_ACC_dataset` dataset directory; you must download the dataset and extract it in your working directory, see links in the article
* `downsample` directory with trained models for different decimation factors, named after number of window samples, number of overlap samples, epochs and test accuracy

Note that the main code can be run locally or from Google Colab. In the latter case the dataset directory must be on your Google Drive, where results are saved, too. Hopefully this and other details can be understood by inspecting the code.
