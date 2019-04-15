## WaveGlow: a Flow-based Generative Network for Speech Synthesis

## Setup

1. Clone our repo and initialize submodule

    ```command
    git clone https://github.com/jinhan/waveglow.git
    cd waveglow
    git submodule init
    git submodule update
    ```

2. Install PyTorch 1.0 

3. Install other requirements `pip3 install -r requirements.txt`


## Train your own model

1. Prepare your dataset. In this example it's in `data/`

2. Make a list of the file names to use for training/testing

    ```command
    ls data/*.wav | tail -n+10 > train_files.txt
    ls data/*.wav | head -n10 > test_files.txt
    ```

3. Train your WaveGlow networks

    ```command
    mkdir checkpoints
    python train.py -c config.json
    ```

4. Train from checkpoint
    
    edit `"checkpoint_path": ""` from `config.json`

5. Make test set mel-spectrograms

    `python mel2samp.py -f test_files.txt -o . -c config.json`

6. Do inference with your network

    ```command
    ls *.pt > mel_files.txt
    python3 inference.py -f mel_files.txt -w checkpoints/waveglow_10000 -o . --is_fp16 -s 0.6
    ```
   
7. Do inference within Tacotron2 
   
    ```command
    import sys
    sys.path.append('waveglow/')

    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda()
    with torch.no_grad():
       synth = waveglow.infer(melspectrogram, sigma=0.666)
    ```
   

## config.json

- The model provided by NVIDIA's Repository is a model that is learned with a data sampled at 22kHz.
- At sampling rate of 16kHz, you have to start training from the beginning.

1. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
    * English
    * Sampling rate: 16000Hz
    * n_group: 8
    * batch_size: 1 (on Titan V 12GB)
    * `./models/ljspeech_16k_1790000`

2. Korean Emotion Speech Data
    * Korean Speech Emotion Dataset ([more info](http://aicompanion.or.kr/kor/main/))
    * Single Female Voice Actor recorded six diffrent emotions(neutral, happy, sad, angry, disgust, fearful), each with 3,000 sentences.
    * total 30 Hours
    * Sampling rate: 16000Hz
    * n_group: 16
    * batch_size: 14 (on Tesla P40 24GB)
    * `./models/koemo_16k_150000`
    
    
## Pre-trained Models
path: `./models/`

| Name  | Dataset | Note | epochs
| - | -| - | - |
| waveglow_old.pt | LJSpeech(22kHz) | "n_group": 8, "n_channels":512 ||
| waveglow_256channels.pt | LJSpeech(22kHz) | "n_group": 8, "n_channels":256 ||
| ljspeech_16k_1790000 | LJSpeech(16kHz) | "n_group": 8, "n_channels":512 | 137 |
| koemo_16k_150000 | KoreanEmotionSpeech(16kHz) | "n_group": 16, "n_channels":512 | 117 |


## Inference Time
Based on 'waveglow_old.pt', the average of the generation time with [10 melspectrogram files]((https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing)(mean play time: 6s)

| GPU  | Time |
| - | - |
| GTX 1060  | 50.6s  |
| Tesla P40  | 20.4s  |
| TITAN V | 0.507s |

## References
- [paper]
- [code] 
- [samples]


[code]: https://github.com/NVIDIA/waveglow
[website]: https://nv-adlr.github.io/WaveGlow
[paper]: https://arxiv.org/abs/1811.00002
