
 ## Alexnet

Alexnet implementation of my own,including training (finetune) and test phase.

### Installation

1. Clone Alexnet repository
	```Shell
	$ git clone https://github.com/baobeila/Alexnet.git
    $ cd Alexnet
	```

2. Download the pretrained weights, and create correct directories
	```Shell
	http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
	```

3. put the pretrained weights in the working directory


4. Modify configuration in `alex/config.py`

5. Training
	```Shell
	$ python train.py
	```

6. Test
	```Shell
	$ python test.py
	```

### Requirements
1. Tensorflow-gpu 1.13.1

2. OpenCV 3.4.5
