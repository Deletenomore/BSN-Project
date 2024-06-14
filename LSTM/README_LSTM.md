## Create requirements.txt file & add some required libraries to install via pip
```bash
numpy
pandas
matplotlib
scikit-learn
```


## Getting Started

(1) Check if `pip` is installed:
```bash
$ pip --version

#If `pip` is not installed, follow steps below:
$ cd ~
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py
```

(2) Install virtual environment first & then activate:
```bash
$ cd <project-directory>
$ python3 -m pip install --user virtualenv #Install virtualenv if not installed in your system
$ python3 -m virtualenv env #Create virtualenv for your project
$ source env/bin/activate #Activate virtualenv for linux/MacOS
$ env\Scripts\activate #Activate virtualenv for Windows
```

(3) Install torch via pip by running following command (If possible always install torch via stable wheel they provide: https://pytorch.org/):
```bash
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

(4) Install all dependencies for your project from `requirement.txt` file:
```bash
$ pip install -r requirements.txt