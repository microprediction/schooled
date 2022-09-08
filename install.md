Don't use python 10!

      conda create -n schooled python=3.9
      conda activate schooled
      conda install pytorch torchvision torchaudio -c pytorch-nightly
      
Then virtual

      cd ~
      mkdir virtualenvs
      cd virtualenvs
      python -m venv schooled
      source schooled/bin/activate
      pip install --upgrade pip
      pip install --upgrade setuptools
      pip install --upgrade wheel
      
Then NNI. You have to [build from source](https://nni.readthedocs.io/en/stable/notes/build_from_source.html).

      cd ~
      mkdir github
      cd github
      git clone https://github.com/microsoft/nni.git
      cd nni
      pip install jupyterlab==3.0.9
      export NNI_RELEASE=2.0
      curl --compressed -o- -L https://yarnpkg.com/install.sh | bash
      python setup.py build_ts
      python setup.py bdist_wheel
      
      
      
