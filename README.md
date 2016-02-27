#WEB SERVER FOR DEEP NEURAL NETWORKS

##Step 1 – Install Linux, Python, Pip, Git
You can use any web server which includes a linux distribution. 
Python version 2.7 comes preinstalled on many Linux OS distributions, if it is not installed, it can be downloaded from https://www.python.org/downloads/
Pip comes preinstalled on many Linux OS distributions, if it is not installed, it can be downloaded from https://pip.pypa.io/en/stable/installing/
GIT comes preinstalled on many Linux OS distributions, if it is not installed, it can be downloaded using apt-get (example for Ubuntu) using:
    $ sudo apt-get install git-all 

##Step 2 – Clone GitHub repository
Navigate to your working directory, and then run:
    $ git clone https://github.com/yplatner/dnnserver
This will create a GIT repository with all the files needed to run a DNN server.

##Step 3 – Add requirements
Inside your working folder, run:
    $ pip install –r requirements.txt
This will install Django and Lasagne frameworks, and all pre-requirements.

##Step 4 – Run with integrated web server and database
In order to run on localhost port 8080, run the following command –
    $ python manage.py runserver 0.0.0.0:8080

##Step 5 (optional) – Install and run GUNUNICORN web server
Gunicorn (‘Green Unicorn’) is a pure-Python WSGI server for UNIX. It has no dependencies and is easy to install and use. To install it:
    $ pip install gununicorn
To run your server:
    $ gununicorn dnnserver.wsgi

##Step 6 (optional) – Install and run POSTGRES database
In order to install and configure postgres, run the following commands in your terminal:

    $ sudo apt-get install libpq-dev postgresql postgresql-contrib
    $ sudo su - postgres
    $ psql
    $ CREATE DATABASE myproject;
    $ CREATE USER <user> WITH PASSWORD '<password>';
    $ ALTER ROLE <user> SET client_encoding TO 'utf8';
    $ ALTER ROLE <user> SET default_transaction_isolation TO 'read committed';
    $ ALTER ROLE <user> SET timezone TO 'UTC';
    $ GRANT ALL PRIVILEGES ON DATABASE dnnserver TO <user>;
    $ \q
    $ Exit

Now we need to edit the Django settings file in order to enable us to connect to the POSTGRES database. The relevant file is: dnnserver/settings.py
Replace the following lines :
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
With :
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'dnnserver',
        'USER': '<user>',
        'PASSWORD': '<password>',
        'HOST': 'localhost',
        'PORT': '',
    }
}

Now, run the following commands :
    $ python manage.py makemigrations
    $ python manage.py migrate
    $ python manage.py createsuperuser
    <create user + password>
    $ python manage.py runserver 0.0.0.0:8000

##Step 7 (optional) – Enable GPU optimizations
Install the latest CUDA Toolkit and possibly the corresponding driver available from NVIDIA: https://developer.nvidia.com/cuda-downloads
Closely follow the Getting Started Guide linked underneath the download table to be sure you don’t mess up your system by installing conflicting drivers.
After installation, make sure /usr/local/cuda/bin is in your PATH. Also make sure /usr/local/cuda/lib64 is in your LD_LIBRARY_PATH, so the toolkit libraries can be found.
NVIDIA also provides a library for common neural network operations that especially speeds up Convolutional Neural Networks (CNNs). It can be obtained from NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn
To install it, copy the *.h files to /usr/local/cuda/include and the lib* files to /usr/local/cuda/lib64.
To configure Theano to use the GPU by default, create a file .theanorc directly in your home directory, with the following contents:
[global]
floatX = float32
device = gpu
Optionally add “allow_gc = False” for some extra performance at the expense of (sometimes substantially) higher GPU memory usage.
