g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 -I/opt/opencv/ SDQ_module.cpp `pkg-config --cflags --libs opencv` -o SDQ.so

python3 test.py