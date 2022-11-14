# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SDQ_module.cpp $(pkg-config opencv4 --cflags --libs) -o SDQ.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SDQ_module.cpp -o SDQ.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SDQ_OptD_module.cpp -o SDQ_OptD.so
# python3 test.py
