# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 HDQ_module.cpp $(pkg-config opencv4 --cflags --libs) -o HDQ.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 HDQ_module.cpp -o HDQ.so


g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 HDQ_OptD_module.cpp -o HDQ_OptD.so

# python3 test.py
